from __future__ import annotations
import json, re, os, math, itertools
from pathlib import Path
from typing import List, Dict, Tuple, Set
import tree_sitter_verilog as tsv
from tree_sitter import Language, Parser, Node
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder  
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from transformers import AutoTokenizer, AutoModelForCausalLM

from collections import defaultdict

LANG = Language(tsv.language())
parser = Parser(LANG)

def gpt_describe(assertion: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Given a SystemVerilog assertion, output a single concise sentence describing its purpose. Do not include code."},
            {"role": "user", "content": assertion}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

class QwenEmbed:
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), 8):
            batch = texts[i:i+8]
            tok = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            last_hidden = self.model(**tok, output_hidden_states=True).hidden_states[-1]
            vecs.append(last_hidden.mean(dim=1).cpu().numpy())
        return np.vstack(vecs)

QWEN = QwenEmbed()

def real_signal(name: str) -> str:
    name = re.sub(r'\[.*?\]', '', name)
    return name.split('.')[-1]

def load_all_v(root: Path) -> Tuple[bytes, Dict[int, str]]:
    pieces, line2file = [], {}
    cur_line = 1
    for f in sorted(root.rglob("*.v")):
        src = f.read_bytes()
        pieces.append(src)
        pieces.append(b"\n")
        for _ in range(src.count(b"\n")):
            line2file[cur_line] = str(f)
            cur_line += 1
    return b"".join(pieces), line2file

def build_node_map(src: bytes) -> Tuple[Dict[str, List[Node]], Node]:
    tree = parser.parse(src)
    root = tree.root_node
    nodes: Dict[str, List[Node]] = {}
    def dfs(n: Node):
        if n.type in ("identifier", "simple_identifier", "hierarchical_identifier", "indexed_identifier"):
            full = src[n.start_byte:n.end_byte].decode()
            key = real_signal(full)
            nodes.setdefault(key, []).append(n)
        for ch in n.children:
            dfs(ch)
    dfs(root)
    return nodes, root

class Assertion:
    def __init__(self, raw: str, vars: Set[str], intent: str, start_line: int):
        self.raw, self.vars, self.intent, self.start_line = raw, vars, intent, start_line

def extract_assertions(file: Path) -> List[Assertion]:
    lines = file.read_text().splitlines(keepends=True)
    pattern = re.compile(r'^\s*assert\s+property\b', re.I)
    sig_re = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+(?:\[[^\]]+\])?\b')
    assertions = []
    i = 0
    while i < len(lines):
        if pattern.match(lines[i]):
            start = i
            stack = 0
            for j in range(i, len(lines)):
                for ch in lines[j]:
                    if ch in "({[":
                        stack += 1
                    elif ch in ")}]":
                        stack -= 1
                    elif ch == ";" and stack == 0:
                        end = j + 1
                        raw = "".join(lines[start:end])
                        vars = {real_signal(m) for m in sig_re.findall(raw)}
                        intent = gpt_describe(raw)
                        assertions.append(Assertion(raw, vars, intent, start + 1))
                        i = end - 1
                        break
                else:
                    continue
                break
        i += 1
    return assertions

def lca_distance(a: Node, b: Node) -> int:
    def ancestors(n: Node):
        path = []
        while n:
            path.append(n)
            n = n.parent
        return path
    pa = ancestors(a)
    pb = ancestors(b)
    common = None
    for x, y in zip(reversed(pa), reversed(pb)):
        if x == y:
            common = x
        else:
            break
    if common is None:
        return 1000
    return abs(pa.index(common) - pb.index(common))

def semantic_features(records: List[Assertion]) -> np.ndarray:
    intents = [r.intent for r in records]
    T = QWEN.encode(intents)
    return T.astype(np.float32)


def semantic_distance_matrix(T: np.ndarray) -> np.ndarray:
    return cosine_distances(T).astype(np.float32)

def build_SD_matrix(records: List[Assertion],
                    src_bytes: bytes,
                    node_map: Dict[str, List[Node]],
                    root_node: Node) -> np.ndarray:
    N = len(records)
    SD = np.zeros((N, N), dtype=np.float32)
    sig2nodes = {k: nl for k, nl in node_map.items()}

    for i in range(N):
        for j in range(i, N):
            Si, Sj = records[i].vars, records[j].vars
            if not Si or not Sj:
                dist = 1_000.0
            else:
                total = 0.0
                for v in Si:
                    for u in Sj:
                        cand = []
                        for nv in sig2nodes.get(v, []):
                            for nu in sig2nodes.get(u, []):
                                if nv is nu:              
                                    continue
                                lca = find_lca(nv, nu)      
                                d_s = depth_from_root(nv, root_node)
                                d_t = depth_from_root(nu, root_node)
                                d_lca = depth_from_root(lca, root_node)
                                raw_d = abs(d_s - d_lca) + abs(d_t - d_lca)
                                cand.append(raw_d)
                        total += min(cand) if cand else 1_000.0
                dist = total / (len(Si) * len(Sj))
            SD[i, j] = SD[j, i] = dist
    return SD


def build_path_matrix(records: List[Assertion], src_bytes: bytes, node_map: Dict[str, List[Node]], root_node: Node) -> np.ndarray:
    all_paths = []        
    max_len = 0
    for rec in records:
        for v in sorted(rec.vars):                  
            for node in node_map.get(v, []):
                path = ast_path_node_types(node, root_node)
                all_paths.append(path)
                max_len = max(max_len, len(path))
    if max_len == 0:     
        return np.zeros((len(records), 1), dtype=np.float32)
    flat_types = [ty for p in all_paths for ty in p]
    encoder = LabelEncoder()
    encoder.fit(flat_types)
    Q_list = []
    for rec in records:
        vec = []
        for v in sorted(rec.vars):
            for node in node_map.get(v, []):
                path = ast_path_node_types(node, root_node)
                vec.extend(encoder.transform(path).tolist())
        Q_list.append(np.array(vec, dtype=np.int32))
    max_dim = max(len(v) for v in Q_list) if Q_list else 1
    Q = np.zeros((len(records), max_dim), dtype=np.float32)
    for i, v in enumerate(Q_list):
        Q[i, :len(v)] = v
    mu  = Q.mean(0)
    sig = Q.std(0) + 1e-8
    Q   = (Q - mu) / sig
    return Q
def first_cluster(T: np.ndarray, eps_sem: float = 0.8) -> np.ndarray:
    cls = DBSCAN(eps=eps_sem, min_samples=2, metric="cosine").fit(T)
    return cls.labels_
def fuse(T: np.ndarray, Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    mask = C != -1
    oh = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(C[mask].reshape(-1, 1))
    K = oh.shape[1]
    d_Q = Q.shape[1]               
    scale = d_Q / K                 

    Qprime = np.zeros((Q.shape[0], d_Q + K), dtype=np.float32)
    Qprime[mask, :d_Q] = Q[mask]
    Qprime[mask, d_Q:] = oh * scale  
    Qprime[~mask, :d_Q] = Q[~mask]
    return Qprime
def final_cluster(Qprime: np.ndarray, affinity: np.ndarray) -> np.ndarray:
    n_samples = Qprime.shape[0]
    max_k = min(6, n_samples - 1)           
    k_range = range(2, max_k + 1)

    # 1. 粗聚类（Spectral）
    best_sc, best_coarse = -1, None
    for k in k_range:
        coarse = SpectralClustering(n_clusters=k, affinity='precomputed',
                                    random_state=42).fit_predict(affinity)
        if len(set(coarse)) > 1:
            sc = silhouette_score(Qprime, coarse, metric='euclidean')
            if sc > best_sc:
                best_sc, best_coarse = sc, coarse
    if best_coarse is None:
        return np.full(n_samples, -1)

    global_lab = np.full(n_samples, -1, dtype=int)
    next_lbl = 0
    for cid in range(best_coarse.max() + 1):
        mask = best_coarse == cid
        idx = np.where(mask)[0]
        sub_Q = Qprime[mask]

        dist = pairwise_distances(sub_Q, metric='euclidean')
        best_sub = None
        for eps in np.logspace(-4, -2, 8):         
            sub_lab = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit_predict(dist)
            if len(set(sub_lab)) > 1:
                best_sub = sub_lab
                break
        if best_sub is None:                   
            best_sub = np.zeros(len(idx), dtype=int)

        uniq, cnt = np.unique(best_sub, return_counts=True)
        if np.min(cnt[uniq != -1]) < 2:
            best_sub = np.zeros(len(idx), dtype=int)

        best_sub[best_sub != -1] += next_lbl
        global_lab[idx] = best_sub
        next_lbl += (best_sub.max() + 1)
    return global_lab

def depth_from_root(node: Node, root: Node) -> int:
    d = 0
    cur = node
    while cur and cur is not root.parent:
        d += 1
        cur = cur.parent
    return d

def find_lca(a: Node, b: Node) -> Node:
    if a is b:
        return a
    path_a = {}
    cur = a
    while cur:
        path_a[cur] = True
        cur = cur.parent
    cur = b
    while cur:
        if path_a.get(cur):
            return cur
        cur = cur.parent
    return a.tree.root_node


def _tau_split(SD: np.ndarray, tau: float) -> List[List[int]]:
    from networkx import Graph, find_cliques  

    n = SD.shape[0]
    G = Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if SD[i, j] < tau:
                G.add_edge(i, j)
    cliques = list(find_cliques(G))   
    covered = set()
    blocks = []
    for clq in sorted(cliques, key=len, reverse=True):
        clq = list(set(clq) - covered)
        if clq:                      
            blocks.append(clq)
            covered.update(clq)
    for u in range(n):
        if u not in covered:
            blocks.append([u])
    return blocks


def child_index(node: Node) -> int:
    if node.parent is None:
        return 0
    for idx, sibling in enumerate(node.parent.children):
        if sibling is node:
            return idx
    return 0   


def ast_path_step(node: Node) -> str:
    return f"{node.type}@{child_index(node)}"

def ast_path_node_types(node: Node, root: Node) -> List[str]:
    path = []
    cur = node
    while cur and cur is not root.parent:
        path.append(ast_path_step(cur))
        cur = cur.parent
    return path[::-1]  

def cluster(records: List[Assertion],
            rtl_dir: Path,        
            src_bytes: bytes,
            node_map: dict,
            root_node: Node) -> List[int]:
    return cluster_with_tau_filter(
        records,                 
        src_bytes,              
        node_map,               
        root_node,            
        tau=15               
    )

def cluster_with_tau_filter(records, src_bytes, node_map, root_node, *, tau: float = 15):
    n_total = len(records)
    T = semantic_features(records)
    Q = build_path_matrix(records, src_bytes, node_map, root_node)
    pca = PCA(n_components=20, svd_solver='full', random_state=42)
    Q64 = pca.fit_transform(Q)
    C = first_cluster(T)
    Qprime = fuse(T, Q, C)
    SD = build_SD_matrix(records, src_bytes, node_map, root_node)
    blocks = _tau_split(SD, tau)
    global_labels = np.full(n_total, -1, dtype=int)
    next_lbl = 0
    for block in blocks:
        if len(block) == 1:                
            global_labels[block[0]] = next_lbl
            next_lbl += 1
            continue
        sub_Q = Qprime[block]
        sub_SD = SD[np.ix_(block, block)]
        affinity = np.exp(-sub_SD)
        affinity[sub_SD >= tau] = 1e-6
        np.fill_diagonal(affinity, 1.0)
        sub_lab = final_cluster(sub_Q, affinity)  
        sub_lab = np.array(sub_lab)
        sub_lab[sub_lab != -1] += next_lbl
        global_labels[block] = sub_lab
        next_lbl += (sub_lab.max() + 1)
    return global_labels.tolist()

def write_groups(records: List[Assertion], labels: List[int], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.txt"):
        old.unlink()
    clusters: Dict[int, List[Tuple[int, Assertion]]] = {}
    for idx, (rec, lab) in enumerate(zip(records, labels)):
        clusters.setdefault(lab, []).append((idx, rec))
    for cid, indexed_recs in clusters.items():
        indexed_recs.sort(key=lambda t: t[0])
        lines = []
        for original_idx, rec in indexed_recs:
            lines.append(f"// assertion-{original_idx}\n")
            lines.append(rec.raw)
            if not rec.raw.endswith("\n"):
                lines.append("\n")
            lines.append("\n")
        fname = out_dir / f"group_{cid+1:02d}.txt"
        fname.write_text("".join(lines))

def main(rtl_dir: str, assert_file: str, out_dir: str):
    rtl_root    = Path(rtl_dir)
    assert_path = Path(assert_file)
    out_root    = Path(out_dir)
    src, _      = load_all_v(rtl_root)
    node_map, root_node = build_node_map(src)   
    records     = extract_assertions(assert_path)
    if not records:
        return
    labels = cluster(records, rtl_root, src, node_map, root_node)
    write_groups(records, labels, out_root)
    
    
if __name__ == '__main__':
    RTL_DIR     = "/rtl/design_name"
    ASSERT_FILE = "/assertion/design_name/design_assertion.txt"
    OUT_DIR     = "/result/group"
    main(RTL_DIR, ASSERT_FILE, OUT_DIR)