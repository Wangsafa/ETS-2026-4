import os
import glob
from models import gpt4o
from pathlib import Path
from langchain_core.documents import Document
import re
import cohere


from langchain_community.vectorstores import FAISS #
from langchain_community.retrievers import BM25Retriever#4ss
from langchain_community.embeddings import HuggingFaceEmbeddings



def rag_find(vector_dataset, assertion_spec, score):
    text_retriever = vector_dataset.as_retriever(search_kwargs = {'k':3})
    visited = []
    filtered_docs = []
    query = f"What are the evidences that: {assertion_spec}"
    display = text_retriever.invoke(query)
    rerank_display = [{'text': doc.page_content} for doc in display]
    response = co.rerank(query=query, documents=rerank_display, model="rerank-english-v3.0")
    for doc in response.results:
        if doc.relevance_score > score:
            filtered_docs.append(doc)
            visited.append(doc['document']['id'])
    final_context = "\n".join([doc.document['text'] for doc in filtered_docs])

    return final_context, visited

def rag_find_new(sub_specs, assertion_spec, score):
    # text_retriever = vector_dataset.as_retriever(search_kwargs = {'k':3})
    # visited = []
    # filtered_docs = []
    query = f"What are the evidences that: {assertion_spec}"
    # display = text_retriever.invoke(query)
    # rerank_display = [{'text': doc.page_content} for doc in display]
    response = co.rerank(query=query, documents=sub_specs, model="rerank-english-v3.0")
    import time
    time.sleep(12)
    find_idx = []
    for doc in response.results:
        if doc.relevance_score > score:
            find_idx.append(doc.index)

    return find_idx

def mapping(sub_specs, assertion_spec, score):
    has_find_in_rag = []
    spec_has_visited = [False for _ in assertion_spec]
    embedding =HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    sub_spec_documents = [Document(page_content=spec) for spec in sub_specs]
    vs = FAISS.from_documents(sub_spec_documents,embedding)
    has_find_in_rag = [rag_find(vs, a_spec, score) for a_spec in assertion_spec]
    for has_find in has_find_in_rag:
        for spec_idx in has_find:
            spec_has_visited[spec_idx] = True
    has_not_mapping = [spec for (spec, visited) in zip(sub_specs, spec_has_visited) if not visited]

    return has_not_mapping

def mapping_new(sub_specs, assertion_spec, score):
    has_find_in_rag = []
    spec_has_visited = [False for _ in sub_specs]
    has_find_in_rag = [rag_find_new(sub_specs, a_spec, score) for a_spec in assertion_spec]
    for finds in has_find_in_rag:
        for find_idx in finds:
            spec_has_visited[find_idx] = True
    has_not_mapping = [spec for (spec, visited) in zip(sub_specs, spec_has_visited) if not visited]

    return has_not_mapping


def group_to_as(groups:list[str]):
    assertion_specs = []
    prompt_file_path = "/prompt/group_function.txt"  
    question = Path(prompt_file_path).read_text()
    system_message = "You're an experienced hardware validator, and your job is to summarize the overall function of multiple assertions."
    for group in groups:
        context = 'Assertion group:\n' + group
        user_content = f"Please generate an overall description of the following assertions in detail: {group}"
        # resp = deepseek.generate_response(system_message, context, user_content)
        resp = gpt4o.generate_response(system_message, context, question)
        assertion_specs.append(resp)
    return assertion_specs

def main():
    group_path = "/design_name/group/"
    sub_spec_dir = "/sub-spec/design_name/"
    group_files = glob.glob(group_path + "*.txt")
    # read all groups
    groups = []
    for group_file in group_files:
        with open(group_file, "r") as f1:
            group = f1.read()
            groups.append(group)
    sub_specs = []
    for sub_spec_file in glob.glob(sub_spec_dir + "*.txt"):
        with open(sub_spec_file, "r") as f2:
            sub_spec = f2.read()
            sub_specs.append(sub_spec)

    assertion_spec = group_to_as(groups)

    score = 0.5

    x = mapping_new(sub_specs, assertion_spec, score)

if __name__ == "__main__":
    main()
