# ETS-2026-#4
CoverAssert

## Project Overview
The CoverAssert project aims to extract features from hardware verification assertions, cluster them, and map them to sub-specifications to enhance validation efficiency and accuracy.

## Files

### CoverAssert_Feature Extractor and Cluster.py
- **Purpose**: Extracts assertions from SystemVerilog files, generates semantic features, and performs clustering.
- **Dependencies**: `tree_sitter_verilog`, `transformers`, `sklearn`.
- **Usage**: Run the script with specified RTL directory, assertion file, and output directory.

### CoverAssert_Mapper.py
- **Purpose**: Maps clustered assertions to sub-specifications using semantic similarity.
- **Dependencies**: `cohere`, `langchain_community`, `HuggingFaceEmbeddings`.
- **Usage**: Run the script with directories containing assertion groups and sub-specifications.
