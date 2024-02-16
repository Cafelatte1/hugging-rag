# HuggingRAG
High-Level Library for RAG task with HuggingFace API

# Structure
## Data Control Modules
### 1. Vector Data Container
This module migrate and controls raw data as HuggingRAG format with Pandas library.<br>
Chunks of documents also includes to this module.
### 2. Vector Embedding
This module extract embedding vector from chunks.
### 3. Vector Store
This module stores extracted vector and searches K-Nearest Neighbors with FAISS or ScaNN library.
### 4. Vector Ranker
This module scores embedding vectors(chunks) and aggregate the score by documents.<br>
The supported algorithms in aggregating functions are 'first_matching', 'equal_weighted' and 'exponential_weighted'.
## Generation Modules
### 1. HuggingFaceAPI
This module provide generation function with hugginface model. (Batch-generation is also supported)
### 2. ChatGPTAPI
This module provide generation function with chatgpt.

# Tutorials
1. HuggingFace API  
The core function of HuggingRAG is to make it easy to use huggingface models for RAG task.  
[Quickstart in Colab](https://colab.research.google.com/drive/1B56CaYywB1FZUp2a566i8bOyoLvvPnd8?usp=sharing)
2. ChatGPT API  
HuggingRAG also support ChatGPT API.  
[Quickstart in Colab](https://colab.research.google.com/drive/1oZLkRW4YYqHPSXXM3XJWoXMeWhPyxIa2?usp=sharing)
