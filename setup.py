from setuptools import setup

setup(
    name="HuggingRAG",
    version="0.0.8",
    description="High-Level Library for RAG task with HuggingFace API",
    url="https://github.com/Cafelatte1/hugging-rag.git",
    author="Cafelatte1",
    license="MIT",
    python_requires='>=3.10',
    zip_safe=False,
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'sentencepiece',
        'transformers',
        'bitsandbytes',
        'accelerate',
        'langchain',
        'openai',
        'faiss-cpu',
        'scann',
    ],
)