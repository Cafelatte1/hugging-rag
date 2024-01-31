from setuptools import setup

setup(
    name="HuggingRAG",
    version="0.0.5",
    description="High-level library for RAG task with huggingface API",
    url="https://github.com/Cafelatte1/hugging-rag",
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
    ],
)