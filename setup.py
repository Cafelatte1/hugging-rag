from setuptools import setup

setup(
    name="hugging-rag",
    version="0.0.1",
    description="High-level library for RAG task with huggingface API",
    url="https://github.com/Cafelatte1/hugging-rag",
    author="Cafelatte",
    license="MIT",
    python_requires='>=3.10',
    zip_safe=False,
    install_requires=[
        'numpy==1.25.2',
        'pandas==2.1.4',
        'tqdm',
        'transformers',
        'sentencepiece',
        'bitsandbytes',
        'git+https://github.com/huggingface/peft.git',
        'git+https://github.com/huggingface/accelerate.git',
        'langchain',
    ],
)