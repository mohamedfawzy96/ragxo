# RagXO 🚀

[![PyPI version](https://badge.fury.io/py/ragxo.svg)](https://badge.fury.io/py/ragxo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

RagXO extends the capabilities of traditional RAG (Retrieval-Augmented Generation) systems by providing a unified way to package, version, and deploy your entire RAG pipeline with LLM integration. Export your complete system—including embedding functions, preprocessing steps, vector store, and LLM configurations—into a single, portable artifact.

## Features ✨

- **Complete RAG Pipeline**: Package your entire RAG system into a versioned artifact
- **LLM Integration**: Built-in support for OpenAI models
- **Flexible Embedding**: Compatible with any embedding function (Sentence Transformers, OpenAI, etc.)
- **Custom Preprocessing**: Chain multiple preprocessing steps
- **Vector Store Integration**: Built-in Milvus support
- **System Prompts**: Include and version your system prompts

## Installation 🛠️

```bash
pip install ragxo
```

## Usage Guide 📚

### Import

```python
from ragxo import Ragxo, Document

ragxo_client = Ragxo(dimension=768)

```

### Adding Preprocessing Steps

```python
import re

def remove_special_chars(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def lowercase(text: str) -> str:
    return text.lower()

ragxo_client.add_preprocess(remove_special_chars)
ragxo_client.add_preprocess(lowercase)
```

### Custom Embedding Functions

```python
# Using SentenceTransformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(text: str) -> list[float]:
    return model.encode(text).tolist()

ragxo.add_embedding_fn(get_embeddings)

# Or using OpenAI
from openai import OpenAI
client = OpenAI()

def get_openai_embeddings(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

ragxo.add_embedding_fn(get_openai_embeddings)
```


### Creating Documents

```python
from ragxo import Document

doc = Document(
    text="Your document content here",
    metadata={"source": "wiki", "category": "science"},
    id=1
)

ragxo_client.index([doc])

```

### LLM Configuration

```python
# Set system prompt
ragxo_client.add_system_prompt("""
You are a helpful assistant. Use the provided context to answer questions accurately.
If you're unsure about something, please say so.
""")

# Set LLM model
ragxo_client.add_model("gpt-4")
```

### Export and Load

```python
# Export your RAG pipeline
ragxo_client.export("rag_pipeline_v1")

# Load it elsewhere
loaded_ragxo_client = Ragxo.load("rag_pipeline_v1")
```

## Best Practices 💡

1. **Version Your Exports**: Use semantic versioning for your exports:
```python
ragxo.export("my_rag_v1.0.0")
```

2. **S3**: Use S3 to store your exports

```shell
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

```python
ragxo_client.export("my_rag_v1.0.0", s3_bucket="my_bucket")
```

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.