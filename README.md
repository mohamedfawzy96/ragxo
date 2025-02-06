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

## Quick Start 🚀

```python
from ragxo import Ragxo, Document
from openai import OpenAI
client = OpenAI()

def get_openai_embeddings(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def preprocess_text(text: str) -> str:
    return text.lower()

# Initialize and configure RagXO
ragxo = Ragxo(dimension=384)
ragxo.add_preprocess(preprocess_text)
ragxo.add_embedding_fn(get_openai_embeddings)

# Add system prompt and model
ragxo.add_system_prompt("You are a helpful assistant.")
ragxo.add_model("gpt-4o-mini")

# Create and index documents
documents = [
    Document(
        text="Sample document for indexing",
        metadata={"source": "example"},
        id=1
    )
]
ragxo.index(documents)

# Export the pipeline
ragxo.export("my_rag_v1")

# Load and use elsewhere
loaded_ragxo = Ragxo.load("my_rag_v1")

# Query and generate response
similar_docs = loaded_ragxo.query("sample query")
llm_response = loaded_ragxo.generate_llm_response("What can you tell me about the sample?")
```

## Usage Guide 📚

### Creating Documents

```python
from ragxo import Document

doc = Document(
    text="Your document content here",
    metadata={"source": "wiki", "category": "science"},
    id=1
)
```

### Adding Preprocessing Steps

```python
import re

def remove_special_chars(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def lowercase(text: str) -> str:
    return text.lower()

ragxo.add_preprocess(remove_special_chars)
ragxo.add_preprocess(lowercase)
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

### LLM Configuration

```python
# Set system prompt
ragxo.add_system_prompt("""
You are a helpful assistant. Use the provided context to answer questions accurately.
If you're unsure about something, please say so.
""")

# Set LLM model
ragxo.add_model("gpt-4")
```

### Export and Load

```python
# Export your RAG pipeline
ragxo.export("rag_pipeline_v1")

# Load it elsewhere
loaded_ragxo = Ragxo.load("rag_pipeline_v1")
```

## Best Practices 💡

1. **Version Your Exports**: Use semantic versioning for your exports:
```python
ragxo.export("my_rag_v1.0.0")
```

2. **Validate After Loading**: Always test your loaded pipeline:
```python
loaded_ragxo = Ragxo.load("my_rag")
try:
    # Test similarity search
    similar_docs = loaded_ragxo.query("test query")
    # Test LLM generation
    llm_response = loaded_ragxo.generate_llm_response("test question")
    print("Pipeline loaded successfully!")
except Exception as e:
    print(f"Error loading pipeline: {e}")
```

3. **Document Your Pipeline Configuration**: Keep track of your setup:
```python
pipeline_config = {
    "preprocessing_steps": ["remove_special_chars", "lowercase"],
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "gpt-4",
    "dimension": 384
}
```

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.