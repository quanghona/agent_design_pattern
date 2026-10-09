# LangChain RetrieverAdapter

## Overview

The LangChain `RetrieverAdapter` provides a bridge between the AI Agent Pattern core framework and [LangChain](https://python.langchain.com/)'s retrieval abstractions. It wraps any LangChain `BaseRetriever` (e.g., `FAISSRetriever`, `VectorStoreRetriever`, `EmbeddingsRetriever`) and adapts it to the framework-agnostic `BaseRetriever` interface, enabling seamless integration of LangChain-powered retrieval into agents and chains.

## Architecture

The `RetrieverAdapter` sits in the integration layer, translating between the core `AgentMessage` type and LangChain's document model. It receives an `AgentMessage`, extracts the query, invokes the wrapped LangChain retriever, and stores the retrieved document content in the message's `context` dictionary under a configurable key.

The adapter integrates with LangChain's core retrieval abstractions:

- **`BaseRetriever`** (LangChain): The abstract base class for all LangChain retrievers. Any concrete retriever (FAISS, Chroma, Pinecone, etc.) can be wrapped.
- **`BaseDocument`**: LangChain's document type containing `page_content` and `metadata`. The adapter extracts `page_content` from each document.

![diagram](img/retriever-architecture.jpg)

## Key Concepts

### BaseRetriever: The Retrieval Contract

`RetrieverAdapter` inherits from `aap_core.retriever.BaseRetriever`, which defines the contract for all retrievers in the AI Agent Pattern framework. The base class provides:

- **`retrieve(message, **kwargs)`**: Implemented by `RetrieverAdapter` to invoke the wrapped LangChain retriever.
- **`post_process`**: An optional `BaseChain` that runs after `retrieve()`, enabling post-processing steps such as reranking or summarization.

The `__call__` method orchestrates the pipeline: it first calls `retrieve()`, then applies `post_process` if configured, and returns the final `AgentMessage`.

### Context Key Management

The `data_key` field controls where retrieved data is stored in the message context. The key must start with the `"context."` prefix, which is enforced by a Pydantic validator:

```python
# data_key = "context.data" → stored at message.context["data"]
# data_key = "context.langchain_docs" → stored at message.context["langchain_docs"]
```

This design allows multiple retrievers to store data under different keys in the same message context, enabling complex multi-source retrieval scenarios.

## Usage

### Basic Example: FAISS Retriever

```python
from aap_langchain.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Set up embedding and vector store
embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.create_documents(["Your document text here"])
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Wrap with RetrieverAdapter
adapter = RetrieverAdapter(retriever=retriever)

# Use the adapter on a message
message = AgentMessage(query="What is this document about?")
result = adapter(message)

print(result.context["data"])  # List of retrieved document chunks
```

### Basic Example: Custom Retriever

```python
from aap_langchain.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class CustomRetriever(BaseRetriever):
    """A simple custom retriever that returns hardcoded documents."""

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return [Document(page_content=f"Result for: {query}")]


adapter = RetrieverAdapter(retriever=CustomRetriever())
message = AgentMessage(query="Find relevant info")
result = adapter(message)
print(result.context["data"])
```

### Advanced Example: Multi-Source Retrieval with Custom Keys

```python
from aap_langchain.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Two retrievers with different data keys
products_retriever = RetrieverAdapter(
    retriever=FAISS.from_documents(..., embeddings).as_retriever(),
    data_key="context.products",
)

reviews_retriever = RetrieverAdapter(
    retriever=FAISS.from_documents(..., embeddings).as_retriever(),
    data_key="context.reviews",
)

# Chain both retrievers
message = AgentMessage(query="Compare products")
message = products_retriever(message)
message = reviews_retriever(message)

# Both datasets are available in the context
print(message.context["products"])  # Product documents
print(message.context["reviews"])   # Review documents
```

### Advanced Example: Retrieval with Post-Processing

```python
from aap_langchain.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from aap_core import BaseChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class SummarizingPostProcess(BaseChain):
    """A post-processor that summarizes retrieved data."""

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        data = message.context.get("data", [])
        message.context["summary"] = f"[Summarized: {len(data)} documents retrieved]"
        return message


adapter = RetrieverAdapter(
    retriever=FAISS.from_documents(..., embeddings).as_retriever(),
)
adapter.post_process = SummarizingPostProcess()

message = AgentMessage(query="Show me relevant info")
result = adapter(message)

print(result.context["data"])     # Raw retrieved documents
print(result.context["summary"])  # Post-processed summary
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retriever` | `BaseRetriever` | *(required)* | The LangChain retriever to wrap (e.g., `FAISSRetriever`, `VectorStoreRetriever`). |
| `data_key` | `str` | `"context.data"` | The key under which retrieved data is stored in `message.context`. Must start with `"context."`. |

## API Reference

::: aap_langchain.retriever.RetrieverAdapter
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Retriever](../../../core/components/retriever.md) — The base retriever concept and `BaseRetriever` interface
- [LangChain Overview](../overview.md) — LangChain integration overview
- [LangChain Documentation](https://python.langchain.com) — Official LangChain documentation
