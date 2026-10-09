# LlamaIndex RetrieverAdapter

## Overview

The LlamaIndex `RetrieverAdapter` provides a bridge between the AI Agent Pattern core framework and [LlamaIndex](https://docs.llamaindex.ai/)'s retrieval abstractions. It wraps any LlamaIndex `BaseRetriever` (e.g., `VectorStoreRetriever`, `KnowledgeGraphRetriever`, `IPythonRetriever`) and adapts it to the framework-agnostic `BaseRetriever` interface, enabling seamless integration of LlamaIndex-powered retrieval into agents and chains.

## Architecture

The `RetrieverAdapter` sits in the integration layer, translating between the core `AgentMessage` type and LlamaIndex's node-based retrieval model. It receives an `AgentMessage`, extracts the query, invokes the wrapped LlamaIndex retriever, and stores the retrieved node text content in the message's `context` dictionary under a configurable key.

The adapter integrates with LlamaIndex's core retrieval abstractions:

- **`BaseRetriever`** (LlamaIndex): The abstract base class for all LlamaIndex retrievers. Any concrete retriever (vector store, knowledge graph, etc.) can be wrapped.
- **`BaseNode`**: LlamaIndex's node type containing text content accessible via `get_text()`. The adapter extracts text from each node.

![diagram](img/retriever-architecture.jpg)

## Key Concepts

### BaseRetriever: The Retrieval Contract

`RetrieverAdapter` inherits from `aap_core.retriever.BaseRetriever`, which defines the contract for all retrievers in the AI Agent Pattern framework. The base class provides:

- **`retrieve(message, **kwargs)`**: Implemented by `RetrieverAdapter` to invoke the wrapped LlamaIndex retriever.
- **`post_process`**: An optional `BaseChain` that runs after `retrieve()`, enabling post-processing steps such as reranking or summarization.

The `__call__` method orchestrates the pipeline: it first calls `retrieve()`, then applies `post_process` if configured, and returns the final `AgentMessage`.

### Context Key Management

The `data_key` field controls where retrieved data is stored in the message context. The key must start with the `"context."` prefix, which is enforced by a Pydantic validator:

```python
# data_key = "context.data" → stored at message.context["data"]
# data_key = "context.llamaindex_nodes" → stored at message.context["llamaindex_nodes"]
```

This design allows multiple retrievers to store data under different keys in the same message context, enabling complex multi-source retrieval scenarios.

## Usage

### Basic Example: Vector Store Retriever

```python
from aap_llamaindex.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

# Set up LlamaIndex vector store
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=3)

# Wrap with RetrieverAdapter
adapter = RetrieverAdapter(retriever=retriever)

# Use the adapter on a message
message = AgentMessage(query="What is this document about?")
result = adapter(message)

print(result.context["data"])  # List of retrieved node texts
```

### Basic Example: Knowledge Graph Retriever

```python
from aap_llamaindex.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from llama_index.core import VectorStoreIndex, KnowledgeGraphIndex
from llama_index.core.storage.storage_context import StorageContext

# Set up knowledge graph retriever
storage_context = StorageContext.from_defaults()
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_kg_edges=3,
)
retriever = kg_index.as_retriever()

# Wrap with RetrieverAdapter
adapter = RetrieverAdapter(retriever=retriever)

# Use the adapter on a message
message = AgentMessage(query="What entities are related?")
result = adapter(message)

print(result.context["data"])  # Retrieved knowledge graph node texts
```

### Advanced Example: Multi-Source Retrieval with Custom Keys

```python
from aap_llamaindex.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from llama_index.core import VectorStoreIndex

# Two retrievers with different data keys
products_retriever = RetrieverAdapter(
    retriever=VectorStoreIndex.from_documents(...).as_retriever(),
    data_key="context.products",
)

reviews_retriever = RetrieverAdapter(
    retriever=VectorStoreIndex.from_documents(...).as_retriever(),
    data_key="context.reviews",
)

# Chain both retrievers
message = AgentMessage(query="Compare products")
message = products_retriever(message)
message = reviews_retriever(message)

# Both datasets are available in the context
print(message.context["products"])  # Product node texts
print(message.context["reviews"])   # Review node texts
```

### Advanced Example: Retrieval with Post-Processing

```python
from aap_llamaindex.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from aap_core import BaseChain
from llama_index.core import VectorStoreIndex


class SummarizingPostProcess(BaseChain):
    """A post-processor that summarizes retrieved data."""

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        data = message.context.get("data", [])
        message.context["summary"] = f"[Summarized: {len(data)} nodes retrieved]"
        return message


adapter = RetrieverAdapter(
    retriever=VectorStoreIndex.from_documents(...).as_retriever(),
)
adapter.post_process = SummarizingPostProcess()

message = AgentMessage(query="Show me relevant info")
result = adapter(message)

print(result.context["data"])     # Raw retrieved node texts
print(result.context["summary"])  # Post-processed summary
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retriever` | `BaseRetriever` | *(required)* | The LlamaIndex retriever to wrap (e.g., `VectorStoreRetriever`, `KnowledgeGraphRetriever`). |
| `data_key` | `str` | `"context.data"` | The key under which retrieved data is stored in `message.context`. Must start with `"context."`. |

## API Reference

::: aap_llamaindex.retriever.RetrieverAdapter
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Retriever](../../../core/components/retriever.md) — The base retriever concept and `BaseRetriever` interface
- [LlamaIndex Overview](../overview.md) — LlamaIndex integration overview
- [LlamaIndex Documentation](https://docs.llamaindex.ai) — Official LlamaIndex documentation
