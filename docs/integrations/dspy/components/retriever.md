# DSPy RetrieverAdapter

## Overview

The DSPy `RetrieverAdapter` provides a bridge between the AI Agent Pattern core framework and [DSPy](https://dspy.ai/)'s retrieval abstractions. It wraps either a DSPy `Retrieve` module or a DSPy `Embeddings` object, adapting them to the framework-agnostic `BaseRetriever` interface. This enables DSPy-powered retrieval to be composed seamlessly with agents and chains in the AI Agent Pattern framework.

## Architecture

The `RetrieverAdapter` sits in the integration layer, translating between the core `AgentMessage` type and DSPy's retrieval model. It receives an `AgentMessage`, extracts the query, invokes the wrapped DSPy retriever, and stores the retrieved passages in the message's `context` dictionary under a configurable key.

The adapter integrates with two DSPy retrieval abstractions:

- **`dspy.Retrieve`**: DSPy's built-in retrieval module that supports multiple search backends (KILT, web search, local vector stores). It returns a list of retrieved passages directly.
- **`dspy.Embeddings`**: DSPy's embedding-based retrieval module that returns an object with a `.passages` attribute containing the retrieved results.

![diagram](img/retriever-architecture.jpg)

## Key Concepts

### BaseRetriever: The Retrieval Contract

`RetrieverAdapter` inherits from `aap_core.retriever.BaseRetriever`, which defines the contract for all retrievers in the AI Agent Pattern framework. The base class provides:

- **`retrieve(message, **kwargs)`**: Implemented by `RetrieverAdapter` to invoke the wrapped DSPy retriever.
- **`post_process`**: An optional `BaseChain` that runs after `retrieve()`, enabling post-processing steps such as reranking or summarization.

The `__call__` method orchestrates the pipeline: it first calls `retrieve()`, then applies `post_process` if configured, and returns the final `AgentMessage`.

### Dual Retriever Support

The adapter supports two types of DSPy retrieval modules, detected automatically at runtime:

| DSPy Module | Type | Retrieval Method | Return Format |
|---|---|---|---|
| `dspy.Retrieve` | Standalone retriever | `self.retriever(message.query)` | List of passage strings |
| `dspy.Embeddings` | Embedding-based retriever | `self.retriever(message.query).passages` | List of passage strings from `.passages` attribute |

This dual support allows the adapter to work with both DSPy's built-in retrieval module and custom embedding-based retrieval pipelines.

### Context Key Management

The `data_key` field controls where retrieved data is stored in the message context. The key must start with the `"context."` prefix, which is enforced by a Pydantic validator:

```python
# data_key = "context.data" → stored at message.context["data"]
# data_key = "context.dspy_passages" → stored at message.context["dspy_passages"]
```

When multiple results are retrieved, they are joined with spaces into a single string. When a single result is retrieved, it is stored as-is (not wrapped in a list).

## Usage

### Basic Example: DSPy Retrieve Module

```python
from aap_dspy.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
import dspy

# Set up DSPy retriever
dspy.configure(engine="your-dspy-engine")
retriever = dspy.Retrieve(k=3)

# Wrap with RetrieverAdapter
adapter = RetrieverAdapter(retriever=retriever)

# Use the adapter on a message
message = AgentMessage(query="What are the latest developments in AI?")
result = adapter(message)

print(result.context["data"])  # Retrieved passages as a string
```

### Basic Example: DSPy Embeddings Module

```python
from aap_dspy.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
import dspy

# Set up DSPy embeddings-based retriever
embeddings = dspy.Embeddings(model="your-embedding-model")
adapter = RetrieverAdapter(retriever=embeddings)

# Use the adapter on a message
message = AgentMessage(query="Find relevant information")
result = adapter(message)

print(result.context["data"])  # Retrieved passages from embeddings
```

### Advanced Example: Multi-Source Retrieval with Custom Keys

```python
from aap_dspy.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
import dspy

# Two retrievers with different data keys
knowledge_retriever = RetrieverAdapter(
    retriever=dspy.Retrieve(k=5),
    data_key="context.knowledge",
)

paper_retriever = RetrieverAdapter(
    retriever=dspy.Retrieve(k=3),
    data_key="context.papers",
)

# Chain both retrievers
message = AgentMessage(query="Research topic X")
message = knowledge_retriever(message)
message = paper_retriever(message)

# Both datasets are available in the context
print(message.context["knowledge"])  # Knowledge base passages
print(message.context["papers"])     # Paper passages
```

### Advanced Example: Retrieval with Post-Processing

```python
from aap_dspy.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from aap_core import BaseChain
import dspy


class SummarizingPostProcess(BaseChain):
    """A post-processor that summarizes retrieved data."""

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        data = message.context.get("data", "")
        message.context["summary"] = f"[Summarized: {len(data)} characters of data retrieved]"
        return message


adapter = RetrieverAdapter(retriever=dspy.Retrieve(k=3))
adapter.post_process = SummarizingPostProcess()

message = AgentMessage(query="Tell me about transformers")
result = adapter(message)

print(result.context["data"])     # Raw retrieved passages
print(result.context["summary"])  # Post-processed summary
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retriever` | `Retrieve \| Embeddings` | *(required)* | The DSPy retriever module to wrap. Can be `dspy.Retrieve` or `dspy.Embeddings`. |
| `data_key` | `str` | `"context.data"` | The key under which retrieved data is stored in `message.context`. Must start with `"context."`. |

## API Reference

::: aap_dspy.retriever.RetrieverAdapter
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Retriever](../../../core/components/retriever.md) — The base retriever concept and `BaseRetriever` interface
- [DSPy Overview](../overview.md) — DSPy integration overview
- [DSPy Documentation](https://dspy.ai) — Official DSPy documentation
