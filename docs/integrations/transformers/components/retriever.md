# Transformers RetrieverAdapter

## Overview

The Transformers `RetrieverAdapter` provides a bridge between the AI Agent Pattern core framework and [Hugging Face Transformers](https://huggingface.co/docs/transformers/)'s retrieval abstractions. It wraps a Transformers-based retriever and adapts it to the framework-agnostic `BaseRetriever` interface, enabling seamless integration of Transformers-powered retrieval into agents and chains.

## Architecture

The `RetrieverAdapter` sits in the integration layer, translating between the core `AgentMessage` type and Transformers' retrieval model. It receives an `AgentMessage`, extracts the query, invokes the wrapped Transformers retriever, and stores the retrieved data in the message's `context` dictionary under a configurable key.

The adapter integrates with the `BaseRetriever` interface from `aap_core.retriever`, making it directly composable with other components in chains and orchestration patterns.

![diagram](img/retriever-architecture.jpg)

## Key Concepts

### BaseRetriever: The Retrieval Contract

`RetrieverAdapter` inherits from `aap_core.retriever.BaseRetriever`, which defines the contract for all retrievers in the AI Agent Pattern framework. The base class provides:

- **`retrieve(message, **kwargs)`**: Implemented by `RetrieverAdapter` to invoke the wrapped Transformers retriever.
- **`post_process`**: An optional `BaseChain` that runs after `retrieve()`, enabling post-processing steps such as reranking or summarization.

The `__call__` method orchestrates the pipeline: it first calls `retrieve()`, then applies `post_process` if configured, and returns the final `AgentMessage`.

### Context Key Management

The `data_key` field controls where retrieved data is stored in the message context. The key must start with the `"context."` prefix, which is enforced by a Pydantic validator:

```python
# data_key = "context.data" → stored at message.context["data"]
# data_key = "context.transformers_results" → stored at message.context["transformers_results"]
```

This design allows multiple retrievers to store data under different keys in the same message context, enabling complex multi-source retrieval scenarios.

## Usage

### Basic Example

```python
from aap_transformers.retriever import RetrieverAdapter
from aap_core.types import AgentMessage

# TODO: Add a basic working example with a Transformers retriever
```

### Advanced Example

```python
from aap_transformers.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from aap_core import BaseChain

# TODO: Add a more complex usage example
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retriever` | *Transformers retriever type* | *(required)* | The Transformers retriever to wrap. |
| `data_key` | `str` | `"context.data"` | The key under which retrieved data is stored in `message.context`. Must start with `"context."`. |

## API Reference

::: aap_transformers.retriever.RetrieverAdapter
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Retriever](../../../core/components/retriever.md) — The base retriever concept and `BaseRetriever` interface
- [Transformers Overview](../overview.md) — Transformers integration overview
- [Transformers Documentation](https://huggingface.co/docs/transformers) — Official Transformers documentation
