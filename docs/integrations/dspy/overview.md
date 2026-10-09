# DSPy Integration

## Overview

The `aap_dspy` package integrates the DSPy framework into the AI Agent Pattern architecture. It provides a bridge between the core's `AgentMessage` type system and DSPy's declarative `Signature` paradigm, enabling agents to leverage DSPy's programmatic optimization capabilities. The package includes a `ChatCausalMultiTurnsChain` for multi-turn LM calls with DSPy predictors, a signature adapter for bidirectional conversion between `AgentMessage` and `dspy.Signature`, a retriever adapter for DSPy's retrieval primitives, and utility functions for token usage tracking.

## Architecture

The DSPy integration maps core concepts to DSPy's abstractions through a signature adapter layer and a chain implementation. The `BaseSignatureAdapter` class converts `AgentMessage` objects into lists of `dspy.Signature` instances before flowing into DSPy predictors, and converts DSPy predictions back into `AgentResponse` tuples. This adapter supports a prefill dictionary for static fields that exist in signatures but not in `AgentMessage`. The `ChatCausalMultiTurnsChain` class implements the core's multi-turn chain interface using a DSPy predictor and adapter, supporting both fully managed tool calling (via `dspy.ReAct`) and manual tool handling (via `dspy.ToolCalls` fields in signatures). The integration also wraps DSPy's `Retrieve` and `Embeddings` classes as `BaseRetriever` implementations, allowing DSPy-powered retrieval to plug directly into the core's retrieval pipeline.

![diagram](img/dspy-architecture.jpg)

## Key Differences from Core

Unlike the core package which defines imperative agent and chain interfaces, `aap_dspy` embraces DSPy's declarative approach. Instead of writing explicit execution logic, developers define input/output signatures and let DSPy's optimizers find the best prompt structure and model configuration. The integration does not provide its own `BaseAgent` or `BaseLLMChain` implementations — instead, it focuses on the adapter layer that makes DSPy's signatures compatible with the core's message passing protocol.

## Usage

### Installation

```bash
pip install aap_dspy
```

### Basic Example

```python
from aap_dspy.chain import BaseSignatureAdapter
from aap_core.types import AgentMessage

# Create a signature adapter with prefill values
adapter = BaseSignatureAdapter.with_prefill({"static_field": "value"})

# Convert an AgentMessage to DSPy signatures
signatures = adapter.msg2sig(message)

# Convert DSPy predictions back to AgentMessage responses
responses = adapter.sig2msg(predictions, name="my_agent")
```

### Advanced Example

```python
from aap_dspy.retriever import RetrieverAdapter
from dspy import Retrieve

# Create a DSPy retriever adapter
retriever = RetrieverAdapter(
    retriever=Retrieve(k=5),
    data_key="context.retrieved_docs"
)

# Use it within a core agent workflow
message = AgentMessage(query="What is RAG?")
message = retriever(message)
# message.context["retrieved_docs"] now contains the retrieved passages
```

## Components

- [`ChatCausalMultiTurnsChain`](https://github.com/quanghona/agent_design_pattern/blob/main/src/dspy/src/aap_dspy/chain.py) — DSPy-specific multi-turn chain using a `dspy.Module` predictor and `BaseSignatureAdapter`
- [`BaseSignatureAdapter`](https://github.com/quanghona/agent_design_pattern/blob/main/src/dspy/src/aap_dspy/chain.py) — Converts between `AgentMessage` and `dspy.Signature` with support for prefill dictionaries
- [`RetrieverAdapter`](https://github.com/quanghona/agent_design_pattern/blob/main/src/dspy/src/aap_dspy/retriever.py) — Wraps DSPy's `Retrieve` or `Embeddings` as a core `BaseRetriever`
- [`token_from_response`](https://github.com/quanghona/agent_design_pattern/blob/main/src/dspy/src/aap_dspy/utils.py) — Converts DSPy `Prediction` objects to core `TokenUsage`

## API Reference

::: aap_dspy
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Agent](../../core/components/agent.md) — The base agent concept
- [Core Chain](../../core/components/chain.md) — The base chain concept
- [Example Notebooks](../../example/dspy/) — Jupyter notebooks with DSPy examples
