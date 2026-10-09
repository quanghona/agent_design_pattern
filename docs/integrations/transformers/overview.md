# Transformers Integration

## Overview

The `aap_transformers` package integrates Hugging Face Transformers into the AI Agent Pattern architecture. It provides a `ChatCausalMultiTurnsChain` that implements the core's `BaseCausalMultiTurnsChain` interface using Hugging Face's `AutoModelForCausalLM` and `AutoTokenizer`. The package enables running local causal language models directly within the agent framework, with support for both CPU and GPU inference.

## Architecture

The Transformers integration maps the core's chain abstraction to Hugging Face's model inference pipeline. `ChatCausalMultiTurnsChain` accepts either a model path string or a pre-loaded `(tokenizer, model)` tuple, then uses the tokenizer to format conversations and the model to generate responses. The chain converts `AgentMessage` objects into a list of role-content message dictionaries (`TransformersChainMessage`) and back again. Tool support is provided through callable interfaces, with tool call extraction handled via regex parsing of the model's output (e.g., extracting `<tool_call>`-delimited tool call blocks).

![diagram](img/transformers-architecture.jpg)

## Key Differences from Core

The Transformers integration provides a concrete implementation of the core's abstract chain interface, binding it to Hugging Face's local model inference. Unlike the LangChain and LlamaIndex integrations which rely on cloud API models or managed services, `aap_transformers` runs models entirely locally, giving full control over model weights, inference parameters, and hardware allocation. The integration also includes built-in support for removing thinking tags (e.g., `<think>`/`</think>`) from model outputs, which is particularly relevant for models like Qwen that use these special tokens.

## Usage

### Installation

```bash
pip install aap_transformers
```

### Basic Example

```python
from aap_transformers.chain import ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage

# Create a chain with a local model
chain = ChatCausalMultiTurnsChain(
    model="meta-llama/Llama-3.1-8B-Instruct",
    system_prompt="You are a helpful assistant.",
    user_prompt_template="{query}",
    device="cuda",
)

# Use the chain with an AgentMessage
message = AgentMessage(query="What is RAG?")
result = chain.invoke(message)
```

### Advanced Example

```python
from aap_transformers.chain import ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer manually for custom configuration
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="cuda",
    torch_dtype="auto",
)

# Create a chain with the pre-loaded model
chain = ChatCausalMultiTurnsChain(
    model=(tokenizer, model),
    system_prompt="You are a research assistant. Answer concisely.",
    user_prompt_template="{query}",
    device="cuda",
    include_history=3,
)

# Use the chain
message = AgentMessage(query="Explain retrieval-augmented generation")
result = chain.invoke(message)
```

## Components

- [`ChatCausalMultiTurnsChain`](https://github.com/quanghona/agent_design_pattern/blob/main/src/transformers/src/aap_transformers/chain.py) — Transformers-specific implementation of the core's multi-turn chain, using `AutoModelForCausalLM` with local inference
- [`RetrieverAdapter`](https://github.com/quanghona/agent_design_pattern/blob/main/src/transformers/src/aap_transformers/retriever.py) — Transformers-specific retriever (currently a placeholder; no implementation yet)

## API Reference

::: aap_transformers
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Agent](../../core/components/agent.md) — The base agent concept
- [Core Chain](../../core/components/chain.md) — The base chain concept
- [Example Notebooks](../../example/transformers/) — Jupyter notebooks with Transformers examples
