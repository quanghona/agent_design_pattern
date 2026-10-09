# LlamaIndex Integration

## Overview

The `aap_llamaindex` package integrates LlamaIndex into the AI Agent Pattern architecture. It provides a `ChatCausalMultiTurnsChain` that implements the core's `BaseCausalMultiTurnsChain` interface using LlamaIndex's `FunctionCallingLLM` with built-in tool calling. The package also includes a retriever adapter for LlamaIndex's base retriever primitives and utility functions for token usage tracking.

## Architecture

The LlamaIndex integration maps the core's chain abstraction to LlamaIndex's message-based conversation model. `ChatCausalMultiTurnsChain` takes a `FunctionCallingLLM`, a system prompt, and a user prompt template, then uses LlamaIndex's `chat_with_tools` method to manage multi-turn conversations with tool calling. The chain converts `AgentMessage` objects into LlamaIndex's `ChatMessage` hierarchy with `MessageRole` annotations (`SYSTEM`, `USER`, `ASSISTANT`, `TOOL`). Tool support is provided through LlamaIndex's `BaseTool` and callable interfaces, with tool execution handled natively by LlamaIndex's function calling infrastructure.

![diagram](img/llamaindex-architecture.jpg)

## Key Differences from Core

The LlamaIndex integration provides a concrete implementation of the core's abstract chain interface, binding it to LlamaIndex's function-calling LLM ecosystem. While the core defines `BaseCausalMultiTurnsChain` as an abstract template method pattern, `aap_llamaindex` implements it using LlamaIndex's `chat_with_tools` method, which handles the ReAct loop and tool execution internally. The integration is specifically designed for LlamaIndex's document-centric RAG workflows, making it the natural choice when the agent needs to interact with structured knowledge bases.

## Usage

### Installation

```bash
pip install aap_llamaindex
```

### Basic Example

```python
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage
from llama_index.llms.ollama import Ollama

# Create a chain with an Ollama model
chain = ChatCausalMultiTurnsChain(
    model=Ollama(model="llama3.1", request_timeout=60.0),
    system_prompt="You are a helpful assistant.",
    user_prompt_template="{query}",
)

# Use the chain with an AgentMessage
message = AgentMessage(query="What is RAG?")
result = chain.invoke(message)
```

### Advanced Example

```python
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from aap_llamaindex.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# Set up a retriever with FAISS
faiss_index = faiss.IndexFlatL2(768)
vector_store = FaissVectorStore(faiss_index=faiss_index)
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
retriever = RetrieverAdapter(
    retriever=index.as_retriever(),
    data_key="context.retrieved_docs"
)

# Create a chain with tool support
chain = ChatCausalMultiTurnsChain(
    model=Ollama(model="llama3.1", request_timeout=60.0),
    system_prompt="You are a research assistant. Use retrieved documents to answer questions.",
    user_prompt_template="{query}",
    tools=[retriever],
    include_history=3,
)

# Use the chain
message = AgentMessage(query="Explain retrieval-augmented generation")
result = chain.invoke(message)
```

## Components

- [`ChatCausalMultiTurnsChain`](https://github.com/quanghona/agent_design_pattern/blob/main/src/llamaindex/src/aap_llamaindex/chain.py) — LlamaIndex-specific implementation of the core's multi-turn chain, using `FunctionCallingLLM` with native tool calling
- [`RetrieverAdapter`](https://github.com/quanghona/agent_design_pattern/blob/main/src/llamaindex/src/aap_llamaindex/retriever.py) — Wraps LlamaIndex's `BaseRetriever` as a core `BaseRetriever`
- [`token_from_response`](https://github.com/quanghona/agent_design_pattern/blob/main/src/llamaindex/src/aap_llamaindex/utils.py) — Converts LlamaIndex `CompletionResponse`/`ChatResponse` to core `TokenUsage`

## API Reference

::: aap_llamaindex
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Agent](../../core/components/agent.md) — The base agent concept
- [Core Chain](../../core/components/chain.md) — The base chain concept
- [Example Notebooks](../../example/llamaindex/) — Jupyter notebooks with LlamaIndex examples
