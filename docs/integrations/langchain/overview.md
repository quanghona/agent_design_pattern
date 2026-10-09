# LangChain Integration

## Overview

The `aap_langchain` package integrates LangChain into the AI Agent Pattern architecture. It provides a `ChatCausalMultiTurnsChain` that implements the core's `BaseCausalMultiTurnsChain` interface using LangChain's `BaseChatModel` and tool-calling infrastructure. The package also includes a retriever adapter for LangChain's retriever primitives and utility functions for converting LangChain's `UsageMetadata` to the core's `TokenUsage` type.

## Architecture

The LangChain integration maps the core's chain abstraction to LangChain's message-based conversation model. `ChatCausalMultiTurnsChain` takes a `BaseChatModel`, a system prompt, and a user prompt template, then uses LangChain's internal chain infrastructure to manage multi-turn conversations. The chain converts `AgentMessage` objects into LangChain's `BaseMessage` hierarchy (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`) and back again. Tool support is provided through LangChain's `BaseTool` and callable interfaces, with automatic tool calling handled by the underlying LangChain chain.

![diagram](img/langchain-architecture.jpg)

## Key Differences from Core

The LangChain integration provides a concrete implementation of the core's abstract chain interface, binding it to LangChain's chat model ecosystem. While the core defines `BaseCausalMultiTurnsChain` as an abstract template method pattern, `aap_langchain` implements it using LangChain's `Chain.invoke()` method, which handles the ReAct loop internally. The integration also leverages LangChain's built-in tool calling rather than implementing tool execution manually.

## Usage

### Installation

```bash
pip install aap_langchain
```

### Basic Example

```python
from aap_langchain.chain import ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage
from langchain_ollama import ChatOllama

# Create a chain with an Ollama model
chain = ChatCausalMultiTurnsChain(
    model=ChatOllama(model="llama3.1"),
    system_prompt="You are a helpful assistant.",
    user_prompt_template="{query}",
)

# Use the chain with an AgentMessage
message = AgentMessage(query="What is RAG?")
result = chain.invoke(message)
```

### Advanced Example

```python
from aap_langchain.chain import ChatCausalMultiTurnsChain
from aap_langchain.retriever import RetrieverAdapter
from aap_core.types import AgentMessage
from langchain_ollama import ChatOllama
from langchain_weaviate import WeaviateVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up a retriever with Weaviate
vector_store = WeaviateVectorStore(...)
retriever = RetrieverAdapter(
    retriever=vector_store.as_retriever(),
    data_key="context.retrieved_docs"
)

# Create a chain with tool support
chain = ChatCausalMultiTurnsChain(
    model=ChatOllama(model="llama3.1"),
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

- [`ChatCausalMultiTurnsChain`](https://github.com/quanghona/agent_design_pattern/blob/main/src/langchain/src/aap_langchain/chain.py) — LangChain-specific implementation of the core's multi-turn chain, using `BaseChatModel` and tool calling
- [`RetrieverAdapter`](https://github.com/quanghona/agent_design_pattern/blob/main/src/langchain/src/aap_langchain/retriever.py) — Wraps LangChain's `BaseRetriever` as a core `BaseRetriever`
- [`token_from_response`](https://github.com/quanghona/agent_design_pattern/blob/main/src/langchain/src/aap_langchain/utils.py) — Converts LangChain `UsageMetadata` to core `TokenUsage`

## API Reference

::: aap_langchain
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Agent](../../core/components/agent.md) — The base agent concept
- [Core Chain](../../core/components/chain.md) — The base chain concept
- [Example Notebooks](../../example/langchain/) — Jupyter notebooks with LangChain examples
