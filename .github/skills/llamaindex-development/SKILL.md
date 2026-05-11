---
name: llamaindex-development
description: "A structured workflow for designing, implementing, and testing new features using the LlamaIndex framework within the `aap_llamaindex` package. This skill ensures that all new components are compatible with the `aap_core` architecture and follow LlamaIndex best practices, particularly focusing on RAG (Retrieval-Augmented Generation) and agentic workflows."
---

# LlamaIndex Development Skill

## Overview

This skill provides a structured workflow for designing, implementing, and testing new features using the LlamaIndex framework within the `aap_llamaindex` package. For comprehensive LlamaIndex best practices (RAG optimization, indexing strategies, query engines, agents, production deployment, and testing), refer to [best-practices.md](best-practices.md).

## Core Concepts & API Reference

LlamaIndex is a data framework for LLM applications. When developing within `aap_llamaindex`, keep these core abstractions in mind:

### 1. Data Ingestion & Indexing
- **Data Connectors (LlamaHub):** Used to ingest data from various sources.
- **Documents & Nodes:** The fundamental units of data. `Document` is the raw text/metadata, while `Node` is a chunk of a document with relationship information.
- **Indices:** Data structures that allow for efficient retrieval (e.s., `VectorStoreIndex`, `SummaryIndex`, `KnowledgeGraphIndex`).

### 2. Retrieval & Querying
- **Retriever:** An abstraction that takes a query and returns a list of `Node` objects.
- **Query Engine:** A higher-level abstraction that takes a query and returns a response (often by combining a retriever and a response synthesizer).
- **Post-processors:** Modules used to refine retrieved nodes (e.g., `SimilarityPostprocessor`, `Re-ranking`).

### 3. Agents & Workflows
- **Agentic Workflows:** LlamaIndex provides advanced workflows for multi-agent orchestration and complex reasoning.
- **Tools:** Functions or classes that agents can call to interact with the world or retrieve information.
- **Chat Engines:** Specialized query engines designed for conversational interfaces (e.g., `CondenseQuestionChatEngine`).

## Integration with `aap_core`

Our framework uses `aap_core` as the foundational layer. When implementing LlamaIndex features, follow these integration patterns:

### 1. Chain Implementation
Implement `BaseCausalMultiTurnsChain` from `aap_core` to create LlamaIndex-powered chains.
- **Mapping:** Map LlamaIndex `ChatResponse` and `ChatMessage` to `aap_core` types.
- **Tool Handling:** Use `aap_core`'s tool calling mechanism to wrap LlamaIndex tools.

### 2. Retriever Adapter
Use the `RetrieverAdapter` in `aap_llamaindex.retriever` to wrap LlamaIndex `BaseRetriever` instances.
- This allows LlamaIndex retrievers to be used seamlessly within the `aap_core` ecosystem.
- Ensure the `data_key` is correctly configured to store retrieved context in the `AgentMessage`.

### 3. Type Conversion
Always use the utility functions in `aap_llamaindex.utils` for consistent type handling:
- `token_from_response`: Converts LlamaIndex `ChatResponse` to `aap_core.types.TokenUsage`.

## Development Workflow

1.  **Consult Best Practices:** Review [best-practices.md](best-practices.md) for RAG optimization, indexing strategies, and production deployment patterns.
2.  **Define the Goal:** Determine if you are building a new retriever, a specialized query engine, or a complex agentic workflow.
3.  **Research LlamaIndex API:** Consult the [LlamaIndex API Reference](https://developers.llamaindex.ai/python/framework-api-reference/) for the specific module you are working with.
3.  **Implement the Component:**
    - If it's a retriever, implement `RetrieverAdapter`.
    - If it's a chain, implement `ChatCausalMultiTurnsChain`.
    - **Follow Python Standards**: Adhere to `@file:python-development.instructions.md`.
4. **Integrate with `aap_core`:** Ensure all inputs and outputs are correctly mapped to `AgentMessage` and `AgentResponse`.
5. **Testing (LlamaIndex Specific):**
Follow `@file:writing-and-running-tests.instructions.md` and apply these LlamaIndex-specific strategies. See [best-practices.md](best-practices.md) for testing guidelines and evaluation patterns.

- **Mocking the LLM**:
  - **NEVER** use real LLM calls in unit tests.
  - Use `unittest.mock` to mock LlamaIndex LLM objects.
- **Retriever Testing**:
  - Test `RetrieverAdapter` with various retrieval scenarios (empty results, multiple nodes, etc.).
  - Verify that retrieved `Node` content is correctly mapped to `AgentMessage` context.
- **Chain Testing**:
  - Test `ChatCausalMultiTurnsChain` with mocked LlamaIndex `ChatResponse` objects.
  - Verify that conversation history is correctly maintained and passed to the LlamaIndex engine.
- **Type Conversion Testing**:
  - Verify that `token_from_response` correctly handles different `ChatResponse` formats.

## Example: Implementing a Custom Retriever Adapter

```python
from aap_core.retriever import BaseRetriever
from aap_llamaindex.retriever import RetrieverAdapter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. Setup LlamaIndex components
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
llamaindex_retriever = index.as_retriever()

# 2. Wrap with AAP RetrieverAdapter
aap_retriever = RetrieverAdapter(
    retriever=llamaindex_retriever,
    data_key="context.documents"
)

# 3. Use within AAP ecosystem
# (Assuming 'message' is an AgentMessage)
updated_message = aap_retriever.retrieve(message)
```

## Example: Implementing a Prompt Augmenter

```python
from aap_core.prompt_augmenter import BasePromptAugmenter
from aap_core.types import AgentMessage
from llama_index.core import QueryEngine

class LlamaIndexPromptAugmenter(BasePromptAugmenter):
    def __init__(self, query_engine: QueryEngine, **kwargs):
        super().__init__(**kwargs)
        self.query_engine = query_engine

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Use LlamaIndex query engine to get context
        response = self.query_engine.query(message.query)

        # Add the response as context to the message
        if message.context is None:
            message.context = {}
        message.context["llamaindex_response"] = str(response)

        return message
```

### 3. Testing the Query Engine
```python
import pytest
from unittest.mock import MagicMock
from llama_index.core.response import Response

def test_query_engine():
    # Arrange
    mock_engine = MagicMock()
    mock_engine.query.return_value = Response(
        response="AI is a branch of computer science.",
        source_nodes=[]
    )

    # Act
    augmenter = LlamaIndexPromptAugmenter(query_engine=mock_engine)
    msg = AgentMessage(query="What is AI?")
    result = augmenter.augment(msg)

    # Assert
    assert "llamaindex_response" in result.context
    assert "AI is a branch" in result.context["llamaindex_response"]
```

## Example: Implementing a Causal Multi-Turn Chain

```python
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from llama_index.core.llms import OpenAI

# 1. Setup LlamaIndex LLM
llm = OpenAI(model="gpt-4")

# 2. Implement the AAP Chain
chain = ChatCausalMultiTurnsChain(
    model=llm,
    system_prompt="You are a helpful assistant specialized in document analysis.",
    user_prompt_template="Analyze the following: {query}",
    tools=[] # Add LlamaIndex tools here
)

# 3. Execute (following aap_core patterns)
response = chain.generate(message)
```
