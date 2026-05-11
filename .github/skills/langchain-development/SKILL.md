---
name: langchain-development
description: "A structured workflow for designing, implementing, and testing new features using the LangChain framework within the `aap_langchain` package. This skill ensures that all new components are compatible with the `aap_core` architecture and follow LangChain best practices."
---

# LangChain Feature Development Skill

## Overview

This skill provides a structured workflow for designing, implementing, and testing new features using the LangChain framework within the `aap_langchain` package. For comprehensive LangChain best practices (architecture, RAG systems, agent design, tool calling, prompt engineering, error handling, observability, performance, and testing), refer to [best-practices.md](best-practices.md).

## Workflow

### 1. Research & Design
- **Consult Best Practices**: Review [best-practices.md](best-practices.md) for architecture patterns, RAG systems, agent design, and tool calling guidelines.
- **Identify Components**: Determine which LangChain primitives are required:
  - `BaseChatModel`: For LLM interaction.
  - `BaseTool`: For tool/function calling.
  - `BaseRetriever`: For information retrieval.
  - `BaseMessage` subclasses (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`): For conversation state.
  - `Runnable` (LCEL): For composing chains.
- **Define Data Flow**: Map how `AgentMessage` (from `aap_core`) will be transformed into LangChain messages and how the results will be converted back.
- **Design for Extensibility**: Use Pydantic models and `PrivateAttr` for internal state to ensure compatibility with LangChain's architecture.

### 2. Component Implementation
- **Consult Best Practices**: Review [best-practices.md](best-practices.md) for tool design principles, prompt engineering patterns, and error handling strategies.
- **Implement Adapters**: If using existing LangChain components, create adapters (e.g., `RetrieverAdapter`) to bridge them with `aap_core` types.
- **Implement Tools**: Create new tools by subclassing `BaseTool` or using the `@tool` decorator. Ensure they handle input/output types correctly and provide clear descriptions. Follow the production tool pattern from best-practices.md.
- **Implement Chains (LCEL focus)**:
  - For complex logic, leverage LangChain Expression Language (LCEL) using `RunnableSequence`, `RunnableParallel`, and `RunnableLambda`.
  - Extend `BaseCausalMultiTurnsChain` to wrap these LCEL chains, ensuring they can handle `AgentMessage` inputs and outputs.
- **Follow Python Standards**: Adhere to `@file:python-development.instructions.md`.

### 3. Integration & Composition
- **Consult Best Practices**: Review [best-practices.md](best-practices.md) for multi-agent architecture patterns, state management, and observability guidelines.
- **Compose the Chain**: Use LCEL to link the prompt, model, tools, and retrievers (e.g., `chain = prompt | model.bind_tools(tools) | output_parser`).
- **Handle State**: Ensure the chain correctly manages message history and tool call sequences. The `_prepare_conversation` method in `ChatCausalMultiTurnsChain` is the key bridge for converting `AgentMessage` history to `List[BaseMessage]`.
- **Token Usage**: Implement logic to extract and convert `UsageMetadata` from `AIMessage` to `TokenUsage` (refer to `aap_langchain.utils.token_from_response`).
- **Error Handling**: Implement retry with exponential backoff and fallback strategies as described in [best-practices.md](best-practices.md).

### 4. Testing (LangChain Specific)
Follow `@file:writing-and-running-tests.instructions.md` and apply these LangChain-specific strategies. See [best-practices.md](best-practices.md) for testing guidelines and production checklist.

- **Mocking LLMs**:
  - **NEVER** make real API calls.
  - Use `unittest.mock.MagicMock` to mock `BaseChatModel.invoke`.
  - Configure the mock to return `AIMessage` objects with desired `content`, `tool_calls`, and `usage_metadata`.
- **Message Sequence Testing**:
  - Test `_prepare_conversation` logic: verify that `AgentMessage` history is correctly converted to a list of `BaseMessage` objects (including `HumanMessage`, `AIMessage`, `SystemMessage`, and `ToolMessage`) in the right order.
  - Verify `SystemMessage` and `HumanMessage` formatting.
- **Tool Calling Tests**:
  - Test that `AIMessage.tool_calls` are correctly processed by the chain.
  - Verify that `ToolMessage` is correctly appended to the conversation after tool execution and that it correctly references the `tool_call_id`.
- **Retriever Integration Tests**:
  - Test that `RetrieverAdapter` correctly populates the `context` field in `AgentMessage`.
  - Verify that retrieved `Document` content is correctly extracted and formatted.
  - Test wrapping complex `Runnable` chains in `RetrieverAdapter` to support advanced retrieval (e.g., reranking).
- **Round-trip Validation**:
  - Ensure that the output of a chain can be successfully converted back into an `AgentMessage`.


### 5. Validation & Cleanup
- **Lint & Format**: Run `uv run ruff check --fix` and `uv run ruff format`.
- **Full Test Suite**: Run `uv run pytest -v` within the `src/langchain` directory.

## Implementation Examples

### 1. Creating a Tool
```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny and 25°C."
```

### 2. Implementing a Chain using LCEL
```python
from langchain_core.prompts import ChatPromptTemplate
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_core.language_models import BaseChatModel

class WeatherChain(ChatCausalMultiTurnsChain):
    def __init__(self, model: BaseChatModel, tools: list):
        # The chain must accept List[BaseMessage] as input
        # ChatCausalMultiTurnsChain passes the conversation list to self._chain.invoke
        self._chain = model.bind_tools(tools)
        super().__init__(
            model=model,
            system_prompt="You are a helpful weather assistant.",
            tools=tools
        )
```

### 3. Testing a Chain with Mocks
```python
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from aap_langchain.chain import ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage, TokenUsage

def test_weather_chain_response():
    # Arrange
    mock_model = MagicMock()
    mock_response = AIMessage(
        content="The weather is sunny.",
        tool_calls=[],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    mock_model.invoke.return_value = mock_response

    # Act
    chain = WeatherChain(model=mock_model, tools=[])
    msg = AgentMessage(query="What is the weather?")
    response = chain.call(msg)

    # Assert
    assert response.query == "What is the weather?"
    assert response.responses[0][1] == "The weather is sunny."
```

### 4. Implementing a Prompt Augmenter
```python
from langchain_core.prompts import ChatPromptTemplate
from aap_langchain.chain import ChatCausalMultiTurnsChain
from aap_core.prompt_augmenter import BasePromptAugmenter
from aap_core.types import AgentMessage

class LangChainPromptAugmenter(BasePromptAugmenter):
    def __init__(self, chain: ChatCausalMultiTurnsChain, **kwargs):
        super().__init__(**kwargs)
        self.chain = chain

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Use the LangChain chain to rewrite the query
        # This assumes the chain is designed to return a response that can be used as a new query
        response = self.chain.call(message)

        # Update the message with the augmented query
        message.query = response.responses[0][1]
        return message
```
