# LangChain Chain

## Overview

The LangChain chain provides a bridge between the AI Agent Pattern core framework and the [LangChain](https://python.langchain.com/) framework. It implements `ChatCausalMultiTurnsChain`, which extends `BaseCausalMultiTurnsChain` to handle LangChain's message-based conversation model. The chain supports both simple chat model invocations and multi-turn tool calling, with full support for binding and swapping tools at runtime.

## Architecture

The LangChain chain sits within the integration layer, translating between the framework-agnostic `AgentMessage` type and LangChain's `BaseMessage` hierarchy (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`). It manages the LLM chain internally, rebuilding it when tools or prompts change.

The chain integrates with LangChain's core abstractions:

- **`BaseChatModel`**: The underlying chat model (e.g., `ChatOpenAI`, `ChatAnthropic`). The chain stores a reference and can be reconfigured at runtime.
- **`BaseMessage`**: LangChain's message type hierarchy. The chain builds a conversation list of these messages for each invocation.
- **`BaseTool` / `Callable`**: Both LangChain tools and plain Python functions are supported as tool implementations.

![diagram](img/chain-architecture.jpg)

## Key Concepts

### Conversation Preparation

The `_prepare_conversation` method constructs a LangChain conversation list from an `AgentMessage`. It always starts with a `SystemMessage` (from `system_prompt`) followed by a `HumanMessage` (formatted from `user_prompt_template` and the message's query). Previous turns from `message.responses` are appended in order, with each response mapped to the appropriate LangChain message type:

| Response Type | LangChain Message |
|---------------|-------------------|
| `"user"` | `HumanMessage` |
| `"assistant"` (default) | `AIMessage` |
| `"tool"` | `ToolMessage` |
| `"system"` | `SystemMessage` |

The number of previous turns included is controlled by `include_history`. A value of `-1` includes all turns; a non-negative value limits to that many recent turns.

### Tool Binding and Management

Tools are bound to the chain via the `bind_tools()` method, which accepts a sequence of `BaseTool` objects or plain `Callable` functions. The chain maintains an internal `_tool_dict` that maps tool names to their implementations:

- For `BaseTool` instances, the tool's `.name` attribute is used as the key.
- For `Callable` functions, the function's `__name__` is used as the key.

After binding, the chain calls `self._model.bind_tools()` to create a tool-aware model chain stored in `self._chain`. This allows the LLM to discover and invoke tools by name.

```python
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"The weather in {location} is sunny."

chain = ChatCausalMultiTurnsChain(
    model=ChatOpenAI(model="gpt-4"),
    system_prompt="You are a helpful assistant.",
    tools=[get_weather],
)
```

### Tool Execution

When the LLM returns an `AIMessage` with `tool_calls`, the `_process_tools` method executes them:

1. For each tool call, it looks up the tool name in `_tool_dict`.
2. If the tool exists and is a `BaseTool`, it invokes `tool.invoke(tool_call["args"])`.
3. If the tool exists and is a `Callable`, it invokes `tool_func(**tool_call["args"])`.
4. The result is wrapped in a `ToolMessage` and appended to the conversation.
5. If the tool doesn't exist or an error occurs, an error `ToolMessage` is appended instead.

This loop continues until the LLM returns a response with no tool calls.

### Runtime Reconfiguration

The chain supports full runtime reconfiguration:

- **`update_prompt(system_prompt, user_prompt_template)`**: Changes both prompts and rebuilds the tool-bound chain.
- **`model` property (setter)**: Replaces the underlying model and rebuilds the chain with the currently bound tools.
- **`tools` property**: Returns the list of currently bound tools.

This makes the chain suitable for scenarios where prompts or models need to change between invocations without recreating the chain object.

## Usage

### Basic Example: Simple Chat

```python
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_openai import ChatOpenAI
from aap_core.types import AgentMessage


chain = ChatCausalMultiTurnsChain(
    model=ChatOpenAI(model="gpt-4"),
    system_prompt="You are a helpful assistant.",
    user_prompt_template="Answer the following: {query}",
    name="langchain-agent",
)

message = AgentMessage(query="What is machine learning?")
result = chain.invoke(message)
print(result.responses)
```

### Advanced Example: With Tools

```python
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from aap_core.types import AgentMessage


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    return f"Results for: {query}"


chain = ChatCausalMultiTurnsChain(
    model=ChatOpenAI(model="gpt-4"),
    system_prompt="You are a research assistant. Use tools to find information.",
    tools=[search_knowledge_base],
    name="research-assistant",
    max_turns=10,
)

message = AgentMessage(query="Tell me about transformer architectures")
result = chain.invoke(message)
```

### Runtime Reconfiguration

```python
# Update prompts at runtime
chain.update_prompt(
    system_prompt="You are a coding assistant.",
    user_prompt_template="Solve this problem: {query}",
)

# Swap the model
from langchain_anthropic import ChatAnthropic
chain.model = ChatAnthropic(model="claude-3-sonnet-20240229")
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BaseChatModel` | *(required)* | The LangChain chat model to use (e.g., `ChatOpenAI`, `ChatAnthropic`). |
| `system_prompt` | `str` | *(required)* | The system prompt that sets the assistant's behavior. |
| `user_prompt_template` | `str` | `"{query}"` | Template for formatting the user's query. Must contain `{query}`. |
| `tools` | `Sequence[Callable \| BaseTool]` | `[]` | Tools available to the model. Can be `BaseTool` objects or plain functions. |
| `tool_choice` | `str \| None` | `None` | LangChain's `tool_choice` parameter for `bind_tools`. Controls how the model selects tools. |
| `name` | `str` | — | The name of the chain, used as the source name in responses. |
| `max_turns` | `int` | `50` | Maximum number of tool-call turns before the loop terminates. |
| `include_history` | `int` | — | Number of previous turns to include. `-1` means all. |
| `store_immediate_steps` | `bool` | `False` | Whether to store all intermediate steps or only the final response. |

## API Reference

::: aap_langchain.ChatCausalMultiTurnsChain
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Chain](../../../core/components/chain.md) — The base chain concept and `BaseCausalMultiTurnsChain` interface
- [LangChain Overview](../overview.md) — LangChain integration overview
- [LangChain Documentation](https://python.langchain.com) — Official LangChain documentation
