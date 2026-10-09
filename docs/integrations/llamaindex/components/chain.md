# LlamaIndex Chain

## Overview

The LlamaIndex chain provides a bridge between the AI Agent Pattern core framework and the [LlamaIndex](https://docs.llamaindex.ai/) framework. It implements `ChatCausalMultiTurnsChain`, which extends `BaseCausalMultiTurnsChain` to handle LlamaIndex's `ChatMessage` and `ChatResponse` types. The chain leverages LlamaIndex's built-in function calling capabilities through `FunctionCallingLLM.chat_with_tools()`, supporting both text and tool call blocks in model responses.

## Architecture

The LlamaIndex chain sits within the integration layer, translating between the framework-agnostic `AgentMessage` type and LlamaIndex's message and response types. It uses LlamaIndex's `FunctionCallingLLM` abstraction, which provides unified tool calling across multiple LLM providers.

The chain integrates with LlamaIndex's core abstractions:

- **`FunctionCallingLLM`**: The LLM interface that supports tool calling. The chain uses `chat_with_tools()` for multi-turn tool execution.
- **`ChatMessage`**: LlamaIndex's message type, parameterized by `MessageRole` (`SYSTEM`, `USER`, `ASSISTANT`, `TOOL`).
- **`BaseTool` / `FunctionTool`**: LlamaIndex's tool abstraction. Plain `Callable` functions are automatically wrapped into `FunctionTool` instances.

![diagram](img/chain-architecture.jpg)

## Key Concepts

### Conversation Preparation

The `_prepare_conversation` method constructs a list of `ChatMessage` objects from an `AgentMessage`. The conversation always begins with a `SYSTEM` message (from `system_prompt`) followed by a `USER` message (formatted from `user_prompt_template`). Previous turns are appended with their roles mapped as follows:

| Response Type | LlamaIndex Role |
|---------------|-----------------|
| `"user"` | `MessageRole.USER` |
| `"assistant"` (default) | `MessageRole.ASSISTANT` |
| `"tool"` | `MessageRole.TOOL` |
| `"system"` | `MessageRole.SYSTEM` |

The `include_history` parameter controls how many previous turns are included, with `-1` meaning all turns.

### Tool Calling with `chat_with_tools`

The chain uses LlamaIndex's `FunctionCallingLLM.chat_with_tools()` method, which handles the full tool-calling loop internally for a single turn:

```python
response = self.model.chat_with_tools(
    user_msg=conversation[-1],       # The latest user/assistant message
    chat_history=conversation[:-1],  # All previous messages
    tools=self.tools,                # Available tools
    **kwargs,
)
```

The response contains `ChatMessage.blocks`, which can include:
- **Text blocks** (`block_type == "text"`): The model's text response. The chain applies `utils.remove_thinking()` to strip any thinking tags.
- **Tool call blocks** (`block_type == "tool_call"`): Indicate that the model wants to invoke a tool.

### Tool Execution

When tool calls are detected, `_process_tools` extracts them using `self.model.get_tool_calls_from_response()` and executes each one:

1. Looks up the tool by name in `_tool_dict`.
2. If found, invokes the tool with `tool(**tool_call.tool_kwargs)` and appends a `TOOL` message with the result.
3. If not found or an error occurs, appends a `TOOL` message with an error description.

Each tool result message includes `additional_kwargs` with `tool_call_id` and the tool's name, which LlamaIndex requires for proper conversation tracking.

### Automatic Tool Wrapping

When plain Python functions are passed to the chain, they are automatically wrapped into `FunctionTool` instances:

```python
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from llama_index.llms.openai import OpenAI


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI from weight and height."""
    return weight_kg / (height_m ** 2)


llm = OpenAI(model="gpt-4")
chain = ChatCausalMultiTurnsChain(
    model=llm,
    system_prompt="You are a health assistant.",
    tools=[calculate_bmi],  # Automatically wrapped
    name="health-assistant",
)
```

## Usage

### Basic Example: Simple Chat

```python
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from llama_index.llms.openai import OpenAI
from aap_core.types import AgentMessage


llm = OpenAI(model="gpt-4")

chain = ChatCausalMultiTurnsChain(
    model=llm,
    system_prompt="You are a helpful assistant.",
    user_prompt_template="Answer the following: {query}",
    name="llamaindex-agent",
)

message = AgentMessage(query="What is the speed of light?")
result = chain.invoke(message)
print(result.responses)
```

### Advanced Example: With Tools

```python
from aap_llamaindex.chain import ChatCausalMultiTurnsChain
from llama_index.llms.openai import OpenAI
from aap_core.types import AgentMessage


def convert_temperature(celsius: float, unit: str) -> str:
    """Convert Celsius to the specified unit (F or K)."""
    if unit == "F":
        return f"{celsius * 9/5 + 32}°F"
    elif unit == "K":
        return f"{celsius + 273.15}K"
    return "Invalid unit"


llm = OpenAI(model="gpt-4")

chain = ChatCausalMultiTurnsChain(
    model=llm,
    system_prompt="You are a weather assistant. Use tools for conversions.",
    tools=[convert_temperature],
    name="weather-assistant",
    max_turns=10,
)

message = AgentMessage(query="Convert 100 Celsius to Fahrenheit")
result = chain.invoke(message)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `FunctionCallingLLM` | *(required)* | The LlamaIndex function-calling LLM (e.g., `OpenAI`, `ChatAnthropic`). |
| `system_prompt` | `str` | *(required)* | The system prompt that sets the assistant's behavior. |
| `user_prompt_template` | `str` | *(required)* | Template for formatting the user's query. Must contain `{query}`. |
| `tools` | `Sequence[BaseTool \| Callable]` | `[]` | Tools available to the model. Plain functions are auto-wrapped into `FunctionTool`. |
| `name` | `str` | — | The name of the chain, used as the source name in responses. |
| `max_turns` | `int` | `50` | Maximum number of tool-call turns before the loop terminates. |
| `include_history` | `int` | — | Number of previous turns to include. `-1` means all. |
| `store_immediate_steps` | `bool` | `False` | Whether to store all intermediate steps or only the final response. |

## API Reference

::: aap_llamaindex.ChatCausalMultiTurnsChain
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Chain](../../../core/components/chain.md) — The base chain concept and `BaseCausalMultiTurnsChain` interface
- [LlamaIndex Overview](../overview.md) — LlamaIndex integration overview
- [LlamaIndex Documentation](https://docs.llamaindex.ai) — Official LlamaIndex documentation
