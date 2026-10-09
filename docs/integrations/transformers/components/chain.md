# Transformers Chain

## Overview

The Transformers chain provides a bridge between the AI Agent Pattern core framework and the [Hugging Face Transformers](https://huggingface.co/transformers/) library. It implements `ChatCausalMultiTurnsChain` to run local causal language models with built-in tool calling support. The chain uses the model's tokenizer chat template for conversation formatting and extracts tool calls from model output using regex pattern matching on the `<tool_call>` format.

## Architecture

The Transformers chain sits within the integration layer, translating between the framework-agnostic `AgentMessage` type and a simple dictionary-based message format (`TransformersChainMessage`). It manages both the model and tokenizer internally, loading them from a model identifier or a pre-loaded `(tokenizer, model)` tuple.

The chain integrates with Transformers' core abstractions:

- **`AutoModelForCausalLM`**: The Hugging Face causal language model. The chain loads and manages the model on the specified device (CPU or CUDA).
- **`AutoTokenizer`**: The tokenizer that converts conversations into model inputs. It uses `apply_chat_template()` with tool support for proper message formatting.
- **`TransformersChainMessage`**: A `TypedDict` with `role` and `content` keys that represents each message in the conversation.

![diagram](img/chain-architecture.jpg)

## Key Concepts

### Conversation Preparation

The `_prepare_conversation` method builds a list of `TransformersChainMessage` dictionaries from an `AgentMessage`. The conversation starts with a `system` message followed by a `user` message. Previous turns are appended with role mapping:

| Response Type | Transformers Role |
|---------------|-------------------|
| `"user"` | `"user"` |
| `"assistant"` (default) | `"assistant"` |
| `"tool"` | `"tool"` |
| `"system"` | `"system"` |

The `include_history` parameter controls how many previous turns are included.

### Model Input Generation

The chain uses the tokenizer's `apply_chat_template()` method to convert the conversation into model inputs:

```python
inputs = self._tokenizer.apply_chat_template(
    conversation,
    tools=self.tools,          # Tools passed to the template for proper formatting
    add_generation_prompt=True, # Adds the assistant prompt token
    return_dict=True,
    return_tensors="pt",
)
```

The `tools` parameter ensures the chat template includes proper tool definitions in the formatted prompt, enabling the model to understand available tools.

### Token Usage Tracking

The chain tracks token usage precisely:

- **Input tokens**: `inputs.input_ids.shape[-1]` — the length of the tokenized conversation.
- **Output tokens**: The length of the generated tokens beyond the input — `output[:, input_tokens:].shape[-1]`.
- **Total tokens**: Sum of input and output tokens.

This information is stored in a `TokenUsage` object and attached to the resulting `AgentMessage`.

### Tool Call Extraction

The Transformers chain uses regex-based extraction to find tool calls in the model's output. It looks for the `<tool_call>(?s:.*?)<\/tool_call>` pattern in the generated text. Each match is stripped of its delimiters and parsed as JSON to extract the tool name and arguments.

```python
# From the chain source
tool_calls = ChatCausalMultiTurnsChain.extract_json(response)
# Returns: [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
```

If the model output contains no `<tool_call>` markers, the chain treats the entire output as a plain text response and terminates the loop.

### Tool Execution

When tool calls are extracted, `_process_tools` executes each one:

1. Parses the tool call JSON to get the tool name and arguments.
2. Looks up the tool name in `_tool_dict`.
3. If found, invokes `tool_func(**arguments)` and appends a `tool` message with the result.
4. If not found or an error occurs, appends a `tool` message with an error description.

### Device Management

The chain requires a `device` parameter (`"cpu"` or `"cuda"`) that determines where the model runs. The model is loaded once during initialization and stays on the specified device throughout the chain's lifetime. You can also replace the model at runtime via the `model` property setter, which reloads both the tokenizer and model from the new path or tuple.

## Usage

### Basic Example: Simple Chat

```python
from aap_transformers.chain import ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage


chain = ChatCausalMultiTurnsChain(
    model="meta-llama/Llama-3.1-8B-Instruct",
    system_prompt="You are a helpful assistant.",
    user_prompt_template="Answer the following: {query}",
    device="cuda",
    name="transformers-agent",
)

message = AgentMessage(query="What is the capital of France?")
result = chain.invoke(message)
print(result.responses)
```

### Advanced Example: With Tools

```python
from aap_transformers.chain import ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage


def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"The weather in {location} is sunny, 22°C."


chain = ChatCausalMultiTurnsChain(
    model="meta-llama/Llama-3.1-8B-Instruct",
    system_prompt="You are a weather assistant. Use tools to find information.",
    tools=[get_weather],
    device="cuda",
    name="weather-assistant",
    max_turns=10,
)

message = AgentMessage(query="What's the weather in Tokyo?")
result = chain.invoke(message)
```

### Using a Pre-loaded Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from aap_transformers.chain import ChatCausalMultiTurnsChain


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="cuda",
)

chain = ChatCausalMultiTurnsChain(
    model=(tokenizer, model),  # Pass pre-loaded tuple
    system_prompt="You are a helpful assistant.",
    device="cuda",
    name="transformers-agent",
)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| os.PathLike \| Tuple[Any, Any]` | *(required)* | Model identifier string (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`), path to a local model, or a `(tokenizer, model)` tuple. |
| `device` | `Literal["cpu", "cuda"]` | *(required)* | Device to run the model on. `"cuda"` for GPU, `"cpu"` for CPU. |
| `system_prompt` | `str` | *(required)* | The system prompt that sets the assistant's behavior. |
| `user_prompt_template` | `str` | *(required)* | Template for formatting the user's query. Must contain `{query}`. |
| `tools` | `Sequence[Callable]` | `[]` | Tools available to the model. Plain functions are stored by their `__name__`. |
| `name` | `str` | — | The name of the chain, used as the source name in responses. |
| `max_turns` | `int` | `50` | Maximum number of tool-call turns before the loop terminates. |
| `include_history` | `int` | — | Number of previous turns to include. `-1` means all. |
| `store_immediate_steps` | `bool` | `False` | Whether to store all intermediate steps or only the final response. |

## API Reference

::: aap_transformers.ChatCausalMultiTurnsChain
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Chain](../../../core/components/chain.md) — The base chain concept and `BaseCausalMultiTurnsChain` interface
- [Transformers Overview](../overview.md) — Transformers integration overview
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers) — Official Transformers documentation