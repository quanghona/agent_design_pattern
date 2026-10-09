# DSPy Chain

## Overview

The DSPy chain provides a bridge between the AI Agent Pattern core framework and the [DSPy](https://dspy.ai/) programming framework. It implements `ChatCausalMultiTurnsChain`, which extends `BaseCausalMultiTurnsChain` to handle DSPy's signature-based programming model. The chain supports both fully-managed tool calling (e.g., `dspy.ReAct`) and manual tool handling patterns, making it flexible for a wide range of DSPy workflows.

## Architecture

The DSPy chain sits within the integration layer, translating between the framework-agnostic `AgentMessage` type and DSPy's `dspy.Signature` and `dspy.Prediction` types. It uses an adapter pattern (`BaseSignatureAdapter`) to manage the bidirectional conversion between these representations.

The chain integrates with DSPy's core abstractions:

- **`dspy.Signature`**: Defines the input and output fields for a DSPy program. The chain uses signatures to structure the conversation history and model inputs.
- **`dspy.Prediction`**: The output type from DSPy predictors. The chain extracts responses, tool calls, and token usage from predictions.
- **`dspy.Module`**: The predictor that the chain invokes. This can be any DSPy module, from simple predictors to complex multi-step programs.

![diagram](img/chain-architecture.jpg)

## Key Concepts

### Signature Adapter Pattern

The `BaseSignatureAdapter` class is the core abstraction that enables communication between the AI Agent Pattern framework and DSPy. It handles two conversions:

1. **`msg2sig`**: Converts `AgentMessage` objects into a list of `dspy.Signature` objects that represent the conversation history. This includes both static prefill values (set via `with_prefill`) and dynamic values from the message's context.

2. **`sig2msg`**: Converts DSPy `Prediction` outputs back into `AgentResponse` objects. The adapter determines whether a response is an assistant message or a tool message based on the presence of `OutputField` and `ToolCalls` in the signature.

The adapter supports a **prefill dictionary** — a set of static key-value pairs that are injected into every signature. This is useful for fields that don't exist in `AgentMessage` but are required by the DSPy signature, such as system instructions or retrieved context.

```python
adapter = MySignatureAdapter.with_prefill({
    "system_instruction": "You are a helpful assistant.",
    "retrieved_context": "Some static context...",
})
```

### Tool Calling Approaches

DSPy supports two distinct patterns for tool calling, and the chain handles both:

#### Approach 1: Fully Managed (e.g., `dspy.ReAct`)

When using DSPy modules that manage tool calling internally (like `dspy.ReAct`), the module's signature does **not** include a `dspy.ToolCalls` field. All tool execution happens inside the DSPy module, and the chain only receives the final output. In this case, `_process_tools` is never called.

```python
import dspy

# dspy.ReAct manages tools internally
predictor = dspy.ReAct("query -> answer", tools=[my_tool])
chain = ChatCausalMultiTurnsChain(
    predictor=predictor,
    adapter=MyAdapter(),
)
```

#### Approach 2: Manual Tool Handling

When the DSPy signature includes a `dspy.ToolCalls` output field, the chain takes over tool execution. After `_generate_response` produces a prediction with tool calls, the chain's `_process_tools` method executes each tool and appends the results back to the conversation, continuing the loop until no more tool calls are needed.

```python
# Signature with explicit ToolCalls field
class MySignature(dspy.Signature):
    query: dspy.InputField
    tool_calls: dspy.ToolCalls
    answer: dspy.OutputField

chain = ChatCausalMultiTurnsChain(
    signature=MySignature,
    predictor=my_predictor,
    adapter=MyAdapter(),
)
```

The chain automatically detects the presence of a `dspy.ToolCalls` field in the signature's output fields during initialization and sets the `_tool_calls_field` attribute accordingly.

### Language Model Context

The chain supports setting a custom `dspy.LM` via the `with_lm()` method. This allows you to swap the language model at runtime without recreating the chain:

```python
chain = ChatCausalMultiTurnsChain(
    predictor=predictor,
    adapter=MyAdapter(),
)
chain = chain.with_lm(dspy.LM("anthropic/claude-3-opus-20240229"))
```

When an LM is set, the chain uses `dspy.context` to apply it during prediction, with token usage tracking enabled.

### History Field Management

The chain automatically detects the `dspy.History` field in the signature during initialization. When present, it converts the history from a dictionary (in the signature) to a `dspy.History` object (required by the DSPy library) before passing it to the predictor.

## Usage

### Basic Example: Simple Predictor

```python
import dspy
from aap_dspy.chain import ChatCausalMultiTurnsChain
from aap_dspy.utils import MySignatureAdapter
from aap_core.types import AgentMessage


class MyAdapter(BaseSignatureAdapter):
    def msg2sig(self, message: AgentMessage) -> list[dspy.Signature]:
        # Convert AgentMessage to DSPy signatures
        ...

    def sig2msg(self, signatures: list[dspy.Signature], name: str) -> list[AgentResponse]:
        # Convert DSPy predictions back to AgentResponse
        ...


# Define a simple signature
class QASignature(dspy.Signature):
    """Answer the user's question."""
    query: dspy.InputField
    answer: dspy.OutputField

# Create a predictor
predictor = dspy.Predict(QASignature)

# Create the chain
chain = ChatCausalMultiTurnsChain(
    signature=QASignature,
    predictor=predictor,
    adapter=MyAdapter(),
    name="dspy-agent",
)

# Invoke with a message
message = AgentMessage(query="What is the capital of France?")
result = chain.invoke(message)
```

### Advanced Example: Multi-Turn with Tools

```python
import dspy
from aap_dspy.chain import ChatCausalMultiTurnsChain


# Define a signature with explicit tool calls
class ToolSignature(dspy.Signature):
    """Use tools to answer the user's question."""
    query: dspy.InputField
    tool_calls: dspy.ToolCalls
    answer: dspy.OutputField


# Create a predictor that uses the signature
predictor = MyCustomModule()  # Any dspy.Module

# Create the chain with manual tool handling
chain = ChatCausalMultiTurnsChain(
    signature=ToolSignature,
    predictor=predictor,
    adapter=MyAdapter(),
    name="dspy-tool-agent",
    max_turns=10,
)

# Set a specific language model
chain = chain.with_lm(dspy.LM("google/gemini-2.0-flash"))

# Invoke — the chain will loop through tool calls automatically
message = AgentMessage(query="What's the weather in Tokyo?")
result = chain.invoke(message)
```

### Using Prefill Values

```python
# Create an adapter with static prefill values
adapter = MyAdapter.with_prefill({
    "system_instruction": "You are a helpful chemistry tutor.",
    "domain": "chemistry",
})

chain = ChatCausalMultiTurnsChain(
    signature=MySignature,
    predictor=predictor,
    adapter=adapter,
)

# Add or remove prefill values at runtime
adapter.add_prefill("retrieved_context", "Some retrieved document...")
adapter.remove_prefill("domain")
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictor` | `dspy.Module` | *(required)* | The DSPy predictor module to invoke. Can be a `dspy.Predict`, `dspy.ReAct`, or any custom `dspy.Module`. |
| `adapter` | `BaseSignatureAdapter` | *(required)* | The signature adapter that converts between `AgentMessage` and `dspy.Signature`. |
| `signature` | `str \| type[dspy.Signature]` | *(required)* | The DSPy signature defining input/output fields. Can be a string or a signature class. |
| `name` | `str` | — | The name of the chain, used as the source name in responses. |
| `max_turns` | `int` | `50` | Maximum number of tool-call turns before the loop terminates. |
| `include_history` | `int` | — | Number of previous turns to include in the conversation. `-1` means all. |
| `store_immediate_steps` | `bool` | `False` | Whether to store all intermediate steps or only the final response. |
| `with_lm()` | `dspy.LM \| None` | `None` | Sets a custom language model via the chain's fluent API. |

## API Reference

::: aap_dspy.ChatCausalMultiTurnsChain
    options:
        show_root_heading: true
        show_signature: true

::: aap_dspy.BaseSignatureAdapter
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Chain](../../../core/components/chain.md) — The base chain concept and `BaseCausalMultiTurnsChain` interface
- [DSPy Overview](../overview.md) — DSPy integration overview
- [DSPy Documentation](https://dspy.ai) — Official DSPy framework documentation
