# Chain

## Overview

The `Chain` component provides the foundational abstractions for composing agents and LLM calls into structured, multi-step workflows. The `aap_core.chain` module defines two primary base classes: `TypicalLLMChain`, which implements a standard four-stage response pipeline (input guardrail → prompt augmentation → generation → output guardrail), and `BaseCausalMultiTurnsChain`, which implements a ReAct-style iterative loop (Thought → Act → Thought → … → Final) for multi-turn tool-use scenarios. Both classes inherit from `BaseLLMChain` and integrate with the `AgentMessage` type to ensure seamless interoperability with the rest of the framework.

## Architecture

Chains sit between individual agents and orchestration patterns in the component hierarchy. An agent processes a single `AgentMessage`; a chain composes multiple processing steps — potentially involving multiple agents, LLM calls, tools, and guardrails — into a single coherent workflow. Chains receive an `AgentMessage` and return a transformed `AgentMessage`, making them directly composable with other agents in orchestration patterns.

The two chain types serve different purposes:

- **`TypicalLLMChain`** follows a linear, single-pass pipeline: the message flows through input validation, prompt enhancement, response generation, and output validation in sequence.
- **`BaseCausalMultiTurnsChain`** follows a causal, iterative loop: the model generates a response, tools are processed, and the cycle repeats until no further tool calls are needed or the maximum number of turns is reached.

![diagram](img/chain-architecture.jpg)

## Key Concepts

### TypicalLLMChain: The Four-Stage Pipeline

`TypicalLLMChain` implements a standard response generation pipeline with four distinct stages, each represented by a configurable component:

1. **Input Guardrail** (`input_guardrail`): Validates the incoming `AgentMessage` before any processing. By default, `PassGuardRail` passes messages through unchanged. Custom guardrails can reject, modify, or enrich messages at this stage.

2. **Prompt Augmenter** (`prompt_augmenter`): Enhances or rewrites the prompt by adding external context (e.g., RAG results, retrieved documents, structured data). The `BasePromptAugmenter` abstraction supports both data augmentation and structural rewriting, with optional looping via a count or stop condition.

3. **Generate** (`generate`): The core generation step — an abstract method that subclasses implement to produce the actual response. This step is responsible for calling the LLM, invoking tools, and finalizing the response.

4. **Output Guardrail** (`output_guardrail`): Validates the generated response before it is returned. Like the input guardrail, this defaults to `PassGuardRail` but can be customized for quality checks, safety filtering, or format validation.

### BaseCausalMultiTurnsChain: The ReAct Loop

`BaseCausalMultiTurnsChain` implements a ReAct (Reason + Act) iterative pattern, where the model alternates between thinking and acting until a final response is reached:

1. **Prepare Conversation** (`_prepare_conversation`): Converts the initial `AgentMessage` into a conversation history (a list of `ChainMessage` objects) that the LLM can process.

2. **Generate Response** (`_generate_response`): Calls the LLM with the current conversation. Returns the updated conversation, the response, a flag indicating whether tool calls are needed, and token usage information.

3. **Process Tools** (`_process_tools`): If the response contains tool calls, this method executes them and appends the results back to the conversation.

4. **Append Responses** (`_append_responses`): Once the loop terminates (no more tools or max turns reached), this method converts the conversation back into an `AgentMessage`.

The loop is bounded by `max_turns` (default: 50) to prevent infinite execution. The `_last_response_as_context` mechanism allows the final response to be stored and later injected as context into the message.

### Token Usage Tracking

Both chain types track token usage at each step. The `TokenUsage` type records `input_tokens`, `output_tokens`, and `total_tokens` per step, with cumulative totals maintained across the entire chain execution. This information is stored in the `AgentMessage.token_usage` field, enabling monitoring and cost analysis.

## Usage

### Basic Example: TypicalLLMChain

```python
from aap_core import AgentMessage, TypicalLLMChain
from aap_core.guardrail import PassGuardRail
from aap_core.prompt_augmenter import IdentityPromptAugmenter


class MyLLMChain(TypicalLLMChain):
    def generate(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Call an LLM, invoke tools, finalize response
        message.responses.append(("my-chain", f"Response to: {message.query}"))
        message.execution_result = "success"
        return message


# Instantiate with default guardrails and augmenter
chain = MyLLMChain(name="my-chain")

# Invoke the chain
message = AgentMessage(query="What is AI?")
result = chain.invoke(message)
print(result.responses)  # [("my-chain", "Response to: What is AI?")]
```

### Advanced Example: Multi-Turn ReAct Chain

`BaseCausalMultiTurnsChain` is a generic class parameterized by two type variables: `ChainMessage` (the message type used in the conversation list) and `ChainResponse` (the response type). Each integration framework substitutes its own concrete types — LangChain uses `BaseMessage` / `AIMessage`, LlamaIndex uses `ChatMessage` / `ChatResponse`, and Transformers uses its own message type.

Below is a minimal mock implementation that demonstrates the interface. In practice, you would extend this with an actual LLM backend (see the [LangChain](../../integrations/langchain/overview.md), [LlamaIndex](../../integrations/llamaindex/overview.md), or [Transformers](../../integrations/transformers/overview.md) integration docs for real implementations).

```python
from typing import List, Tuple

from aap_core.chain import BaseCausalMultiTurnsChain
from aap_core.types import AgentMessage, ChainMessage, ChainResponse, TokenUsage


class MyReActChain(BaseCausalMultiTurnsChain[ChainMessage, ChainResponse]):
    """A minimal ReAct chain for demonstration."""

    def _prepare_conversation(self, message: AgentMessage) -> List[ChainMessage]:
        # Convert the AgentMessage into a conversation list.
        # In practice, this would build a list of framework-specific message objects
        # (e.g., LangChain's BaseMessage, LlamaIndex's ChatMessage).
        return []

    def _generate_response(
        self, conversation: List[ChainMessage], **kwargs
    ) -> Tuple[List[ChainMessage], ChainResponse, bool, TokenUsage]:
        # Call the LLM with the conversation.
        # Return: updated conversation, response, has_tool flag, token usage.
        response = "Hello! How can I help?"
        has_tool = False  # Set to True if the LLM wants to call a tool
        usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
        return conversation, response, has_tool, usage

    def _process_tools(
        self,
        conversation: List[ChainMessage],
        response: ChainResponse,
    ) -> List[ChainMessage]:
        # Execute tool calls from the response and append results to conversation.
        # This is called only when has_tool is True.
        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[ChainMessage]
    ) -> AgentMessage:
        # Convert the final conversation back into an AgentMessage.
        message.responses.append(("my-react-chain", "final response"))
        message.execution_result = "success"
        return message


# Instantiate with custom settings
chain = MyReActChain(
    name="my-react-chain",
    max_turns=10,
    include_history=2,
    store_immediate_steps=True,
)

# Invoke the chain
message = AgentMessage(query="What is the weather?")
result = chain.invoke(message)
print(result.responses)       # [("my-react-chain", "final response")]
print(result.token_usage)     # Tracks tokens across all turns
print(result.origin)          # "my-react-chain"
```

### Using Guardrails and Prompt Augmentation

```python
from aap_core import AgentMessage, TypicalLLMChain
from aap_core.guardrail import PassGuardRail
from aap_core.prompt_augmenter import IdentityPromptAugmenter


class ValidatedChain(TypicalLLMChain):
    def generate(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append(("validated-chain", f"Answer: {message.query}"))
        message.execution_result = "success"
        return message


# The pipeline: input guardrail → prompt augmenter → generate → output guardrail
chain = ValidatedChain(
    name="validated-chain",
    input_guardrail=PassGuardRail(),      # Validate input
    prompt_augmenter=IdentityPromptAugmenter(),  # Enhance prompt
    output_guardrail=PassGuardRail(),     # Validate output
)

message = AgentMessage(query="Tell me about chains")
result = chain.invoke(message)
```

## Configuration

### TypicalLLMChain Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_guardrail` | `BaseGuardRail` | `PassGuardRail()` | Validates the input `AgentMessage` before processing. |
| `prompt_augmenter` | `BasePromptAugmenter` | `IdentityPromptAugmenter()` | Enhances or rewrites the prompt by adding external context. |
| `tools` | `Sequence[Callable]` | `[]` | Tools (callables) available for the LLM to invoke during generation. |
| `output_guardrail` | `BaseGuardRail` | `PassGuardRail()` | Validates the generated response before returning it. |

### BaseCausalMultiTurnsChain Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_history` | `int` | `0` | Number of history message turns to include in the conversation context. |
| `store_immediate_steps` | `bool` | `False` | Whether to store intermediate step outputs for debugging or analysis. |
| `max_turns` | `int` | `50` | Maximum number of ReAct loop iterations before the chain stops. |

## API Reference

See the full API reference: [`TypicalLLMChain`][aap_core.chain.TypicalLLMChain], [`BaseCausalMultiTurnsChain`][aap_core.chain.BaseCausalMultiTurnsChain]

::: aap_core.chain.TypicalLLMChain
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.chain.BaseCausalMultiTurnsChain
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Agent](agent.md) — The building block that chains compose
- [Types](types.md) — `AgentMessage`, `TokenUsage`, and other core types
- [Guardrail](guardrail.md) — Input/output validation components used by chains
- [Prompt Augmenter](prompt_augmenter.md) — Context enrichment components used by chains
- [Orchestration](orchestration.md) — Patterns for coordinating multiple agents and chains
