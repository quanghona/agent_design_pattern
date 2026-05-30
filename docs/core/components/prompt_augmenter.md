# Prompt Augmenter

## Overview

The `PromptAugmenter` component enhances or rewrites prompts before they are sent to an LLM. It sits between the input guardrail and the generation stage in a `TypicalLLMChain` pipeline, transforming `AgentMessage` objects by adding external context or restructuring the prompt text. There are two broad categories of prompt augmentation: **data augmentation**, which injects external information such as retrieved documents, structured data, or RAG results into the prompt; and **structural augmentation**, which rewrites or refines the prompt text itself, potentially using an LLM to generate a better-formulated version.

## Architecture

Prompt augmenters inherit from `BaseChain` and integrate directly into the chain pipeline. When a chain executes, the message flows through the input guardrail, then through the configured `BasePromptAugmenter`, then through the generation step, and finally through the output guardrail. The augmenter receives an `AgentMessage` and returns a modified `AgentMessage` with an updated `query` field.

The framework provides four concrete implementations:

- **`IdentityPromptAugmenter`** — a no-op default that passes messages through unchanged.
- **`SimplePromptAugmenter`** — concatenates the original prompt with external data using a configurable template.
- **`MetaPromptAugmenter`** — delegates prompt rewriting to an LLM chain.
- **`DeduplicationPromptAugmenter`** — removes exact and near-duplicate sentences from the prompt (documented separately).

![diagram](img/prompt_augmenter-architecture.jpg)

## Key Concepts

### Loop Control

All prompt augmenters support optional looping via the `loop` parameter on `BasePromptAugmenter`. This allows the same augmenter to be applied multiple times:

- **Integer**: Apply the augmenter a fixed number of times.
- **Callable**: Accept a function `AgentMessage -> bool` that returns `True` to continue looping and `False` to stop.
- **`None`** (default): Apply the augmenter exactly once.

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import SimplePromptAugmenter

# Apply augmentation 3 times
augmenter = SimplePromptAugmenter(
    format="{query}\n\nAdditional context:\n{data}",
    data_key="context.data",
    loop=3,
)

# Apply until the query length exceeds 500 characters
augmenter = SimplePromptAugmenter(
    format="{query}\n\n{data}",
    data_key="context.data",
    loop=lambda msg: len(msg.query) < 500,
)
```

### Data Augmentation with SimplePromptAugmenter

`SimplePromptAugmenter` is the simplest data augmentation strategy. It takes the original prompt text and concatenates it with external data stored in the `AgentMessage.context` dictionary, using a user-defined format string. The format string must contain both `{query}` (the original prompt) and `{data}` (the external data). Additional keyword arguments passed to `augment()` are also interpolated into the format string.

The `data_key` parameter specifies where to find the data in `message.context`. It must start with the prefix `context.` — for example, `context.data` reads from `message.context["data"]`. This convention allows the same format string to reference multiple data sources if needed.

### Structural Augmentation with MetaPromptAugmenter

`MetaPromptAugmenter` delegates prompt rewriting to an LLM chain. Instead of template-based concatenation, it passes the `AgentMessage` through a `BaseLLMChain` (such as a LangChain or LlamaIndex chain) that can reason about the prompt and produce a rewritten version. This is useful when the prompt needs semantic restructuring, tone adjustment, or intelligent context selection that a simple template cannot achieve.

## Usage

### Basic Example: Identity Prompt Augmenter

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import IdentityPromptAugmenter

# Identity augmenters pass messages through unchanged
augmenter = IdentityPromptAugmenter()

message = AgentMessage(query="What is AI?")
result = augmenter(message)
print(result.query)  # "What is AI?"
```

### Basic Example: Simple Data Augmentation

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import SimplePromptAugmenter

# Define the format: original query + external data
augmenter = SimplePromptAugmenter(
    format="Question: {query}\n\nContext:\n{data}",
    data_key="context.retrieved_docs",
)

message = AgentMessage(
    query="What is reinforcement learning?",
    context={
        "retrieved_docs": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving rewards."
    },
)
result = augmenter(message)
print(result.query)
# "Question: What is reinforcement learning?\n\nContext:\nReinforcement learning is a type of machine learning..."
```

### Advanced Example: Simple Augmenter with Multiple Data Sources

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import SimplePromptAugmenter

# Format string can include additional kwargs beyond query and data
augmenter = SimplePromptAugmenter(
    format="You are an expert. Answer the question using the provided context.\n\nQuestion: {query}\n\nContext:\n{data}\n\nConstraints: {constraints}",
    data_key="context.knowledge_base",
)

message = AgentMessage(
    query="Explain quantum entanglement.",
    context={
        "knowledge_base": "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated in such a way that the quantum state of each particle cannot be described independently.",
    },
)
result = augmenter(
    message,
    constraints="Keep the explanation under 200 words and avoid mathematical notation.",
)
print(result.query)
# "You are an expert. Answer the question using the provided context.
#
# Question: Explain quantum entanglement.
#
# Context:
# Quantum entanglement is a physical phenomenon...
#
# Constraints: Keep the explanation under 200 words..."
```

### Advanced Example: Meta Prompt Augmenter with an LLM Chain

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import MetaPromptAugmenter

# Assume you have a LangChain or LlamaIndex chain configured
# with a prompt template that instructs the LLM to rewrite the prompt
from aap_langchain import LangChainLLMChain

llm_chain = LangChainLLMChain(
    # ... configure your LLM and prompt template ...
    name="prompt-rewriter",
)

augmenter = MetaPromptAugmenter(chain=llm_chain)

message = AgentMessage(query="tell me about ai")
result = augmenter(message)
# The LLM chain rewrites the query to be more structured and detailed
print(result.query)
# "Please provide a comprehensive explanation of artificial intelligence, including its key subfields..."
```

## Configuration

### BasePromptAugmenter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loop` | `int \| Callable[[AgentMessage], bool] \| None` | `None` | Loop control: an integer for fixed iterations, a callable for conditional looping, or `None` for single application |

### IdentityPromptAugmenter

No additional parameters. Inherits `loop` from `BasePromptAugmenter`.

### SimplePromptAugmenter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str` | *(required)* | Format string for the augmented prompt. Must contain `{query}` and `{data}`. Additional placeholders are filled by kwargs passed to `augment()`. |
| `data_key` | `str` | `"context.data"` | Key path in `message.context` to read the external data. Must start with `context.`. For example, `context.retrieved_docs` reads from `message.context["retrieved_docs"]`. |

**Validators:**
- `format` must contain both `{query}` and `{data}` placeholders.
- `data_key` must start with the prefix `context.`.

### MetaPromptAugmenter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `BaseLLMChain` | *(required)* | An LLM chain that rewrites the prompt. The chain's `invoke()` method is called with the `AgentMessage` and returns the modified message. |

## API Reference

See the full API reference: [`BasePromptAugmenter`][aap_core.prompt_augmenter.BasePromptAugmenter]

::: aap_core.prompt_augmenter.BasePromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.IdentityPromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.SimplePromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.MetaPromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Chain](chain.md) — How prompt augmenters integrate into the `TypicalLLMChain` pipeline
- [Retriever](retriever.md) — How retrieved data is stored in `AgentMessage.context` for augmentation
- [Dedup](dedup.md) — The `DeduplicationPromptAugmenter` for removing duplicate sentences from prompts
