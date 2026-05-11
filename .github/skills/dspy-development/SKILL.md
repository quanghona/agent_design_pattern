---
name: dspy-development
description: "A structured workflow for designing, implementing, and testing new features using the DSPy framework within the `aap_dspy` package. This skill ensures that all new components are compatible with the `aap_core` architecture and follow DSPy best practices."
---

# DSPy Feature Development Skill

## Overview

This skill provides a structured workflow for designing, implementing, and testing new features using the DSPy framework within the `aap_dspy` package. For comprehensive DSPy best practices (signatures, modules, optimization, guardrails, production patterns, and testing), refer to [best-practices.md](best-practices.md).

## Workflow

### 1. Research & Design
- **Consult Best Practices**: Review [best-practices.md](best-practices.md) for signature design patterns, module selection, and optimization strategies.
- **Identify DSPy Primitives**: Determine which DSPy components are required:
  - **Signatures**: Define the input/output contract (e.g., `dspy.Signature`).
  - **Modules**: Define the computational steps (e.g., `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`).
  - **Optimizers**: Select an algorithm to tune the program (e.g., `BootstrapFewShot`, `MIPROv2`).
  - **Metrics**: Define the evaluation function to guide optimization.
- **Design the Adapter**: Plan the `BaseSignatureAdapter` implementation:
  - `msg2sig`: How to map `AgentMessage` (including context) to `dspy.Signature` fields.
  - `sig2msg`: How to map `dspy.Prediction` back to `AgentResponse`.
- **Define Data Flow**: Map how `AgentMessage` fields will be transformed into Signature fields and how the results will be converted back.
- **Follow Python Standards**: Adhere to `@file:python-development.instructions.md`.

### 2. Component Implementation
- **Consult Best Practices**: Review [best-practices.md](best-practices.md) for module composition, custom module patterns, and guardrail implementation.
- **Implement Signature Adapters**:
  - Implement `msg2sig` to handle both static (prefill) and dynamic (context) data.
  - Implement `sig2msg` to extract assistant or tool responses from the prediction.
- **Implement DSPy Modules**:
  - Create custom `dspy.Module` subclasses if standard ones are insufficient.
  - Ensure modules are compatible with the `ChatCausalMultiTurnsChain` wrapper.
- **Handle Tool Calling**:
  - Decide between **Fully Managed** (using `dspy.ReAct` where tools stay inside the module) and **Manual Tool Handling** (where `aap_dspy` manages the tool execution loop).
- **Integration with `aap_core`**: Ensure the `ChatCausalMultiTurnsChain` correctly manages the `dspy.History` if required by the signature.

### 3. Optimization (The DSPy Way)
- **Consult Best Practices**: Review [best-practices.md](best-practices.md) for optimizer selection, teacher-student pattern, and metric design.
- **Define a Metric**: Create a robust evaluation function that takes a `dspy.Example` and returns a score.
- **Setup Training Data**: Prepare a small set of `dspy.Example` objects for the optimizer.
- **Run Optimization**: Use a DSPy optimizer to compile the program.
- **Verify Compilation**: Ensure the compiled program (with optimized prompts/weights) performs better than the baseline.

### 4. Testing (DSPy Specific)
Follow `@file:writing-and-running-tests.instructions.md` and apply these DSPy-specific strategies. See [best-practices.md](best-practices.md) for testing guidelines and evaluation patterns.

- **Mocking the LM**:
  - **NEVER** use real LLM calls in unit tests.
  - Use `dspy.evaluate.spy` or mock the `dspy.LM` object to return deterministic `dspy.Prediction` objects.
- **Adapter Testing**:
  - Test `msg2sig` with various `AgentMessage` states (empty context, full history, etc.).
  - Test `sig2msg` to ensure it correctly distinguishes between assistant messages and tool messages based on the presence of `ToolCalls`.
- **Signature Validation**:
  - Verify that the `ChatCausalMultiTurnsChain` correctly identifies `dspy.History` and `dspy.ToolCalls` fields in the signature.
- **Module Logic Testing**:
  - Test the internal logic of custom `dspy.Module` implementations using mocked predictions.

### 5. Validation & Cleanup
- **Lint & Format**: Run `uv run ruff check --fix` and `uv run ruff format`.
- **Full Test Suite**: Run `uv run pytest -v` within the `src/dspy` directory.

## Implementation Examples

### 1. Implementing a Signature Adapter
```python
import dspy
from aap_core.types import AgentMessage, AgentResponse
from aap_dspy.chain import BaseSignatureAdapter

class MyTaskSignature(dspy.Signature):
    """Answer the question based on context."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

class MyAdapter(BaseSignatureAdapter[MyTaskSignature]):
    def msg2sig(self, message: AgentMessage) -> list[MyTaskSignature]:
        # Map AgentMessage to Signature
        return [MyTaskSignature(
            question=message.query,
            context=message.context.get("data", "")
        )]

    def sig2msg(self, signatures: list[MyTaskSignature], name: str) -> list[AgentResponse]:
        # Map Signature back to AgentResponse
        return [AgentResponse(content=sig.answer) for sig in signatures]
```

### 2. Testing the Adapter
```python
import pytest
from aap_core.types import AgentMessage

def test_adapter_msg2sig():
    adapter = MyAdapter()
    msg = AgentMessage(query="What is AI?", context={"data": "AI is intelligence."})

    signatures = adapter.msg2sig(msg)

    assert len(signatures) == 1
    assert signatures[0].question == "What is AI?"
    assert signatures[0].context == "AI is intelligence."
```

### 3. Implementing a Prompt Augmenter
```python
import dspy
from aap_core.types import AgentMessage
from aap_core.prompt_augmenter import BasePromptAugmenter
from aap_dspy.chain import BaseSignatureAdapter

class RewriteSignature(dspy.Signature):
    """Rewrite the user query to be more descriptive."""
    query = dspy.InputField()
    rewritten_query = dspy.OutputField()

class DSPyPromptAugmenter(BasePromptAugmenter):
    def __init__(self, module: dspy.Module, adapter: BaseSignatureAdapter, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.adapter = adapter

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Map AgentMessage to Signature inputs
        signatures = self.adapter.msg2sig(message)

        # Run the DSPy module
        predictions = self.module(signatures)

        # Update the message with the rewritten query
        message.query = predictions[0].rewritten_query
        return message
```

### 4. Testing the Module
```python
import pytest
import dspy
from unittest.mock import MagicMock

def test_dspy_module():
    # Arrange
    mock_module = MagicMock()
    mock_module.return_value = dspy.Prediction(rewritten_query="What is artificial intelligence?")

    # Act
    adapter = DSPyPromptAugmenter(module=mock_module, adapter=MyAdapter())
    msg = AgentMessage(query="What is AI?")
    result = adapter.augment(msg)

    # Assert
    assert result.query == "What is artificial intelligence?"
```
