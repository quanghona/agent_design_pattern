---
name: transformers-development
description: "A structured workflow for designing, implementing, and testing new features using the Hugging Face Transformers framework within the `aap_transformers` package. This skill ensures that all new components are compatible with the `aap_core` architecture and follow Transformers best practices."
---

# Transformers Development Skill

## Overview

This skill provides a structured workflow for designing, implementing, and testing new features using the Hugging Face Transformers framework within the `aap_transformers` package. For comprehensive Transformers best practices (model selection, inference optimization, fine-tuning, quantization, production deployment, and testing), refer to [best-practices.md](best-practices.md).

## Core Concepts & API Reference

Hugging Face Transformers is a library for state-of-the-art Machine Learning models. When developing within `aap_transformers`, focus on these core abstractions:

### 1. Model Loading & Configuration
- **AutoClasses**: Use `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`, etc., for easy model discovery and loading.
- **Tokenizer**: Use `AutoTokenizer` to handle text preprocessing and chat templates.
- **Configuration**: `AutoConfig` manages model-specific hyperparameters.

### 2. Inference & Generation
- **Pipeline API**: A high-level abstraction for common tasks (e.g., `text-generation`, `summarization`).
- **Generation Config**: Controls decoding strategies like beam search, temperature, and top-p/top-k sampling.
- **Chat Templates**: Use `tokenizer.apply_chat_template` to format conversation history into the specific format required by the model.

### 3. Tool Calling & Function Calling
- **Chat Templates with Tools**: Many modern models support tool calling via specialized chat templates.
- **Structured Output**: Leveraging models that can output JSON or following specific patterns (like `<tool_call>...</tool_call>`) to facilitate agentic behavior.

## Integration with `aap_core`

Our framework uses `aap_core` as the foundational layer. Follow these integration patterns:

### 1. Chain Implementation
Implement `BaseCausalMultiTurnsChain` from `aap_core` to create Transformers-powered chains.
- **Mapping**: Map Transformers' conversation formats (often lists of dicts with `role` and `content`) to `aap_core`'s `AgentMessage` and `ChatMessage`.
- **Chat Templates**: Use `tokenizer.apply_chat_template` within the chain to ensure the model receives correctly formatted input.
- **Tool Integration**: If the model supports tool calling via chat templates, ensure the `tools` are passed correctly to the tokenizer/model during generation.

### 2. Retriever Adapter
If using Hugging Face models for retrieval (e.g., via `SentenceTransformers`), wrap them in a `RetrieverAdapter` to bridge them with `aap_core`.

### 3. Token Usage
Ensure that token usage (input, output, total) is correctly extracted from the model's generation output and converted to `aap_core.types.TokenUsage`.

## Development Workflow

1.  **Consult Best Practices**: Review [best-practices.md](best-practices.md) for model selection, inference optimization, and production deployment patterns.
2.  **Define the Goal**: Determine if you are building a new model-based chain, a specialized retriever, or a tool-calling agent.
3.  **Research Model Capabilities**: Check the Hugging Face Hub for the model's capabilities (e.g., does it support chat templates? Does it support tool calling?).
3.  **Implement the Component**:
    - For chains, extend `ChatCausalMultiTurnsChain`.
    - For retrievers, implement `RetrieverAdapter`.
    - **Follow Python Standards**: Adhere to `@file:python-development.instructions.md`.
4. **Integrate with `aap_core`**: Ensure all inputs and outputs are correctly mapped to `AgentMessage` and `AgentResponse`.
5. **Testing (Transformers Specific):**
Follow `@file:writing-and-running-tests.instructions.md` and apply these Transformers-specific strategies. See [best-practices.md](best-practices.md) for testing guidelines and evaluation patterns.

- **Mocking Models**:
  - **NEVER** load large models or make real inference calls in unit tests.
  - Use `unittest.mock` to mock `AutoModelForCausalLM.from_pretrained` and the model's `generate` method.
  - Return mocked `ModelOutput` or `GenerateOutput` objects that contain the expected text and token usage.
- **Chat Template Testing**:
  - Verify that `tokenizer.apply_chat_template` is called with the correct parameters.
  - Test that the resulting prompt correctly incorporates the `system_prompt` and conversation history.
- **Tool Call Extraction Testing**:
  - Test the parser's ability to extract tool calls from various model output formats (e.g., XML tags, JSON blocks).
  - Verify that extracted tool calls are correctly converted into `aap_core` tool call structures.
- **Token Usage Testing**:
  - Ensure that the extracted `TokenUsage` matches the expected values from the mocked model output.

## Example: Implementing a Transformers Chain

```python
from aap_transformers.chain import ChatCausalMultiTurnsChain
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load Model and Tokenizer
model_id = "meta-llama/Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 2. Implement the AAP Chain
chain = ChatCausalMultiTurnsChain(
    model=(model, tokenizer),
    device="cuda",
    system_prompt="You are a helpful AI assistant.",
    user_prompt_template="User: {query}\nAssistant:",
    tools=[] # Add tools here
)

# 3. Execute (following aap_core patterns)
response = chain.generate(message)
```

## Example: Implementing a Prompt Augmenter

```python
from aap_transformers.chain import ChatCausalMultiTurnsChain
from aap_core.prompt_augmenter import BasePromptAugmenter
from aap_core.types import AgentMessage

class TransformersPromptAugmenter(BasePromptAugmenter):
    def __init__(self, chain: ChatCausalMultiTurnsChain, **kwargs):
        super().__init__(**kwargs)
        self.chain = chain

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Use the Transformers chain to rewrite the query
        response = self.chain.call(message)

        # Update the message with the augmented query
        message.query = response.responses[0][1]
        return message
```

### 3. Testing the Chain
```python
import pytest
from unittest.mock import MagicMock, patch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_transformers_chain():
    # Arrange
    with patch.object(AutoModelForCausalLM, 'from_pretrained') as mock_from_pretrained:
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "AI is intelligence."

        # Act
        chain = ChatCausalMultiTurnsChain(
            model=(mock_model, mock_tokenizer),
            device="cpu",
            system_prompt="You are helpful.",
        )
        msg = AgentMessage(query="What is AI?")
        response = chain.generate(msg)

        # Assert
        assert response is not None
        mock_model.generate.assert_called_once()
```

## Example: Extracting Tool Calls

If the model uses a custom pattern for tool calls (e.g., XML tags), implement a parser within your chain:

```python
def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    # Example pattern: <tool_call>{"name": "my_tool", "args": {"param": "val"}}</tool_call>
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [json.loads(m) for m in matches]
```
