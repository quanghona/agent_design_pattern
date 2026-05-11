# DSPy Best Practices

A comprehensive guide of best practices for building production-ready DSPy applications, synthesized from industry leaders, technical experts, and real-world implementations.

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Signatures](#signatures)
- [Modules](#modules)
- [Optimization](#optimization)
- [Guardrails](#guardrails)
- [Production Patterns](#production-patterns)
- [Testing](#testing)
- [Common Pitfalls](#common-pitfalls)

---

## Core Philosophy

### 1. Programming Over Prompting

DSPy represents a fundamental shift from **prompting** to **programming** language models. Instead of manually crafting prompts, you define what you want to achieve, and DSPy automatically generates and optimizes the prompts for you.

| Aspect | Traditional Prompting | DSPy |
|--------|----------------------|------|
| Development Time | Hours/days per prompt | Minutes to define task |
| Model Portability | Rewrite for each model | Works across all LLMs |
| Performance Optimization | Manual trial & error | Automatic optimization |
| Consistency | Varies between runs | Predictable outputs |
| Composability | Difficult to combine | Modular & chainable |
| Production Readiness | Fragile at scale | Built for production |

### 2. Three Core Abstractions

1. **Signatures**: Declarative task specifications that define input/output contracts without implementation details
2. **Modules**: Composable building blocks with different reasoning strategies (Chain-of-Thought, ReAct, etc.)
3. **Optimizers**: Automatic prompt improvement algorithms that learn from examples

### 3. Model Portability

Develop locally with free models and seamlessly switch to production models without changing your code:

```python
# Local development (free, no rate limits)
dspy.configure(lm=dspy.LM('ollama/llama3'))

# Production (switch without code changes)
dspy.configure(lm=dspy.LM('openai/gpt-4o'))
```

---

## Signatures

### 1. String-based Signatures (Prototyping)

String signatures are the quickest way to get started. They use a simple arrow notation that's intuitive and readable.

```python
# Basic question answering
"question -> answer"

# Sentiment analysis with type specification
"sentence -> sentiment: bool"

# Multi-input example
"context, question -> answer"

# Multiple outputs
"document -> summary, keywords, sentiment"
```

### 2. Class-based Signatures (Production)

Class-based signatures provide rich metadata that DSPy uses to generate more accurate and reliable prompts.

```python
class QA(dspy.Signature):
    """Answer questions based on provided context."""
    context: str = dspy.InputField(
        desc="Background information that may contain the answer"
    )
    question: str = dspy.InputField(
        desc="The question to be answered"
    )
    answer: str = dspy.OutputField(
        desc="A concise, accurate answer based on the context"
    )
```

**Pro Tip**: The quality of your descriptions directly impacts the quality of DSPy's generated prompts. Be specific and clear about what you expect. Good descriptions lead to 30-40% better performance!

### 3. Signature Design Patterns

**Pattern 1: Structured Output**
```python
class StructuredAnalysis(dspy.Signature):
    """Analyze text and return structured data."""
    text: str = dspy.InputField()
    entities: list[dict] = dspy.OutputField(
        desc="List of {name, type, confidence} dictionaries"
    )
```

**Pattern 2: Conditional Logic**
```python
class ConditionalResponse(dspy.Signature):
    """Provide different responses based on input type."""
    query: str = dspy.InputField()
    query_type: str = dspy.OutputField(
        desc="One of: factual, opinion, action"
    )
    response: str = dspy.OutputField(
        desc="Appropriate response based on query type"
    )
```

**Pattern 3: Multi-step Processing**
```python
class MultiStepAnalysis(dspy.Signature):
    """Perform multi-step analysis on input."""
    input_data: str = dspy.InputField()
    step1_analysis: str = dspy.OutputField()
    step2_analysis: str = dspy.OutputField()
    final_conclusion: str = dspy.OutputField()
```

---

## Modules

### 1. Module Selection Guide

| Module | Use Case | Example |
|--------|----------|---------|
| `dspy.Predict` | Simple input/output transformation | `dspy.Predict("question -> answer")` |
| `dspy.ChainOfThought` | Tasks requiring step-by-step reasoning | `dspy.ChainOfThought("question -> answer")` |
| `dspy.ReAct` | Tasks requiring tool use and multi-step reasoning | `dspy.ReAct("question -> answer", tools=[...])` |
| `dspy.ProgramOfThought` | Tasks requiring code execution | `dspy.ProgramOfThought("question -> answer")` |
| `dspy.AutoChain` | Automatic chain-of-thought generation | `dspy.AutoChain("question -> answer")` |

### 2. Module Composition

```python
class MathQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define the module using Chain-of-Thought reasoning
        self.solve = dspy.ChainOfThought("question -> answer: float")
        self.verify = dspy.Predict("question, answer -> verified: bool")

    def forward(self, question: str):
        result = self.solve(question=question)
        verified = self.verify(question=question, answer=result.answer)
        return result
```

### 3. Custom Modules

For complex logic, create custom `dspy.Module` subclasses:

```python
class CustomModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step1 = dspy.ChainOfThought("input -> intermediate")
        self.step2 = dspy.Predict("intermediate -> output")

    def forward(self, input: str):
        intermediate = self.step1(input=input)
        return self.step2(intermediate=intermediate.intermediate)
```

---

## Optimization

### 1. The Optimization Paradigm

DSPy represents a shift from **Prompt Engineering** (manual trial-and-error) to **Prompt Compilation** (automated optimization). Instead of writing 100-line prompts and iterating based on vibes, you define what you want the model to do (via Signatures) and let DSPy find the optimal prompt tokens through optimization.

### 2. Optimizer Selection Guide

| Optimizer | Use Case | Data Required |
|-----------|----------|---------------|
| `BootstrapFewShot` | Simple few-shot example generation | 3-10 examples |
| `BootstrapFewShotWithRandomSearch` | Improved performance with random search | 10-20 examples |
| `MIPROv2` | Advanced optimization with metric guidance | 20-50 examples |
| `BootstrapAGI` | Multi-task optimization | Multiple tasks |

### 3. Defining Metrics

Create a robust evaluation function that takes a `dspy.Example` and returns a score:

```python
def validate_answer(example, pred, trace=None):
    # Check if the answer is correct
    answer_correct = example.answer.lower() == pred.answer.lower()
    # Check if the prediction has the right length
    length_match = len(pred.answer) > 0
    return answer_correct and length_match
```

### 4. Running Optimization

```python
import dspy

# Prepare training data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
]

# Compile the program
optimizer = dspyBootstrapFewShot(metric=validate_answer)
compiled_qa = optimizer.compile(qa, trainset=trainset)
```

### 5. Teacher-Student Pattern

Distill the reasoning of expensive models (Teacher) into optimized prompts for cheap, fast models (Student) to reduce inference costs by up to 50x:

```python
# Teacher uses expensive model
teacher_lm = dspy.LM('openai/gpt-4')
teacher = MathQA().with_lm(teacher_lm)

# Student uses cheap model with distilled prompts
student_lm = dspy.LM('ollama/llama3')
student = MathQA().with_lm(student_lm)

# Compile teacher to generate demonstrations
compiled_teacher = dspy.BootstrapFewShot(metric=validate_answer).compile(
    student,
    teacher=teacher,
    trainset=trainset
)
```

---

## Guardrails

### 1. DSPy Assertions

Use assertions to enforce constraints on module outputs:

```python
# Assert that the answer is not empty
dspy.Assert(len(pred.answer) > 0, "Answer must not be empty")

# Assert that the answer is a valid number
dspy.Assert(float(pred.answer) > 0, "Answer must be positive")

# Assert that the answer contains certain keywords
dspy.Assert("yes" in pred.answer.lower() or "no" in pred.answer.lower(),
            "Answer must be yes or no")
```

### 2. Multi-Stage Validation

```python
class ValidatedModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str):
        result = self.predict(question=question)

        # Validate output
        dspy.Assert(
            len(result.answer) > 0,
            "Answer must not be empty"
        )

        return result
```

---

## Production Patterns

### 1. Caching

Built-in patterns for caching make pipelines less brittle:

```python
# Enable caching for repeated queries
dspy.configure(lm=dspy.LM('openai/gpt-4', cache=True))
```

### 2. Output Validation

Use type hints in signatures for automatic output validation:

```python
class StructuredOutput(dspy.Signature):
    """Return structured data."""
    input_text: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Score between 0 and 1")
    labels: list[str] = dspy.OutputField(desc="List of classification labels")
```

### 3. Monitoring

Track token usage, latency, and error rates:

```python
# Inspect the last prompt/response
dspy.inspect_history(n=1)

# Track usage statistics
lm = dspy.LM('openai/gpt-4')
print(lm.usage)
```

### 4. Error Handling

```python
class RobustModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str):
        try:
            result = self.predict(question=question)
            return result
        except Exception as e:
            # Fallback to simple prediction
            return dspy.Predict("question -> answer")(question=question)
```

---

## Testing

### 1. Mocking the LM

```python
import dspy

# Use spy mode for deterministic testing
dspy.configure(lm=dspy.evaluate.spy)

# Or mock the LM object
mock_lm = dspy.LM('openai/gpt-4')
mock_lm.inspect_history = lambda n: None
```

### 2. Evaluation with DSPy Evaluate

```python
from dspy.evaluate import Evaluate

evaluate = Evaluate(
    devset=devset,
    metric=validate_answer,
    num_threads=4,
    display_progress=True,
    display_table=0
)

score = evaluate(compiled_qa, devset=devset)
print(f"Evaluation score: {score}")
```

### 3. Testing Signatures

```python
def test_signature():
    sig = QA(context="AI is intelligence.", question="What is AI?")
    assert sig.context == "AI is intelligence."
    assert sig.question == "What is AI?"
```

### 4. Testing Modules

```python
def test_module():
    module = MathQA()
    result = module("What is 2+2?")
    assert result.answer == "4"
```

---

## Common Pitfalls

### 1. Over-Engineering Signatures

Start with string signatures for prototyping, then move to class-based signatures for production. Don't over-complicate early on.

### 2. Insufficient Training Data

Optimizers need enough examples to learn from. Aim for at least 10-20 examples for basic optimizers and 20-50 for MIPROv2.

### 3. Poor Metric Design

Your metric is the signal that guides optimization. A weak metric leads to weak optimization. Make sure your metric accurately reflects what you want.

### 4. Ignoring Model Differences

Different models respond differently to prompts. Test your optimized programs across multiple models to ensure portability.

### 5. Not Using Assertions

Assertions catch errors early and improve reliability. Always add assertions for critical constraints.

### 6. Skipping Evaluation

Always evaluate your programs before and after optimization to measure improvement.

---

## References

- [DSPy Official Documentation](https://dspy.ai/)
- [DSPy Tutorials](https://dspy.ai/tutorials/)
- [DSPy: From Newbie to Expert - Nate Ross](https://nateross.dev/blog/dspy-newbie-to-expert)
- [DSPy Guide 2025 - CodeSota](https://www.codesota.com/guides/dspy)
- [Complete DSPy Guide - Atal Upadhyay](https://atalupadhyay.wordpress.com/2025/01/18/complete-dspy-guide-from-basics-to-advanced-optimization/)
- [DSPy 3: Build and Optimize LLM Pipelines - Amir Teymoori](https://amirteymoori.com/dspy-3-build-evaluate-optimize-llm-pipelines/)
- [DSPy Optimization Patterns - KazKozDev](https://github.com/KazKozDev/dspy-optimization-patterns)
- [DSPy 0-to-1 Guide - EvalOps](https://github.com/evalops/dspy-0to1-guide)
