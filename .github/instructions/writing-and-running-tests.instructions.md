---
name: writing-and-running-tests
description: "Use when: writing, reviewing, or running unit tests in this project — creating test files, running pytest, analyzing test failures, or improving test coverage. Covers all packages under src/ (core, dspy, langchain, llamaindex, transformers)."
applyTo: ["tests/**/*.py", "test_*.py", "**/test_*.py"]
---

# Writing and Running Tests

## Environment & Execution

- **Always use `uv`** for running tests: `uv run pytest` from the relevant package directory.
- Use `uv run pytest -v` for verbose output during development.
- Use `uv run pytest --tb=short` for concise output.
- Use `uv run pytest -x` to stop on first failure.
- Use `uv run pytest -k "pattern"` to run tests matching a substring.
- Use `uv run pytest --co` to collect tests without running them.

## Test File Conventions

- All test files must be in the `tests/` directory within each package.
- Test files must be named `test_*.py`.
- Test functions must start with `test_`.
- Test classes must start with `Test` (without `__init__` method).
- Use `importlib` import mode — do not modify `sys.path` manually.

## Test Writing Best Practices

### 1. Plan Before Writing

- **List all test cases first** before writing any code. Cover:
  - **Normal cases**: expected inputs producing expected outputs
  - **Edge cases**: boundary values, empty inputs, single-element inputs
  - **Extreme values**: very large/small numbers, maximum sequence lengths, zero values
  - **Error cases**: invalid inputs, missing required fields, type mismatches
  - **Expected behaviors**: what exceptions are raised, what warnings are emitted

- **Write / fix one test at a time**. Analyze each test case carefully before writing.
- Follow each test through to validation passing before moving to the next.

### 2. Test Structure (Arrange-Act-Assert)

Every test follows this pattern:
```python
def test_feature_scenario():
    # Arrange: set up inputs, fixtures, mocks
    # Act: call the function/method under test
    # Assert: verify the output matches expected result
```

### 3. Fixtures & Reusability

- Use `@pytest.fixture` for shared test setup (policies, environments, models).
- Use `tmp_path` fixture for file I/O tests (save/load models).
- Use `@pytest.mark.parametrize` for testing multiple input combinations.
- Keep fixtures scoped appropriately: `function` (default), `class`, `module`, `session`.

### 4. Mocking & Isolation

- Mock external dependencies (LLM calls, network requests, file system) with `unittest.mock`.
- Use `patch` to replace slow or non-deterministic operations.
- Keep unit tests fast and deterministic — no real API calls.
- Use `torch.no_grad()` and `model.eval()` for inference tests.

### 5. PyTorch-Specific Testing

- Always set `model.eval()` and use `torch.no_grad()` for inference tests.
- Use `torch.testing.assert_close()` for tensor comparisons with tolerances.
- Test both forward pass and `evaluate_actions` for policy classes.
- Test `save()` / `load()` round-trip: save model, create new instance, load, compare outputs.
- Use small model configurations for tests (fewer layers, smaller embedding dims).

### 6. Assertion Guidelines

- Use `assert` statements (pytest's strength — no `self.assert*` needed).
- Provide descriptive messages: `assert result == expected, f"Expected {expected}, got {result}"`.
- Use `pytest.raises()` for expected exceptions:
  ```python
  with pytest.raises(ValueError, match="expected pattern"):
      function_under_test(invalid_input)
  ```
- Use `pytest.warns()` for expected warnings.

### 7. Parameterized Tests

Use `@pytest.mark.parametrize` to test multiple scenarios concisely:
```python
@pytest.mark.parametrize(
    "input_val,expected",
    [
        (0, "zero"),
        (1, "one"),
        (-1, "negative"),
    ],
)
def test_number_classification(input_val, expected):
    assert classify_number(input_val) == expected
```

### 8. What NOT to Do

- **DO NOT modify source code** — tests only read and validate, never fix.
- **DO NOT use `unittest.TestCase`** — use plain pytest functions/classes.
- **DO NOT hardcode random seeds without setting them** — use `torch.manual_seed()` for reproducibility.
- **DO NOT skip tests without explanation** — use `pytest.skip(reason=...)` with a reason.
- **DO NOT test implementation details** — test public interfaces and observable behavior.

## Testing Workflow

1. **Identify the module/function** to test.
2. **List all test cases** (normal, edge, extreme, error).
3. **Write one test** — analyze the case, write the test, run it.
4. **Validate** — if it fails, analyze why (is it a test bug or a real issue?).
5. **Move to next test** only after the current one passes.
6. **Run full test suite** after completing a module: `uv run pytest -v`.

## Common Test Patterns in This Project

### Policy Tests
```python
class SimplePolicy(BasePolicy):
    """Minimal policy for testing — implement forward and evaluate_actions."""
    ...

def test_policy_forward():
    policy = SimplePolicy(action_space, observation_space)
    obs = torch.randn(2, 3, 10)
    logits = policy(obs)
    assert logits.shape == (2, 3, action_dim)
```

### Policy Trainer Tests
```python
def test_trainer_update():
    policy = SimplePolicy(...)
    env = PromptOptimizationEnv(...)
    trainer = ReinforcePP(policy, env, max_episodes=10, optimizer, lr_scheduler)
    batch = {"obs": ..., "actions": ..., "rewards": ..., "masks": ..., "old_log_probs": ...}
    losses = trainer.update(batch["obs"], batch["actions"], batch["rewards"], batch["masks"], batch["old_log_probs"])
    action_loss, entropy, kl_loss = losses
    assert isinstance(action_loss, torch.Tensor)
```

### Pydantic Model Tests
```python
def test_pydantic_validation():
    with pytest.raises(ValidationError):
        ModelField(field_with_validator="invalid_value")
```

### Chain / Agent Tests
```python
def test_chain_invoke():
    chain = TypicalLLMChain(...)
    message = AgentMessage(query="test query")
    result = chain.invoke(message)
    assert result.execution_result == "success"
```

## Coverage Goals

- Aim for **high coverage on core logic**: policy, policy_trainer, prompt_augmenter, types, utils.
- **Orchestration agents** (ReflectionAgent, LoopAgent, DebateAgent): test state transitions and message flow.
- **Retriever classes**: test data retrieval, formatting, and post-processing.
- **Guardrails**: test pass-through and rejection behaviors.
- **Edge cases are critical**: zero rewards, empty masks, single-step episodes, extreme learning rates.
