# Async Testing Best Practices

## Problem: Testing Async Methods in Synchronous Test Suites

When your codebase has async methods (e.g., `ainvoke`, `aexecute`) but your test suite uses synchronous `pytest` tests, you cannot directly call async methods from sync test functions. Calling an async method returns a coroutine object, not the actual result.

## Symptoms

```python
# WRONG - This returns a coroutine, not the result
async def aexecute(self, message):
    return self.execute(message, **kwargs)

def test_aexecute(self):
    agent = SimpleAgent(card=card)
    message = AgentMessage(query="test")
    result = agent.aexecute(message)  # Returns coroutine!
    assert result.execution_result == "success"  # AttributeError: 'coroutine' object has no attribute...
```

Error message: `AttributeError: 'coroutine' object has no attribute 'execution_result'`

## Solutions

### Option 1: Test the Sync Implementation Directly (Recommended for Simple Wrappers)

If the async method is just a thin wrapper around a sync method, test the sync method directly:

```python
def test_aexecute(self):
    """Test aexecute calls execute (sync version for testing)."""
    agent = SimpleAgent(card=_make_agent_card("simple_agent"))
    message = AgentMessage(query="test")
    # aexecute is async, so we call execute directly for sync testing
    result = agent.execute(message)
    assert result.execution_result == "success"
```

**When to use:**
- The async method is a simple wrapper (e.g., `async def foo(): return self.bar()`)
- You want to keep tests fast and simple
- The async method doesn't add any unique logic

### Option 2: Use pytest-asyncio (For Complex Async Logic)

If the async method has unique async logic (e.g., awaits, concurrent operations), use `pytest-asyncio`:

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_method():
    agent = SimpleAgent(card=_make_agent_card("simple_agent"))
    message = AgentMessage(query="test")
    result = await agent.aexecute(message)
    assert result.execution_result == "success"
```

**Requirements:**
- Install: `uv add --dev pytest-asyncio`
- Configure in `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  asyncio_mode = "auto"  # Auto-detect async test functions
  ```

**When to use:**
- The async method has unique async logic (concurrency, awaits, event loops)
- You need to test async fixtures or async context managers
- Testing async frameworks (FastAPI, aiohttp, async SQLAlchemy)

### Option 3: Use asyncio.run() (Quick Fix for One-Off Tests)

```python
def test_async_method():
    agent = SimpleAgent(card=_make_agent_card("simple_agent"))
    message = AgentMessage(query="test")
    # Run the coroutine in a new event loop
    result = asyncio.run(agent.aexecute(message))
    assert result.execution_result == "success"
```

**When to use:**
- Quick one-off tests
- Not recommended for production test suites (creates new event loop each time)

## Decision Flowchart

```
Does the async method have unique async logic?
├── No (just a wrapper) → Test the sync method directly (Option 1)
└── Yes → Do you need to test async-specific behavior?
    ├── No → Test the sync implementation (Option 1)
    └── Yes → Use pytest-asyncio (Option 2)
```

## Common Pitfalls

### 1. Forgetting to Await

```python
# WRONG
result = agent.ainvoke(message)  # Returns coroutine

# CORRECT (with pytest-asyncio)
@pytest.mark.asyncio
async def test_invoke():
    result = await agent.ainvoke(message)
```

### 2. Mixing Sync and Async in Same Test

```python
# WRONG - Mixing sync and async
def test_mixed():
    sync_result = agent.execute(message)  # OK
    async_result = agent.ainvoke(message)  # Returns coroutine!
```

### 3. Event Loop Conflicts

```python
# WRONG - Creating objects outside async context
client = AsyncClient()  # May fail if event loop doesn't exist

# CORRECT - Create objects inside async functions
@pytest.mark.asyncio
async def test_client():
    async with AsyncClient() as client:  # OK
        ...
```

## Project-Specific Guidance

For the `aap_core` project:
- Most async methods (`ainvoke`, `aexecute`) are thin wrappers around sync methods
- **Recommendation:** Test the sync implementation directly (Option 1)
- Only use `pytest-asyncio` if testing actual async frameworks (FastAPI, async DB clients)

## References

- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [FastAPI Async Tests](https://fastapi.tiangolo.com/advanced/async-tests/)
- [Python asyncio Testing](https://docs.python.org/3/library/asyncio-testing.html)
