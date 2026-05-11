# LangChain Best Practices

A comprehensive guide of best practices for building production-ready LangChain applications, synthesized from industry leaders, technical experts, and real-world implementations.

## Table of Contents

- [Architecture](#architecture)
- [RAG Systems](#rag-systems)
- [Agent Design](#agent-design)
- [Tool Calling](#tool-calling)
- [Prompt Engineering](#prompt-engineering)
- [Error Handling](#error-handling)
- [Observability](#observability)
- [Performance](#performance)
- [Testing](#testing)

---

## Architecture

### 1. Use LangChain Expression Language (LCEL)

LCEL is the modern approach to building LLM applications, offering composability, testability, and native streaming support.

```python
# ✅ Good: LCEL with pipe syntax
chain = prompt | model | output_parser

# ❌ Bad: Legacy chain patterns
```

- LCEL chains support streaming, batching, and fallback mechanisms out of the box
- Use `RunnableSequence`, `RunnableParallel`, and `RunnableLambda` for complex compositions
- Enables serving nearly 20 production instances supporting 3,000+ internal users

### 2. Structured Output with Pydantic

Structured output using Pydantic models reduces post-processing bugs and ensures type safety.

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class WeatherResponse(BaseModel):
    location: str = Field(description="City name")
    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions")

parser = PydanticOutputParser(pydantic_object=WeatherResponse)
```

### 3. Multi-Model Orchestration

Combine multiple models where each handles what they do best:

- **GPT-4o** — Top choice for orchestration (cost-effective, stable, follows instructions)
- **Claude** — Complex reasoning, safety-critical decisions
- **GPT-4.1** — Tool calling (underwent extensive training on tool utilization)
- **Smaller models** — Classification and simple transformations

```python
# Model routing based on task complexity
def route_request(task: str, complexity: float):
    if complexity < 0.3:
        return fast_model      # GPT-4o-mini
    elif complexity < 0.7:
        return orchestrator    # GPT-4o
    else:
        return reasoning_model # Claude
```

---

## RAG Systems

### 1. Document Processing and Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 500-1000 characters recommended
    chunk_overlap=100,    # 100-200 characters overlap
    length_function=len,
)
```

### 2. Advanced Retrieval Strategies

- **MMR (Maximum Marginal Relevance)** — Improves diversity and reduces redundancy
- **Hybrid Search** — Combine semantic search with keyword-based retrieval
- **Re-ranking** — Use LLM re-rankers for table/field selection optimization

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 3. Citation and Grounding

- Enforce citations and ground responses in provided context to prevent hallucinations
- Prompts should explicitly instruct the model to answer only from given context
- Cite sources using numbered references

---

## Agent Design

### 1. Multi-Agent Architecture Pattern

Use specialized agents instead of monolithic approaches:

1. **Planner Agent** — Strategic brain that decomposes user intent into subtasks
2. **Executor Agents** — Specialized workers for specific subtasks
3. **Communicator Agent** — Ensures smooth handoff between agents
4. **Validator Agent** — Quality gates that catch hallucinations and errors

### 2. LangGraph for Complex Workflows

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_intent: str | None
    tool_results: dict
    error_count: int
    resolved: bool

workflow = StateGraph(AgentState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("validate_response", validate_response)
workflow.add_conditional_edges(
    "execute_tools",
    should_continue,
    {"validate": "validate_response", "retry": "execute_tools", "error": END}
)
workflow.set_entry_point("classify_intent")
app = workflow.compile()
```

### 3. State Management and Memory

- **ConversationBufferWindowMemory** — Keep last k messages for most applications
- **Vector-based memory** — Semantic conversation search for historical context
- **Persistent memory** — Use Redis or database for production deployments

---

## Tool Calling

### 1. Production Tool Pattern

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class DatabaseQueryInput(BaseModel):
    query: str = Field(description="SQL query to execute")
    timeout_seconds: int = Field(default=30, description="Query timeout")
    dry_run: bool = Field(default=True, description="Validate without executing")

@tool(args_schema=DatabaseQueryInput)
async def query_database(query: str, timeout_seconds: int = 30, dry_run: bool = True) -> dict:
    """Execute a database query with production safeguards."""
    # Validation, timeout, error handling
    return {"status": "success", "data": [...], "error": None}
```

### 2. Tool Design Principles

- **Simple, narrowly scoped tools** are easier for models to use
- **Well-chosen names and descriptions** significantly improve model performance
- **Use the `@tool` decorator** — automatically infers name, description, and arguments
- **Return structured data** — Always include status, data, and error fields
- **Implement timeouts and retries** — Production systems must be resilient

### 3. Concurrent Tool Execution

LangGraph's `ToolNode` executes multiple tools concurrently by default:

```python
from langgraph.prebuilt import ToolNode

tools = [query_database, call_external_api, process_document]
tool_node = ToolNode(tools)  # Handles concurrency automatically
```

---

## Prompt Engineering

### 1. Three-Tier Prompt Strategy

**Tier 1: System Prompts (The Foundation)**
- Clear constraints, explicit output format, tool visibility
- Temperature awareness guidance

**Tier 2: Few-Shot Examples (The Teacher)**
- Dramatically improves tool calling accuracy
- Include diverse examples covering edge cases

**Tier 3: Dynamic Context Injection (The Optimizer)**
- Use prompt caching for large static context
- Reduces latency and cost significantly

### 2. Prompt Best Practices

1. Use `temperature=0` for deterministic tasks (data extraction, classification, tool calling)
2. Name tools clearly — API-parsed tool descriptions outperform manual injection
3. Iterate systematically — Start simple, measure performance, add complexity only when needed
4. Leverage structured outputs — Use JSON schema validation
5. Include agentic reminders in all agent prompts

---

## Error Handling

### 1. Production Reliability Targets

| Metric | Target |
|--------|--------|
| Tool call error rate | < 3% |
| P95 latency | < 5 seconds |
| Loop containment rate | > 99% |
| Graceful degradation | System transitions to backups, not crash |

### 2. Retry with Exponential Backoff

```python
class ProductionErrorHandler:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def with_retry(self, func, *args, **kwargs):
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                delay = min(self.base_delay * (2 ** attempt), 60.0)
                await asyncio.sleep(delay)
        raise last_exception
```

### 3. Fallback Strategies

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary_llm = ChatOpenAI(model="gpt-4o")
fallback_llm = ChatAnthropic(model="claude-3-sonnet")
llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])
```

---

## Observability

### 1. LangSmith Integration

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-agent-prod"
```

- Track token usage, costs, latencies, and error rates
- Identify bottlenecks and problematic interactions
- Enable comprehensive debugging

### 2. OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("agent_execution")
async def instrumented_agent_call(query: str):
    result = await agent.ainvoke(query)
    return result
```

### 3. Key Metrics to Track

- Total calls, total tokens, average latency
- Error count, error rate, tool call error rate
- Cache hit rate, tokens per request
- Model-specific costs and budget thresholds

---

## Performance

### 1. Semantic Caching

```python
from langchain.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95
)
```

### 2. Cost Optimization Strategies

| Strategy | Implementation | Savings |
|----------|---------------|---------|
| Caching | Redis/SQLite cache | 40-60% |
| Model routing | GPT-3.5 for simple, GPT-4 for complex | 30-50% |
| Token limits | `max_tokens` parameter | Variable |
| Batch processing | Async concurrent calls | Time savings |

### 3. Prompt Caching

Use Anthropic's prompt caching for large static context:

```python
response = client.messages.create(
    model="claude-4-opus-20250514",
    system=[
        {"type": "text", "text": cached_context, "cache_control": {"type": "ephemeral"}}
    ],
    messages=[{"role": "user", "content": user_query}]
)
```

---

## Testing

### 1. Mocking LLMs

```python
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

def test_agent_response():
    mock_model = MagicMock()
    mock_response = AIMessage(
        content="The answer is 42.",
        tool_calls=[],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    mock_model.invoke.return_value = mock_response
    # ... test logic
```

- **NEVER** make real API calls in tests
- Use `unittest.mock.MagicMock` to mock `BaseChatModel.invoke`
- Configure mocks to return `AIMessage` objects with desired content, tool_calls, and usage_metadata

### 2. Testing Tool Interactions

```python
def test_agent_uses_correct_tool():
    with patch.object(search_tool, 'func') as mock_search:
        mock_search.return_value = "Found 5 items"
        result = agent_executor.invoke({"input": "Find all laptops"})
        mock_search.assert_called_once()
```

### 3. Production Checklist

- [ ] Robust error handling — Agents will fail; plan for it
- [ ] Observability — You can't fix what you can't see
- [ ] Cost controls — LLM calls add up fast
- [ ] Testing — Especially for tool interactions
- [ ] Fallbacks — Have backup plans

---

## References

- [LangChain Best Practices - Swarnendu De](https://www.swarnendu.de/blog/langchain-best-practices/)
- [Building Production-Ready AI Agents with LangChain - Kanaeru AI](https://www.kanaeru.ai/blog/2025-10-06-production-ai-agents-langchain)
- [Building Production-Ready AI Agents with LangChain - DevStarsJ](https://devstarsj.github.io/ai/2026/02/02/LangChain-AI-Agents-Production-Guide/)
- [LangChain Official Documentation](https://python.langchain.com/docs/introduction/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
