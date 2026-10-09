# Orchestration

## Overview

The `Orchestration` component provides a collection of agent patterns that coordinate multiple child agents to solve complex tasks. While a single [`BaseAgent`](agent.md) processes one message at a time, orchestration agents compose multiple agents into structured workflows — enabling sequential pipelines, parallel execution, self-reflection, debate, voting, and multi-stage planning. These patterns are the foundation for building sophisticated multi-agent systems on top of the AI Agent Pattern framework.

## Architecture

Orchestration agents sit at the top of the component hierarchy. They receive an `AgentMessage`, delegate work to child agents (which may be simple local agents, LLM chains, or other orchestration agents), and return a transformed `AgentMessage`. The orchestration module lives in `aap_core.orchestration` and all classes inherit from `BaseAgent`, ensuring they can be nested arbitrarily.

The six orchestration patterns are organized by their execution strategy:

- **Sequential**: `SequentialAgent` — linear pipeline of agents
- **Parallel**: `ParallelAgent` — concurrent execution via `ThreadPoolExecutor`
- **Iterative**: `LoopAgent` — repeat an agent until a stop condition
- **Reflective**: `ReflectionAgent` — execute then self-reflect
- **Collaborative**: `DebateAgent`, `VotingAgent` — multi-agent discussion and consensus
- **Planned**: `CoordinatorAgent` — plan → execute → summarize pipeline

![diagram](img/orchestration-architecture.jpg)

## Key Concepts

### Composed State

Every orchestration agent tracks a `composed_state` — a hierarchical string representing the state of the entire agent tree. The `_set_composed_state` method is called automatically to build this tree, using either `"sequential"` or `"parallel"` grouping for child agents. This enables real-time progress tracking across deeply nested agent hierarchies.

### Message Routing

Orchestration agents route `AgentMessage` objects between child agents. The message's `responses` field accumulates results across the workflow, and the `context` field carries intermediate data between stages. Each child agent's response is appended to `message.responses` as a tuple of `(agent_name, response_text)`, preserving provenance.

### Execution Strategies

| Strategy | Agent | Description |
|----------|-------|-------------|
| Sequential | `SequentialAgent` | Agents execute one after another; failure in any agent halts the pipeline. |
| Parallel | `ParallelAgent` | Agents execute concurrently via thread pool; a single message is broadcast to all, or one message per agent is provided. |
| Iterative | `LoopAgent` | A single agent repeats until `is_stop` returns `True` (callable) or the generator is exhausted. |
| Reflective | `ReflectionAgent` | A task chain executes first, then a reflection chain evaluates the result. |
| Debate | `DebateAgent` | Multiple agents take turns with `round_robin`, `random`, `simultaneous`, or a custom pick strategy. |
| Voting | `VotingAgent` | Multiple agents produce candidates; a scoring method selects the best response. |
| Planned | `CoordinatorAgent` | A planner generates steps, workers execute them sequentially, and a summary chain finalizes. |

### Agent-Specific Patterns

#### ReflectionAgent

The `ReflectionAgent` implements a two-stage pipeline: first, a `chain_task` processes the message; then, a `chain_reflection` evaluates the result. If reflection succeeds and `keep_original_response` is `True`, the original response is preserved in the output. The `task_response_key` parameter controls the context key prefix (must start with `context_`).

#### LoopAgent

The `LoopAgent` wraps a single child agent and repeats execution until a stop condition is met. The `is_stop` parameter accepts either:
- A **callable** `Callable[[AgentMessage], bool]` that returns `True` to stop.
- A **generator** that yields `False` values until exhaustion or an explicit `True`.

The `keep_result` parameter controls how many intermediate responses to retain: an integer `n` keeps the last `n` responses, or a callable filters the response list.

#### SequentialAgent

The simplest orchestration pattern: a list of agents executes in order. If any agent returns `execution_result != "success"`, the pipeline halts immediately. The order of `agents` determines the execution order.

#### ParallelAgent

Executes a list of agents concurrently using Python's `concurrent.futures.ThreadPoolExecutor`. Accepts either:
- A **single** `AgentMessage` — broadcast to all agents (each gets a deep copy).
- A **sequence** of `AgentMessage` — one per agent, length must match.

Results are collected via `as_completed()` and combined into a single response message. Note: results are not guaranteed to be in agent order due to thread scheduling.

#### CoordinatorAgent

A three-stage planning agent:
1. **Planning**: `planner_agent` receives the query and produces a plan.
2. **Execution**: `parse_plan` converts the plan into `(message, agent, dependencies)` tuples; each step executes sequentially.
3. **Summary**: Optional `summary_chain` or `summary_prompt` finalizes the output from all intermediate results.

The `parse_plan` callable receives the planner's output and the list of available workers, returning a sequence of step tuples. Each step can declare dependencies on previous results via the `dependencies` list. The `summary_steps_key` parameter controls the context key for intermediate results (must start with `context_`).

#### DebateAgent

Multiple agents participate in a multi-turn discussion. The `pick_strategy` determines turn order:
- **`"round_robin"`** — agents take turns in order.
- **`"random"`** — a random agent is chosen each turn (supports `random_seed`).
- **`"simultaneous"`** — all agents process the same message at once each turn.
- **Callable** — a custom function `Callable[[Sequence[BaseAgent]], BaseAgent]` selects the next agent.

The debate stops when `max_turns` is reached or `should_stop(message)` returns `True`. Note: this agent handles topic expansion only; for complete flows with summarization, combine with other orchestration agents.

#### VotingAgent

Multiple agents produce candidate responses, and a voting method selects the best one. Two voting strategies are supported:

- **`"majority_vote"`** — uses text similarity metrics:
  - `"bleu"` or `"agent_forest"` — BLEU score via `sacrebleu`.
  - `"rougeL"` — ROUGE-L F-measure via `rouge_score`.
  - `"rougeN"` — ROUGE-N F-measure (N is an integer, e.g., `"rouge1"`, `"rouge2"`).

- **`"llm_score"`** — each agent scores responses from other agents using an LLM. Requires a `voting_prompt` and a `scorer` callable that converts LLM text output to a float score.

The response from the highest-scoring agent is appended to `message.responses`.

## Usage

### Basic Example: Sequential Pipeline

```python
from aap_core import AgentMessage
from aap_core.orchestration import SequentialAgent

# Create individual agents
agent1 = MyTaskAgent()   # e.g., a research agent
agent2 = MyTaskAgent()   # e.g., a writing agent
agent3 = MyTaskAgent()   # e.g., an editing agent

# Chain them sequentially
pipeline = SequentialAgent(agents=[agent1, agent2, agent3])

# Execute
message = AgentMessage(query="Write a report on AI safety")
result = pipeline.execute(message)
print(result.responses)  # Responses from all three agents in order
```

### Basic Example: Parallel Execution

```python
from aap_core import AgentMessage
from aap_core.orchestration import ParallelAgent

# Create agents that work on different aspects
agent1 = MyTaskAgent()
agent2 = MyTaskAgent()
agent3 = MyTaskAgent()

# Execute in parallel
parallel = ParallelAgent(agents=[agent1, agent2, agent3])

# Broadcast a single message to all agents
message = AgentMessage(query="Summarize the key points")
result = parallel.execute(message)
print(result.responses)  # One response from each agent
```

### Basic Example: Self-Reflection

```python
from aap_core import AgentMessage
from aap_core.orchestration import ReflectionAgent

# Task chain generates the answer
task_chain = MyLLMChain(prompt="Answer the following: {query}")
# Reflection chain evaluates the answer
reflection_chain = MyLLMChain(prompt="Evaluate this answer: {context_response}. Provide feedback.")

reflector = ReflectionAgent(
    chain_task=task_chain,
    chain_reflection=reflection_chain,
)

message = AgentMessage(query="Explain quantum computing")
result = reflector.execute(message)
print(result.responses)  # Original response + reflection feedback
```

### Basic Example: Debate

```python
from aap_core import AgentMessage
from aap_core.orchestration import DebateAgent

def should_stop(message: AgentMessage) -> bool:
    # Stop if the last response contains "CONSENSUS"
    return "CONSENSUS" in message.responses[-1][1]

debaters = [DebateAgentA(), DebateAgentB(), DebateAgentC()]

debate = DebateAgent(
    agents=debaters,
    pick_strategy="round_robin",
    max_turns=10,
    should_stop=should_stop,
)

message = AgentMessage(query="What is the best approach to AGI alignment?")
result = debate.execute(message)
print(result.responses)  # All turns from all debaters
```

### Basic Example: Voting

```python
from aap_core import AgentMessage
from aap_core.orchestration import VotingAgent

# Multiple agents generate candidate answers
agents = [AnswerAgentA(), AnswerAgentB(), AnswerAgentC()]

# Use BLEU-based majority voting
voter = VotingAgent(
    agents=agents,
    voting_method="majority_vote",
    scorer="bleu",
)

message = AgentMessage(query="What is the capital of France?")
result = voter.execute(message)
print(result.responses)  # The highest-scoring response
```

### Advanced Example: Coordinator with Planning

```python
from aap_core import AgentMessage
from aap_core.orchestration import CoordinatorAgent

def parse_plan(plan_msg, workers):
    """Parse planner output into (message, agent, dependencies) tuples."""
    steps = []
    for step_text in plan_msg.responses:  # Assume planner outputs list of steps
        worker_idx = int(step_text[0])  # First char indicates worker index
        step_msg = AgentMessage(query=step_text[1:])
        steps.append((step_msg, workers[worker_idx], []))
    return steps

coordinator = CoordinatorAgent(
    planner_agent=PlannerAgent(),
    workers=[ResearchAgent(), WriterAgent(), EditorAgent()],
    parse_plan=parse_plan,
    summary_chain=SummaryChain(),
)

message = AgentMessage(query="Compare three approaches to X")
result = coordinator.execute(message)
print(result.responses)  # Final summarized answer
```

### Advanced Example: Nested Orchestration

```python
from aap_core import AgentMessage
from aap_core.orchestration import (
    SequentialAgent,
    ParallelAgent,
    ReflectionAgent,
)

# Inner parallel team
researchers = ParallelAgent(agents=[
    EconomicsResearcher(),
    PoliticsResearcher(),
    SocialResearcher(),
])

# Outer sequential pipeline with reflection
pipeline = SequentialAgent(agents=[
    researchers,  # Parallel research phase
    ReflectionAgent(  # Self-reflection on research
        chain_task=SynthesisChain(),
        chain_reflection=ReflectionChain(),
    ),
])

message = AgentMessage(query="Analyze the impact of AI on society")
result = pipeline.execute(message)
```

## Configuration

### ReflectionAgent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain_task` | `BaseLLMChain` | *(required)* | LLM chain that performs the main task. |
| `chain_reflection` | `BaseLLMChain` | *(required)* | LLM chain that performs reflection on the task result. |
| `task_response_key` | `str` | `"context_response"` | Context key for the task response. Must start with `context_`. |

### LoopAgent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `BaseAgent` | *(required)* | The agent to loop. |
| `is_stop` | `Callable[[AgentMessage], bool] \| Generator[bool]` | *(required)* | Stop condition — a callable returning `True` to stop, or a generator yielding `False` values. |

### SequentialAgent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `Sequence[BaseAgent]` | *(required)* | List of agents to execute in sequence. Minimum length: 1. |

### ParallelAgent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `Sequence[BaseAgent]` | *(required)* | List of agents to execute in parallel. Minimum length: 1. |

### CoordinatorAgent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `planner_agent` | `BaseAgent` | *(required)* | Agent that plans the steps to be executed. |
| `parse_plan` | `Callable[[AgentMessage, Sequence[BaseAgent]], Sequence[Tuple[AgentMessage, BaseAgent, List]]]` | *(required)* | Function that parses the planner's output into step tuples. |
| `workers` | `Sequence[BaseAgent]` | *(required)* | List of agents that execute the planned steps. Minimum length: 1. |
| `summary_chain` | `BaseLLMChain \| None` | `None` | Optional chain for final summarization. |
| `summary_prompt` | `str \| None` | `None` | Optional prompt used by the planner as a fallback summary agent. |
| `summary_steps_key` | `str` | `"context_results"` | Context key for intermediate results. Must start with `context_`. |

### DebateAgent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `Sequence[BaseAgent]` | *(required)* | List of agents participating in the debate. Minimum length: 1. |
| `pick_strategy` | `Literal["round_robin", "random", "simultaneous"] \| Callable` | *(required)* | Strategy for selecting the next speaking agent. |
| `random_seed` | `int \| None` | `None` | Random seed for the `"random"` strategy. |
| `max_turns` | `int` | `5` | Maximum number of debate turns. Must be ≥ 1. |
| `should_stop` | `Callable[[AgentMessage], bool] \| None` | `None` | Optional early-stop condition. |

### VotingAgent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `Sequence[BaseAgent]` | *(required)* | List of agents producing candidate responses. Minimum length: 1. |
| `voting_method` | `Literal["majority_vote", "llm_score"]` | *(required)* | Voting strategy: metric-based or LLM-based scoring. |
| `scorer` | `str \| Callable[[str], float]` | *(required)* | For `majority_vote`: `"bleu"`, `"agent_forest"`, `"rougeL"`, or `"rougeN"`. For `llm_score`: a callable converting text to a float score. |
| `voting_prompt` | `str \| None` | `None` | Required when `voting_method` is `"llm_score"`. The prompt for scoring. |

## API Reference

See the full API reference for each orchestration class:

::: aap_core.orchestration.ReflectionAgent
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.orchestration.LoopAgent
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.orchestration.SequentialAgent
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.orchestration.ParallelAgent
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.orchestration.CoordinatorAgent
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.orchestration.DebateAgent
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.orchestration.VotingAgent
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Agent](agent.md) — The `BaseAgent` foundation for all orchestration patterns
- [Chain](chain.md) — LLM chains used as building blocks within orchestration agents
- [Types](types.md) — `AgentMessage` and `AgentResponse` types used throughout
