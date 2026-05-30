# Agent

## Overview

The `Agent` is the fundamental building block of the AI Agent Pattern framework. It defines a unified interface for any entity that processes messages and produces responses — whether that entity is a simple LLM wrapper, a tool-calling agent, a RAG pipeline, or an orchestrator coordinating multiple child agents. Every agent in the system conforms to the `BaseAgent` abstract base class, which enforces a consistent `execute`/`aexecute` signature and provides built-in state tracking across agent hierarchies.

## Architecture

The `BaseAgent` class sits at the center of the component hierarchy. It receives an `AgentMessage` input, performs its internal logic (which may involve calling an LLM, invoking tools, retrieving documents, or delegating to child agents), and returns a transformed `AgentMessage`. Agents can be composed into chains and orchestration patterns defined in the core package.

The agent's state system is hierarchical: when an agent contains child agents (as in orchestration patterns), the `composed_state` property builds a tree representation of the entire agent hierarchy's state, enabling real-time progress tracking and debugging.

![diagram](img/agent-architecture.jpg)

## Key Concepts

### Agent Types

The framework recognizes three categories of agents:

- **Remote agent**: An agent that runs on a separate server. The local instance acts as a client, communicating with the remote agent via protocols such as Google's Agent-to-Agent (A2A) communication standard.
- **Local agent**: A self-contained agent running on the same machine, typically wrapping an LLM, a set of tools, or a RAG pipeline.
- **Orchestration agent**: A special agent that coordinates multiple child agents to complete a task. It can perform all normal agent operations while also managing the lifecycle and state of its children.

### AgentCard

Every agent carries an `AgentCard` — a self-describing manifest that provides essential metadata:

- Agent identity (name, description)
- Capabilities and supported skills
- Communication methods
- Security requirements

The `AgentCard` follows the A2A specification, making agents interoperable across different implementations.

### State Management

Each agent maintains two state properties:

- **`state`**: The current state of this individual agent (e.g., `"idle"`, `"running"`, `"completed"`).
- **`composed_state`**: A hierarchical string representing the state of this agent and all its descendants. The format follows a tree structure:

```
parent_name:parent_state/child1_name:child1_state
parent_name:parent_state/((child1:state1)-(child2:state2))  # sequential
parent_name:parent_state/((child1:state1)|(child2:state2))  # parallel
```

The state is automatically synchronized through a callback mechanism. When any agent in the hierarchy changes state, the `state_change_callback` propagates the update up the tree, ensuring the `composed_state` is always current.

### Message Flow

Agents communicate exclusively through `AgentMessage` objects. The standard signature is:

```python
def execute(self, message: AgentMessage, **kwargs) -> AgentMessage
```

This uniform interface allows any agent to be chained or orchestrated with any other agent without type incompatibility concerns.

## Usage

### Basic Example

```python
from aap_core import BaseAgent, AgentMessage
from a2a.types import AgentCard

# Create an agent card describing this agent
card = AgentCard(
    name="my-agent",
    description="A simple example agent",
    capabilities={},
)

# Subclass BaseAgent and implement execute
class MyAgent(BaseAgent):
    card: AgentCard = card

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Process the message — e.g., call an LLM, use tools, etc.
        message.responses.append(("my-agent", f"Processed: {message.query}"))
        message.execution_result = "success"
        return message

# Instantiate and use
agent = MyAgent()
result = agent.execute(AgentMessage(query="Hello, agent!"))
print(result.responses)  # [("my-agent", "Processed: Hello, agent!")]
```

### Advanced Example: Orchestration with Child Agents

```python
from aap_core import BaseAgent, AgentMessage
from a2a.types import AgentCard

# Create child agents
child_card_1 = AgentCard(name="researcher", description="Research agent")
child_card_2 = AgentCard(name="writer", description="Writer agent")

class ResearcherAgent(BaseAgent):
    card = child_card_1
    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append(("researcher", "Research complete."))
        return message

class WriterAgent(BaseAgent):
    card = child_card_2
    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append(("writer", "Writing complete."))
        return message

# Create an orchestration agent that runs children sequentially
class OrchestratorAgent(BaseAgent):
    card = AgentCard(name="orchestrator", description="Orchestrates research and writing")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Run children sequentially
        message = self.researcher.execute(message, **kwargs)
        message = self.writer.execute(message, **kwargs)
        return message

# Use the orchestrator
orchestrator = OrchestratorAgent()
result = orchestrator.execute(AgentMessage(query="Write a report."))

# The composed_state tracks the full hierarchy
print(orchestrator.composed_state)
# Output: orchestrator:idle/researcher:idle/writer:idle
```

### Asynchronous Execution

```python
# Use aexecute for async workflows
async def process_async():
    agent = MyAgent()
    result = await agent.aexecute(AgentMessage(query="Async query"))
    return result
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `card` | `AgentCard` | *required* | A self-describing manifest providing the agent's identity, capabilities, skills, supported communication methods, and security requirements. |
| `state_change_callback` | `Callable[[str], None] \| None` | `None` | An optional callback function invoked whenever the agent's state changes. Receives the updated `composed_state` string as its argument. |

### State Properties

| Property | Type | Description |
|----------|------|-------------|
| `state` | `str` | The current state of this individual agent. Settable via property assignment. |
| `composed_state` | `str` | A hierarchical string representing the state of this agent and all its descendants. Read-only. |

## API Reference

See the full API reference: [`BaseAgent`][aap_core.agent.BaseAgent]

::: aap_core.agent.BaseAgent
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Chain](chain.md) — How to chain multiple agents together into a sequence
- [Orchestration](orchestration.md) — Patterns for coordinating multiple agents
- [Types](types.md) — The `AgentMessage` type and other shared definitions
- [Retriever](retriever.md) — How agents integrate with external retrieval systems
