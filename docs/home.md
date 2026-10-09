# AI Agent Pattern (AAP)

**A modular framework for orchestrating multi-agent systems across popular LLM libraries.**

AI Agent Pattern provides a unified, framework-agnostic layer for building and coordinating multi-agent workflows. While LLM interaction is delegated to dedicated libraries like LangChain, LlamaIndex, DSPy, and Hugging Face Transformers, AAP focuses on the orchestration logic — how agents communicate, chain reasoning steps, and collaborate to solve complex tasks.

## What It Solves

Building multi-agent systems involves recurring patterns: chaining agents for multi-turn reasoning, coordinating parallel agent teams, implementing self-reflection loops, and managing retrieval-augmented workflows. AAP extracts these patterns into reusable abstractions so you can focus on agent design rather than boilerplate orchestration code.

### Key Features

- **Framework-agnostic core** — Shared abstractions (`Agent`, `Chain`, `Orchestration`) that work across all supported LLM libraries
- **4 integration packages** — Native adapters for LangChain, LlamaIndex, DSPy, and Hugging Face Transformers
- **Pre-built orchestration patterns** — Coordinator, debate, voting, parallel, sequential, self-reflection, cross-reflection, and iterative refinement
- **Prompt augmentation & retrieval** — Built-in utilities for prompt enhancement and external data retrieval
- **Jupyter-ready examples** — Notebook-based examples for every integration and pattern

## Architecture

```
┌─────────────────────────────────────────────┐
│           Your Agent Application             │
└─────────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────┬─────────────┐
    ▼             ▼             ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Core  │  │ LangChain│  │LlamaIndex│  │  DSPy    │  │Transformers│
└────┬───┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │           │              │             │              │
     └───────────┼──────────────┼─────────────┼──────────────┘
                 ▼              ▼             ▼              ▼
        ┌────────────────────────────────────────────────────────┐
        │  LLM Providers                                         │
        │  (OpenAI, Ollama, HF, Anthropic, etc.)                 │
        └────────────────────────────────────────────────────────┘
```

The core package defines shared types (`AgentMessage`, `TokenUsage`, `BaseLLMChain`) and base interfaces. Every integration package implements these interfaces using its respective LLM library.

## Getting Started

### 1. Install a Package

Choose the integration that matches your preferred LLM library:

#### LangChain integration
```bash
pip install aap_langchain
```

#### LlamaIndex integration
```bash
pip install aap_llamaindex
```

#### Hugging Face Transformers integration
```bash
pip install aap_transformers
```

#### DSPy integration
```bash
pip install aap_dspy
```

### 2. Define Your Agent

Each integration provides an `Agent` class backed by its respective library. Define the agent's role, LLM chain, and any retrievers or prompt augmenters.

### 3. Chain and Orchestrate

Connect agents into chains for multi-turn reasoning, or use orchestration patterns (coordinator, debate, voting, etc.) for multi-agent collaboration.

### 4. Run

Execute the flow and get structured results with token usage tracking.

## Example Notebooks

Ready-to-run Jupyter notebooks demonstrating every pattern across all integrations:

| Pattern | LangChain | LlamaIndex | DSPy | Transformers |
|---------|-----------|------------|------|--------------|
| Sequential | [→](example/langchain/sequential.ipynb) | [→](example/llamaindex/sequential.ipynb) | [→](example/dspy/sequential.ipynb) | [→](example/transformers/sequential.ipynb) |
| Coordinator | [→](example/langchain/coordinator.ipynb) | [→](example/llamaindex/coordinator.ipynb) | [→](example/dspy/coordinator.ipynb) | [→](example/transformers/coordinator.ipynb) |
| Self-Reflection | [→](example/langchain/self_reflection.ipynb) | [→](example/llamaindex/self_reflection.ipynb) | [→](example/dspy/self_reflection.ipynb) | [→](example/transformers/self_reflection.ipynb) |
| Parallel | [→](example/langchain/parallel.ipynb) | [→](example/llamaindex/parallel.ipynb) | [→](example/dspy/parallel.ipynb) | [→](example/transformers/parallel.ipynb) |
| Debate | [→](example/langchain/debate.ipynb) | [→](example/llamaindex/debate.ipynb) | [→](example/dspy/debate.ipynb) | [→](example/transformers/debate.ipynb) |
| Voting | [→](example/langchain/voting.ipynb) | [→](example/llamaindex/voting.ipynb) | [→](example/dspy/voting.ipynb) | [→](example/transformers/voting.ipynb) |
| Retriever | [→](example/langchain/retriever.ipynb) | [→](example/llamaindex/retriever.ipynb) | [→](example/dspy/retriever.ipynb) | — |
| Iterative Refinement | [→](example/langchain/iterative_refinement.ipynb) | [→](example/llamaindex/iterative_refinement.ipynb) | [→](example/dspy/iterative_refinement.ipynb) | — |
| Cross-Reflection | [→](example/langchain/cross_reflection.ipynb) | [→](example/llamaindex/cross_reflection.ipynb) | [→](example/dspy/cross_reflection.ipynb) | [→](example/transformers/cross_reflection.ipynb) |
| Simple Loop | [→](example/langchain/simple_loop.ipynb) | [→](example/llamaindex/simple_loop.ipynb) | [→](example/dspy/simple_loop.ipynb) | [→](example/transformers/simple_loop.ipynb) |

## Documentation

- [Core Package](core/overview.md) — Foundational abstractions and types
- [Integrations](integrations/langchain/overview.md) — Framework-specific packages
- [API Reference](api-reference.md) — Auto-generated API documentation

## Status

This project is in active development. Components are subject to change between versions.
