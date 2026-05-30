# Core Package

## Overview

The `aap_core` package is the foundational framework for the AI Agent Pattern project. It defines the abstractions and building blocks for constructing multi-agent systems: `Agent` as the fundamental execution unit, `Chain` for multi-turn reasoning loops, and `Orchestration` patterns for coordinating multiple agents. The package also provides supporting infrastructure including prompt augmentation, data retrieval, deduplication, guardrails, and policy-based reinforcement learning for prompt optimization.

## Architecture

The core package sits at the base of the architecture, providing framework-agnostic abstractions that all integration packages (`aap_langchain`, `aap_dspy`, `aap_llamaindex`, `aap_transformers`) build upon. It defines the shared type system (`AgentMessage`, `TokenUsage`, `BaseLLMChain`) and the `BaseAgent` / `BaseCausalMultiTurnsChain` interfaces that every integration must implement.

![diagram](img/core-architecture.jpg)

## Components

The core package provides the foundation for all other integrations:

- [Agent](components/agent.md) — The fundamental agent unit
- [Chain](components/chain.md) — Chaining agents together
- [Dedup](components/dedup.md) — Deduplication utilities
- [Guardrail](components/guardrail.md) — Safety constraints
- [Orchestration](components/orchestration.md) — Agent coordination patterns
- [Policy](components/policy.md) — Policy definitions
- [Policy Trainer](components/policy_trainer.md) — Policy optimization
- [Prompt Augmenter](components/prompt_augmenter.md) — Prompt enhancement
- [Retriever](components/retriever.md) — External data retrieval
- [Thought](components/thought.md) — Reasoning structures
- [Types](components/types.md) — Shared type definitions
- [Utils](components/utils.md) — Utility functions

## API Reference

::: aap_core
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Integrations](../integrations/langchain/overview.md) — Framework-specific packages built on core
- [Example Notebooks](../../example/langchain/) — Jupyter notebooks with usage examples
