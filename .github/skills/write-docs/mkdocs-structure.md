# mkdocs.yml Structure Guide

## Purpose

This reference defines the canonical `mkdocs.yml` structure for the AI Agent Pattern project and provides rules for maintaining it as new features and documentation are added.

## Canonical Structure

```yaml
site_name: AI Agent Pattern
site_url: https://quanghona.github.io/agent_design_pattern
repo_url: https://github.com/quanghona/agent_design_pattern
theme:
  name: material
  features:
    - content.code.copy
    - content.code.select
    - navigation.instant
    - navigation.tracking
    - navigation.sections
    - navigation.top
    - search.highlight
    - search.share
    - search.suggest
  palette:
    - scheme: default
      primary: blue
      accent: blue
    - scheme: slate
      primary: blue
      accent: blue

nav:
  - Home: home.md
  - Getting Started:
      - Introduction: getting-started/introduction.md
      - Installation: getting-started/installation.md
  - Core:
      - Overview: core/overview.md
      - Components:
          - Agent: core/components/agent.md
          - Chain: core/components/chain.md
          - Policy Trainer: core/components/policy_trainer.md
          - Retriever: core/components/retriever.md
          - Prompt Augmenter: core/components/prompt_augmenter.md
      - API Reference: core/api.md
  - Integrations:
      - LangChain:
          - Overview: integrations/langchain/overview.md
          - API Reference: integrations/langchain/api.md
      - LlamaIndex:
          - Overview: integrations/llamaindex/overview.md
          - API Reference: integrations/llamaindex/api.md
      - DSPy:
          - Overview: integrations/dspy/overview.md
          - API Reference: integrations/dspy/api.md
      - Transformers:
          - Overview: integrations/transformers/overview.md
          - API Reference: integrations/transformers/api.md
  - Examples:
      - Sequential: examples/sequential.md
      - Parallel: examples/parallel.md
      - Self-Reflection: examples/self_reflection.md
      - Coordinator: examples/coordinator.md
      - Voting: examples/voting.md
      - Debate: examples/debate.md
      - Cross-Reflection: examples/cross_reflection.md
      - Iterative Refinement: examples/iterative_refinement.md
      - Retriever: examples/retriever.md
  - API Reference: api-reference.md

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths:
            - src/core/src/aap_core
            - src/dspy/src/aap_dspy
            - src/langchain/src/aap_langchain
            - src/llamaindex/src/aap_llamaindex
            - src/transformers/src/aap_transformers
          options:
            filters:
              - "!^_"
            show_root_heading: true
            show_root_full_path: false
            show_signature: true
            show_signature_annotations: true
            show_docstring_style: google
            members_order: source
            inherited_members: true
            docstring_style: google
```

## Structure Principles

### 1. Navigation Hierarchy

The navigation follows a top-down structure:

| Level | Section | Purpose |
|-------|---------|---------|
| 1 | Home | Project landing page |
| 1 | Getting Started | Onboarding content |
| 1 | Core | Foundation package documentation |
| 1 | Integrations | Framework-specific packages (LangChain, LlamaIndex, DSPy, Transformers) |
| 1 | Examples | Usage examples and patterns |
| 1 | API Reference | Full API reference index |

### 2. Package Documentation Pattern

Each package (core + integrations) follows the same pattern:

```
<package>/
  overview.md          # High-level description, key concepts
  components/          # Individual component docs
    component_a.md
    component_b.md
  api.md               # Auto-generated API docs via mkdocstrings ::: directives
```

### 3. Integration Pattern

Each integration package has:
- An overview page explaining how the integration works
- Component-level docs for package-specific components
- An API reference page using `:::` directives

## Updating mkdocs.yml

### When Adding a New Feature Document

1. **Determine the section**: Which top-level section does the document belong to?
   - Core feature → `Core/`
   - Integration-specific → `Integrations/<integration>/`
   - Cross-cutting example → `Examples/`

2. **Add the navigation entry**: Insert the entry in the correct position within the `nav` section. Use consistent indentation (2 spaces per level).

3. **Maintain alphabetical or logical order**: Within a subsection, keep entries in a consistent order (alphabetical or by importance).

### When Adding a New Integration Package

1. Add a new entry under `Integrations/` in the navigation.
2. Add the package source path to `plugins.mkdocstrings.handlers.python.paths`.
3. Create the corresponding directory under `docs/integrations/<package>/`.

### When Adding a New Component to an Existing Package

1. Add the component doc under the appropriate `components/` subsection in `nav`.
2. Create the file in `docs/<package>/components/<component>.md`.

## Synchronization Rules

To keep `mkdocs.yml` in sync with the actual document structure:

1. **Every document file must have a navigation entry.** If a file exists in `docs/` but is not in `nav`, add it.
2. **Every navigation entry must point to an existing file.** If a file is deleted or moved, update or remove the entry.
3. **Directory structure must match navigation hierarchy.** A nav entry `Core/Components/Agent: core/components/agent.md` means the file must be at `docs/core/components/agent.md`.
4. **Run a validation check** after any documentation change:
   - Verify all nav paths resolve to existing files
   - Verify all files in `docs/` (excluding `img/` and `asset/`) have nav entries

## Common Pitfalls

- **Broken relative links**: Always use paths relative to the document root (`docs/`), not relative to the current file.
- **Inconsistent indentation**: Use 2-space indentation consistently throughout `mkdocs.yml`.
- **Missing search plugin**: Always include the `search` plugin for discoverability.
- **Outdated paths**: When moving files, update both the file and the nav entry.
- **Duplicate nav entries**: Avoid having multiple nav entries pointing to the same file.
