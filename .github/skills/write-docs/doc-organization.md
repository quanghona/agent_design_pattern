# Document Organization Guide

## Purpose

This reference defines the folder structure conventions for the `docs/` directory and provides rules for placing new documents in the correct location.

## Folder Structure

```
docs/
├── index.md                          # Project landing page
├── api-reference.md                  # Full API reference index
├── img/                              # All images
│   ├── architecture.jpg
│   └── ...
├── asset/                            # Other assets (downloads, etc.)
│   └── ...
├── getting-started/                  # Onboarding content
│   ├── introduction.md
│   └── installation.md
├── core/                             # Core package docs
│   ├── overview.md
│   ├── components/                   # Component-level docs
│   │   ├── agent.md
│   │   ├── chain.md
│   │   ├── policy_trainer.md
│   │   ├── retriever.md
│   │   └── prompt_augmenter.md
│   └── api.md
├── integrations/                     # Integration package docs
│   ├── langchain/
│   │   ├── overview.md
│   │   └── api.md
│   ├── llamaindex/
│   │   ├── overview.md
│   │   └── api.md
│   ├── dspy/
│   │   ├── overview.md
│   │   └── api.md
│   └── transformers/
│       ├── overview.md
│       └── api.md
└── examples/                         # Usage examples
    ├── sequential.md
    ├── parallel.md
    ├── self_reflection.md
    ├── coordinator.md
    ├── voting.md
    ├── debate.md
    ├── cross_reflection.md
    ├── iterative_refinement.md
    └── retriever.md
```

## Placement Rules

### Rule 1: Document Type Determines Location

| Document Type | Location | Example |
|---------------|----------|---------|
| Project landing | `docs/index.md` | (already exists) |
| API reference index | `docs/api-reference.md` | (already exists) |
| Getting started / onboarding | `docs/getting-started/` | introduction.md, installation.md |
| Core package overview | `docs/core/overview.md` | High-level core concepts |
| Core component docs | `docs/core/components/` | One file per component |
| Integration overview | `docs/integrations/<name>/overview.md` | One per integration |
| Integration API reference | `docs/integrations/<name>/api.md` | Auto-generated via mkdocstrings |
| Usage examples | `docs/examples/` | One file per pattern |

### Rule 2: One Concept Per File

Each `.md` file should cover **one** concept or component:
- `docs/core/components/agent.md` — documents the Agent component only
- `docs/core/components/chain.md` — documents the Chain component only
- Do NOT combine multiple components into a single file

### Rule 3: Mirror Source Code Structure

The `docs/` hierarchy should mirror the `src/` hierarchy:

```
src/core/src/aap_core/
├── agent.py          → docs/core/components/agent.md
├── chain.py          → docs/core/components/chain.md
├── policy_trainer.py → docs/core/components/policy_trainer.md
├── retriever.py      → docs/core/components/retriever.md
└── prompt_augmenter.py → docs/core/components/prompt_augmenter.md

src/langchain/src/aap_langchain/  → docs/integrations/langchain/
src/llamaindex/src/aap_llamaindex/ → docs/integrations/llamaindex/
src/dspy/src/aap_dspy/             → docs/integrations/dspy/
src/transformers/src/aap_transformers/ → docs/integrations/transformers/
```

### Rule 4: Asset Placement

| Asset Type | Location |
|------------|----------|
| Images (diagrams, screenshots) | `docs/img/` |
| Downloads, data files | `docs/asset/` |

**Referencing assets:**
- Images: `![alt text](img/filename.jpg)` — path is relative to `docs/` root
- Other assets: `[link](asset/filename.pdf)` — path is relative to `docs/` root

## Naming Conventions

- **File names**: lowercase, hyphen-separated (kebab-case)
  - ✅ `policy_trainer.md` → `policy-trainer.md`
  - ❌ `PolicyTrainer.md`
  - ❌ `policy_trainer_file.md`
- **Directory names**: lowercase, hyphen-separated (kebab-case)
  - ✅ `getting-started/`, `core/`, `integrations/`
  - ❌ `GettingStarted/`, `Core/`
- **Overview files**: always named `overview.md`
- **API reference files**: always named `api.md`
- **Example files**: named after the pattern they demonstrate
  - ✅ `sequential.md`, `self_reflection.md`, `coordinator.md`

## Cross-Reference Strategy

When documents overlap in content:

1. **Write once, reference elsewhere**: The canonical description lives in one file. Other files link to it.
2. **Use relative paths for cross-references**:
   - From `docs/core/components/agent.md` to `docs/core/components/chain.md`: `[Chain](../chain.md)`
   - From `docs/integrations/langchain/overview.md` to `docs/core/components/agent.md`: `[Agent](../../core/components/agent.md)`
3. **Avoid duplicating explanations**: If a concept is explained in the core docs, reference it from integration docs rather than repeating the explanation.

## Workflow for Adding a New Document

1. **Identify the document type** (component, integration, example, getting-started)
2. **Determine the target directory** using the rules above
3. **Create the file** with the correct kebab-case name
4. **Add the navigation entry** in `mkdocs.yml` (see [mkdocs-structure.md](./mkdocs-structure.md))
5. **Write the content** following the guidelines in [doc-writing.md](./doc-writing.md)
