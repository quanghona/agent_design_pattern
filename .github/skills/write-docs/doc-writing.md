# Document Writing Guide

## Purpose

This reference defines the writing standards, structural templates, and quality criteria for all documentation in the AI Agent Pattern project. It ensures every document is clear, complete, consistent, and useful to readers.

## Document Structure Template

Every feature/component document should follow this structure:

```markdown
# <Component/Feature Name>

## Overview

A 2-4 sentence summary of what this component/feature does and why it matters.
Mention its role within the broader architecture.

## Architecture

Explain how this component fits into the system. Include:
- Where it sits in the component hierarchy
- How it interacts with other components
- Any key design patterns it implements

![diagram](img/diagram-name.jpg)

## Key Concepts

Break down the main concepts, parameters, or configuration options.
Use subsections for each major concept.

### <Concept 1>

Description of concept 1.

### <Concept 2>

Description of concept 2.

## Usage

### Basic Example

```python
from aap_core import <Component>

# Basic usage example
component = <Component>(param=value)
result = component.method(input_data)
```

### Advanced Example

```python
# More complex usage with multiple components or configuration
```

## Configuration

List and explain all configurable parameters, options, or settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param1` | `str` | `"default"` | Description of param1 |
| `param2` | `int` | `0` | Description of param2 |

## API Reference

For component documents, link to the auto-generated API docs:

See the full API reference: [`<Component>`][aap_core.<module>.<Component>]

Or use a mkdocstrings directive:

::: aap_core.<module>.<Component>
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Related Component](../related-component.md) — Brief description of relationship
- [Integration Guide](../../integrations/langchain/overview.md) — How to use with LangChain
```

## Section Guidelines

### 1. Overview Section

- **Length**: 2-4 sentences, maximum 1 paragraph
- **Content**: What it does, why it exists, its role in the system
- **Tone**: High-level, accessible to someone new to the project
- **Do NOT**: Go into implementation details or parameter lists

### 2. Architecture Section

- **Content**: Visual diagram (if applicable), component relationships, data flow
- **Diagrams**: Place in `docs/img/`, reference with `![alt](img/filename.jpg)`
- **Tone**: Technical but not implementation-specific
- **Do NOT**: Copy-paste code; describe the logic in words

### 3. Key Concepts Section

- **Content**: Break down the component into understandable pieces
- **Structure**: One subsection per concept/parameter group
- **Tone**: Explanatory, with concrete examples where helpful
- **Do NOT**: Repeat the API signature; explain the *meaning* behind parameters

### 4. Usage Section

- **Content**: Working code examples, from simple to complex
- **Examples must**:
  - Be accurate and match the current source code
  - Include imports
  - Be self-contained (no missing context)
  - Demonstrate the most common use cases first
- **Format**: Use Python code blocks with syntax highlighting
- **Do NOT**: Show incomplete or pseudocode examples

### 5. Configuration Section

- **Content**: All configurable options with types, defaults, and descriptions
- **Format**: Use a table for parameters; use prose for complex options
- **Do NOT**: Omit optional parameters; document every public option

### 6. API Reference Section

- **Content**: Link to auto-generated docs or embed `:::` directives
- **Do NOT**: Manually document every method; let mkdocstrings handle it

### 7. See Also Section

- **Content**: Links to related documents that provide additional context
- **Format**: Bullet list with brief descriptions
- **Do NOT**: Include more than 3-5 references; avoid redundancy

## Style Guidelines

### Tone and Voice

- **Use active voice**: "The Agent processes messages" not "Messages are processed by the Agent"
- **Be concise**: Remove filler words and redundant phrases
- **Be specific**: Use exact component names, parameter names, and types
- **Avoid marketing language**: No "powerful", "revolutionary", "cutting-edge"

### Terminology

- Use consistent terminology throughout:
  - "agent" (lowercase) for the concept, `Agent` (capitalized) for the class
  - "message" for `AgentMessage` objects
  - "chain" for `Chain` objects
  - "orchestration" for the pattern of coordinating agents
- Refer to packages by their import name: `aap_core`, `aap_langchain`, etc.
- Refer to frameworks by their proper name: LangChain, LlamaIndex, DSPy, Hugging Face Transformers

### Code Examples

```python
# ✅ Good: Complete, accurate, with context
from aap_core import Agent, Chain

agent = Agent(model="gpt-4")
chain = Chain([agent, agent])
result = chain.run("Hello")

# ❌ Bad: Incomplete, missing imports, unclear
agent = Agent()
result = agent.run()
```

Rules for code examples:
1. Always include imports
2. Use realistic parameter values (not `None` or empty strings unless that's the point)
3. Show the most common pattern first
4. Keep examples short — remove unnecessary boilerplate
5. Verify examples against the actual source code before including them

### Cross-References

- Use relative paths for internal links: `[Agent](../core/components/agent.md)`
- Use mkdocstrings cross-references for API items: `[Agent][aap_core.agent.Agent]`
- Link to the **overview** page when introducing a component for the first time
- Link to the **API reference** when discussing specific methods or parameters

## Quality Criteria

Every document must pass these checks before being considered complete:

### Completeness
- [ ] Overview explains what and why
- [ ] Architecture describes component relationships
- [ ] All key concepts are covered
- [ ] At least one basic usage example is provided
- [ ] All public parameters/options are documented
- [ ] API reference is linked or embedded

### Accuracy
- [ ] Code examples match the current source code
- [ ] Parameter names, types, and defaults are correct
- [ ] Component and class names are accurate
- [ ] Links resolve to existing pages

### Readability
- [ ] Each section has a clear purpose
- [ ] Headings are descriptive and consistent
- [ ] Code examples are self-contained and runnable
- [ ] No jargon without explanation
- [ ] Sentences are concise and active

### Non-Duplication
- [ ] Core concepts are explained once (in core docs)
- [ ] Integration docs reference core docs instead of repeating
- [ ] See Also section points to related content instead of duplicating it

## Document Types and Variations

### Component Documentation (`docs/core/components/*.md`)

Follow the full template above. Focus on:
- The component's role in the core architecture
- How to instantiate and use it
- Its parameters and configuration
- Its interactions with other components

### Integration Overview (`docs/integrations/*/overview.md`)

Slightly different structure:

```markdown
# <Integration> Integration

## Overview

What this integration provides and how it extends the core.

## Architecture

How the integration maps core concepts to the framework's concepts.

![diagram](img/integration-architecture.jpg)

## Key Differences from Core

What changes or adds functionality compared to the core package.

## Usage

Framework-specific setup and usage examples.

## See Also
- [Core Agent](../../core/components/agent.md)
- [Core Chain](../../core/components/chain.md)
```

### Example Documentation (`docs/examples/*.md`)

Simplified structure focused on demonstrating a pattern:

```markdown
# <Pattern Name> Pattern

## Overview

What this pattern achieves and when to use it.

## How It Works

Step-by-step explanation of the pattern's flow.

![diagram](img/pattern-flow.jpg)

## Example

```python
# Complete working example
```

## When to Use

- Use case 1
- Use case 2

## When Not to Use

- Scenario where this pattern is not appropriate
```

### Getting Started Documentation (`docs/getting-started/*.md`)

Audience-focused, not component-focused:

```markdown
# <Topic>

## Overview

What the reader will learn and who it's for.

## Prerequisites

What the reader needs before proceeding.

## Steps

Step-by-step instructions with code examples.

## Next Steps

What to read or do next.
```

## Common Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Repeating API signatures | Duplicates auto-generated docs | Link to API reference instead |
| Copy-pasting core docs in integration docs | Creates maintenance burden | Cross-reference the core doc |
| Overly long overviews | Buries key information | Keep to 2-4 sentences |
| Missing imports in examples | Examples don't run | Always include full imports |
| Absolute paths in links | Breaks when docs are hosted | Use relative paths |
| No diagram for complex flows | Hard to visualize | Add architecture diagrams |
| Vague parameter descriptions | Reader doesn't understand | Be specific about purpose and constraints |
