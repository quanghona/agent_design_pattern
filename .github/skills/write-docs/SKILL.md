---
name: write-docs
description: 'Write and maintain feature documentation for the AI Agent Pattern project. Use when: creating new feature docs, updating mkdocs.yml navigation, structuring the docs/ folder, writing API or concept documentation, or maintaining documentation consistency across the project.'
---

# Write Docs Skill

## When to Use

- Creating documentation for a new feature, component, or integration
- Updating `mkdocs.yml` navigation structure
- Restructuring the `docs/` folder organization
- Writing or rewriting feature documentation with consistent style
- Ensuring documentation stays in sync with codebase changes

## Prerequisites

Before writing documentation, understand the feature by:
1. Reading the source code in the relevant `src/{package}/src/aap_{package}/` directory
2. Understanding the component's role within the broader architecture
3. Checking existing docs to avoid duplication

## Procedure

This skill is composed of three reference files. Work through them **in order**, completing each part fully before moving to the next.

### Part 1: Update `mkdocs.yml` Navigation

Define the navigation structure and update `mkdocs.yml` accordingly.

👉 **Reference**: [mkdocs-structure.md](./mkdocs-structure.md)

### Part 2: Organize Document Files in `docs/`

Place the document file in the correct location within the `docs/` folder hierarchy.

👉 **Reference**: [doc-organization.md](./doc-organization.md)

### Part 3: Write the Document Content

Write the actual document content following the project's style and structure guidelines.

👉 **Reference**: [doc-writing.md](./doc-writing.md)

## Constraints

- All documents **MUST** be located in the `docs/` folder or its subfolders.
- Images **MUST** be located in `docs/img/` folder.
- Other assets **MUST** be located in `docs/asset/` folder.
- Use relative paths for all internal document links and asset references.
- Follow the existing project conventions for naming and structure.

## Quality Checklist

Before considering a documentation task complete, verify:

- [ ] `mkdocs.yml` navigation is updated and consistent with file structure
- [ ] Document is placed in the correct location in `docs/`
- [ ] All internal links use relative paths and resolve correctly
- [ ] Images are in `docs/img/` and referenced with relative paths
- [ ] Document follows the prescribed structure from `doc-writing.md`
- [ ] No duplicated information — cross-references used where appropriate
- [ ] Code examples are accurate and match the current source code
- [ ] The document is readable by someone unfamiliar with the feature
- [ ] All internal links and cross-references resolve correctly (no broken links)
