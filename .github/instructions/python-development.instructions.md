---
name: python-development
description: "Use when: working with Python development in this project — running scripts with uv, managing virtual environments, installing packages, or running tests with pytest"
applyTo: ["**/*.py", "**/*.md", "**/*.txt", "**/*.log", "**/*.yaml"]
---

# Python Development Instructions

## Environment Management

- **Always use `uv`** for Python environment management and script execution in this project.
- Use `uv run <script>` to execute Python scripts instead of `python` or `python3`.
- Use `uv sync` to install dependencies from `pyproject.toml` and `uv.lock`.
- Use `uv venv` to create virtual environments when needed.
- Use `uv pip install` for package management when `uv add` is not appropriate.
- The project uses a monorepo structure with multiple Python packages under `src/` (core, dspy, langchain, llamaindex, transformers). Each has its own `pyproject.toml`.

## Package Installation

- **Before installing any new Python package, stop and ask the user for permission.**
- When proposing a package, include:
  - Package name and version
  - Which sub-package it belongs to (core, dspy, langchain, llamaindex, transformers)
  - Whether it goes in `dependencies` or `dependency-groups.dev`
  - 1–2 alternative packages with brief trade-offs
- Use `uv add <package>` to add dependencies to the appropriate `pyproject.toml`.
- After installing packages, run `uv sync` to update the lockfile and environment.

## Testing

- **Use `pytest`** as the testing framework for all Python tests.
- Run tests with `uv run pytest` from the relevant package directory.
- Test files should be placed in the `tests/` directory within each package's `src/` folder.
- Use `uv run pytest -v` for verbose output during development.
- Use `uv run pytest --tb=short` for concise output.
- When writing tests, follow existing patterns in `test_implementation.py` at the project root.

## Code Structure

- Each sub-package under `src/` follows the layout:
  ```
  src/<package>/
    pyproject.toml
    README.md
    src/<module>/
      __init__.py
      ...
    tests/
      __init__.py
      test_*.py
  ```
- The core package (`aap_core`) is the foundation — other packages depend on it.
- Python version requirement: `>=3.10.12` across all packages.

## Linting & Formatting

- Use `ruff` (version `0.14.0`) for both linting and formatting.
- Run linting with `uv run ruff check <path>` from the relevant package directory.
- Run formatting with `uv run ruff format <path>` from the relevant package directory.
- Prefer `uv run ruff check --fix` to auto-fix linting issues when possible.
- Prefer `uv run ruff format` to auto-format code before committing.

## Implementation Workflow

- **Work on one function/method at a time.** Implement, run, and test it thoroughly before moving to the next.
- **Break multi-step features into individual steps.** Solve one step at a time — implement, test, verify, then proceed.
