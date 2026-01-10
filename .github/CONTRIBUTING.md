# Contributing to goldeneye

Thank you for your interest in contributing to goldeneye!

## Development Setup

```bash
git clone https://github.com/isaaccorley/goldeneye.git
cd goldeneye
uv sync --dev
uv run pre-commit install
```

## Running Tests

```bash
uv run pytest -vvv
```

## Code Quality

We use pre-commit hooks to ensure code quality. Run all checks with:

```bash
uv run pre-commit run --all-files
```

This runs:

- **ruff** - Linting and formatting
- **ty** - Type checking
- **numpydoc-validation** - Docstring validation

## Adding a New Model

1. Create `src/goldeneye/models/<name>/` with `__init__.py` and `<name>.py`
1. Subclass `BaseAgent` and implement the `recon()` method
1. Register in `registry.py`: add to `_AGENT_REGISTRY` and `_AGENT_CLASS_PATHS`
1. Add unit test in `tests/`

## Pull Requests

1. Fork the repository
1. Create a feature branch
1. Make your changes
1. Ensure all tests pass and pre-commit checks are clean
1. Submit a pull request
