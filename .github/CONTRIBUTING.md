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
2. Subclass `BaseAgent` and implement the `recon()` method
3. Register in `registry.py`: add to `_AGENT_REGISTRY` and `_AGENT_CLASS_PATHS`
4. Add unit test in `tests/`

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass and pre-commit checks are clean
5. Submit a pull request
