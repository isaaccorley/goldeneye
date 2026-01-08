# AGENTS.md

Geospatial VLM wrapper library. Python 3.13+, uv/ruff/ty/pytest stack.

## Quick Reference

```bash
# Setup
uv sync --all-extras

# Validate (pre-commit runs ruff + ty + numpydoc + typos)
uv run pre-commit run --all-files

# Test
uv run pytest -vvv                           # unit tests (excludes @integration)
uv run pytest -vvv -k test_list_models       # single test
uv run pytest -vvv tests/test_models.py      # single file
uv run pytest -m integration                 # integration tests only
uv run pytest --cov=src                      # with coverage

# Type check (standalone)
uv run ty check --python=.venv/
```

## Project Structure

```
src/goldeneye/
  __init__.py          # exports: assets, dispatch_agent, hud
  report.py            # Report dataclass (image, prompt, response)
  hud.py               # render() for visualization with supervision
  models/
    base.py            # BaseAgent ABC with recon() and __call__
    registry.py        # _AGENT_REGISTRY, assets(), dispatch_agent()
    <model>/           # one subdir per model family
  datasets/            # HF dataset loaders (stream_* generators)
tests/
  conftest.py          # fixtures: dummy_image, dummy_prompt
  test_models.py       # unit tests (no GPU required)
  test_models_inference.py  # smoke tests (GOLDENEYE_RUN_INFERENCE=1)
```

## Code Style

### Imports (ruff I)

Order: stdlib -> third-party -> local. One blank line between groups.

```python
from __future__ import annotations       # always first if used

from pathlib import Path                 # stdlib
from typing import Any

import torch                             # third-party
from PIL import Image
from transformers import AutoProcessor

from goldeneye.models.base import BaseAgent   # local
from goldeneye.report import Report
```

### Type Annotations

Full annotations on all public APIs. Use `|` union syntax (3.10+).

```python
def recon(
    self,
    image: str | Path | Image.Image,
    prompt: str = "Describe this image.",
    max_new_tokens: int = 64,
) -> Report:
```

### Docstrings (numpydoc)

Required for public functions in `src/` (tests exempt). Validated by pre-commit.

```python
def stream_xlrs_bench(split: str = "train") -> Iterator[dict]:
    """Stream the XLRS-Bench-lite dataset sample by sample.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"

    Yields
    ------
    dict
        A single sample from the dataset
    """
```

### Naming

- Classes: `PascalCase` (e.g., `BaseAgent`, `GeoR1`, `Report`)
- Functions/methods: `snake_case` (e.g., `dispatch_agent`, `stream_xlrs_bench`)
- Constants: `SCREAMING_SNAKE` (e.g., `DEFAULT_PROCESSOR`, `_AGENT_REGISTRY`)
- Private: prefix `_` (e.g., `_load_image`)

### Error Handling

Raise with descriptive message. Use `msg` variable for long messages.

```python
if codename not in _AGENT_REGISTRY:
    msg = f"Agent {codename} not found. Available agents: {assets()}"
    raise ValueError(msg)
```

### Class Pattern (BaseAgent subclass)

```python
class GeoR1(BaseAgent):
    processor: Any  # type hint for attrs set in __init__
    model: Qwen2_5_VLForConditionalGeneration

    def __init__(
        self, codename: str, device: str | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__(codename, device=device, dtype=dtype)
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)
        # load model...
        self.model.eval()

    @torch.inference_mode()
    def recon(self, image, prompt, max_new_tokens) -> Report:
        # implementation
        return Report(image=image, prompt=prompt, response=response)
```

## Testing

### Unit Tests (default)

No GPU, no model downloads. Mock or use fixtures.

```python
def test_list_models() -> None:
    models = goldeneye.list_models()
    assert isinstance(models, list)
    assert "GeoR1" in models
```

### Integration Tests

Mark with `@pytest.mark.integration`. Excluded by default.

```python
@pytest.mark.integration
def test_model_inference(dummy_image: Path) -> None:
    model = goldeneye.load_agent("GeoR1", device="cpu")
    output = model(dummy_image, "Describe this.")
    assert output.response
```

### Fixtures (conftest.py)

- `dummy_image` - 8x8 gray PNG (session-scoped)
- `dummy_prompt` - "Describe the scene."

## Ruff Config

Line length: 100. Target: py313.

Enabled: ARG, B, C4, E, F, I, SIM, UP, W
Tests ignore: ANN001, ANN201, RUF029, SIM117, SIM300

## Adding a New Model

1. Create `src/goldeneye/models/<name>/` with `__init__.py` and `<name>.py`
2. Subclass `BaseAgent`, implement `recon()` method
3. Register in `registry.py`: add to `_AGENT_REGISTRY` and `_AGENT_CLASSES`
4. Add unit test in `tests/test_models.py`

## CI

GitHub Actions: `.github/workflows/ci.yaml`
- `test`: `uv run pytest -vvv --cov=src`
- `pre-commit`: `uv run pre-commit run --all-files -v`

## Key Dependencies

- torch/torchvision - ML framework
- transformers - HF model loading
- supervision - detection visualization
- datasets - HF dataset streaming
- qwen-vl-utils - Qwen VL preprocessing
- sam2/sam3 - segmentation models
