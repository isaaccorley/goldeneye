from pathlib import Path

import pytest
import torch

import goldeneye

LIGHT_MODELS = [
    "Geo-R1-3B-GRPO-REC-5shot",
]

HEAVY_MODELS = [
    "EarthDial-4B-RGB",
    "geochat-7B",
    "GeoLLaVA-8K",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_assets() -> None:
    models = goldeneye.assets()
    assert isinstance(models, list)
    assert len(models) > 0


def test_dispatch_agent_invalid() -> None:
    with pytest.raises(ValueError, match="Model.*not found"):
        goldeneye.dispatch_agent("InvalidModel")


@pytest.mark.integration
@pytest.mark.parametrize("model_name", LIGHT_MODELS)
def test_model_inference_smoke(model_name: str, dummy_image: Path, dummy_prompt: str) -> None:
    model = goldeneye.dispatch_agent(model_name, device=DEVICE)
    output = model(dummy_image, dummy_prompt, max_new_tokens=4)
    assert isinstance(output, str)
    assert output != ""


@pytest.mark.skip(reason="Heavy models (4B-7B) - too slow for CI")
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("model_name", HEAVY_MODELS)
def test_model_inference_smoke_heavy(model_name: str, dummy_image: Path, dummy_prompt: str) -> None:
    model = goldeneye.dispatch_agent(model_name, device=DEVICE)
    output = model(dummy_image, dummy_prompt, max_new_tokens=4)
    assert isinstance(output, str)
    assert output != ""
