import os
from pathlib import Path

import pytest

import goldeneye

MODELS = [
    "EarthDial-4B-RGB",
    "geochat-7B",
    "GeoLLaVA-8K",
    "GeoPixel-7B",
    "Geo-R1-3B-GRPO-REC-5shot",
]


@pytest.mark.parametrize("model_name", MODELS)
def test_model_inference_smoke(model_name: str, dummy_image: Path, dummy_prompt: str) -> None:
    if os.getenv("GOLDENEYE_RUN_INFERENCE", "0") != "1":
        pytest.skip("Set GOLDENEYE_RUN_INFERENCE=1 to run inference smoke tests.")
    model = goldeneye.dispatch_agent(model_name, device="cpu")
    output = model(dummy_image, dummy_prompt, max_new_tokens=4)
    assert isinstance(output, str)
    assert output != ""
