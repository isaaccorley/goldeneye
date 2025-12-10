import pytest

import geovllm


def test_list_models() -> None:
    models = geovllm.list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "GeoR1" in models
    assert "GeoLLaVA-8K" in models
    assert "geochat-7B" in models
    assert "SAM3" in models


def test_load_model_invalid() -> None:
    with pytest.raises(ValueError, match="Model.*not found"):
        geovllm.load_model("InvalidModel")


def test_load_model_valid() -> None:
    model = geovllm.load_model("GeoR1", device="cpu")
    assert model is not None
    assert hasattr(model, "generate")
    assert hasattr(model, "__call__")
