import pytest

import goldeneye


def test_list_models() -> None:
    models = goldeneye.list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "GeoR1" in models
    assert "GeoLLaVA-8K" in models
    assert "geochat-7B" in models
    assert "SAM3" in models


def test_load_model_invalid() -> None:
    with pytest.raises(ValueError, match="Model.*not found"):
        goldeneye.load_agent("InvalidModel")


def test_load_model_valid() -> None:
    model = goldeneye.load_agent("GeoR1", device="cpu")
    assert model is not None
    assert hasattr(model, "recon")
    assert callable(model)
