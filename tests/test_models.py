import pytest

import goldeneye


def test_assets() -> None:
    models = goldeneye.assets()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "Geo-R1-3B-GRPO-REC-5shot" in models
    assert "GeoLLaVA-8K" in models
    assert "geochat-7B" in models
    assert "SAM3" in models


def test_dispatch_agent_invalid() -> None:
    with pytest.raises(ValueError, match="Model.*not found"):
        goldeneye.dispatch_agent("InvalidModel")
