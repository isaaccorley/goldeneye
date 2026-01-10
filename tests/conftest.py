from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture(scope="session")
def dummy_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "dummy.png"
    image = Image.new("RGB", (8, 8), color=(128, 128, 128))
    image.save(path)
    return path


@pytest.fixture(scope="session")
def dummy_prompt() -> str:
    return "Describe the scene."
