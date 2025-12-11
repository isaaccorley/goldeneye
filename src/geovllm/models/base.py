from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class BaseGeoVLM(ABC):
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def __call__(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> str:
        pass

    @abstractmethod
    def generate(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> str:
        pass

    def referring_segmentation(
        self, image: str | Path | Image.Image, prompt: str
    ) -> tuple[list, list] | None:
        return None
