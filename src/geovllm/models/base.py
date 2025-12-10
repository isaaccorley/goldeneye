from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class BaseGeoVLM(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def __call__(self, image: str | Path | Image.Image, prompt: str) -> str:
        pass

    @abstractmethod
    def generate(self, image: str | Path | Image.Image, prompt: str) -> str:
        pass
