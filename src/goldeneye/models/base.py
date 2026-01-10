from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image
from transformers import BitsAndBytesConfig

from goldeneye.report import Report


class BaseAgent(ABC):
    def __init__(
        self,
        codename: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        quantization_config: BitsAndBytesConfig | None = None,
    ) -> None:
        self.codename = codename
        self.device = device
        self.dtype = dtype
        self.quantization_config = quantization_config

    @abstractmethod
    @torch.inference_mode()
    def recon(
        self,
        image: str | Path | Image.Image,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 64,
    ) -> Report:
        pass

    def __call__(
        self,
        image: str | Path | Image.Image,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 64,
    ) -> Report:
        return self.recon(image, prompt, max_new_tokens)

    def referring_segmentation(
        self,
        image: str | Path | Image.Image,  # noqa: ARG002
        prompt: str,  # noqa: ARG002
    ) -> tuple[list, list] | None:
        return None
