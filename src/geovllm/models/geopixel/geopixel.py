from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GeoPixel(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id, device=device)
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, padding_side="right", use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        model_kwargs: dict[str, Any] = {
            "vision_pretrained": "facebook/sam2-hiera-large",
            "seg_token_idx": seg_token_idx,
            "bop_token_idx": self.tokenizer("<p>", add_special_tokens=False).input_ids[0],
            "eop_token_idx": self.tokenizer("</p>", add_special_tokens=False).input_ids[0],
            "dtype": get_dtype(self.device),
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            device_map=self.device,
            trust_remote_code=True,
            **model_kwargs,
        )
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.tokenizer = self.tokenizer
        self.model.eval()

    def _to_path(self, image: str | Path | Image.Image) -> str:
        if isinstance(image, (str, Path)):
            return str(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.convert("RGB").save(f.name)
            return f.name

    def __call__(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> str:
        return self.generate(image, prompt, max_new_tokens=max_new_tokens)

    def generate(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> str:
        response, _ = self._evaluate(image, prompt, max_new_tokens)
        return response

    def generate_with_masks(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> tuple[str, list[NDArray[np.uint8]]]:
        response, pred_masks = self._evaluate(image, prompt, max_new_tokens)
        masks = [
            (m.detach().cpu().numpy() > 0).astype(np.uint8)
            for m in pred_masks
            if isinstance(m, torch.Tensor)
        ]
        return response, masks

    def _evaluate(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int
    ) -> tuple[str, list]:
        image_path = self._to_path(image)
        device_type = (self.device or "cpu").split(":")[0]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return self.model.evaluate(  # type: ignore[union-attr]
                self.tokenizer, prompt, images=[image_path], max_new_tokens=max_new_tokens
            )
