import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype


class GeoPixel(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id, device=device)
        self.device = get_device(device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="right",
                use_fast=False,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            seg_token_idx, bop_token_idx, eop_token_idx = [
                self.tokenizer(token, add_special_tokens=False).input_ids[0]
                for token in ["[SEG]", "<p>", "</p>"]
            ]
            kwargs: dict[str, Any] = {"torch_dtype": get_dtype(self.device)}
            geo_model_args = {
                "vision_pretrained": "facebook/sam2-hiera-large",
                "seg_token_idx": seg_token_idx,
                "bop_token_idx": bop_token_idx,
                "eop_token_idx": eop_token_idx,
            }
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                device_map=self.device,
                trust_remote_code=True,
                **kwargs,
                **geo_model_args,
            )
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.tokenizer = self.tokenizer
        except Exception as e:
            msg = f"Failed to load GeoPixel model {model_id}: {e}"
            raise RuntimeError(msg) from e
        self.model.eval()

    def _load_image(self, image: str | Path | Image.Image) -> str:
        if isinstance(image, (str, Path)):
            return str(image)
        if isinstance(image, Image.Image):
            tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_file.close()
            image.save(tmp_file.name)
            return tmp_file.name
        msg = f"Unsupported image type: {type(image)}"
        raise TypeError(msg)

    def __call__(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> str:
        return self.generate(image, prompt, max_new_tokens=max_new_tokens)

    def generate(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> str:
        image_path = self._load_image(image)
        if not hasattr(self.model, "evaluate"):
            msg = "Model does not have evaluate method. GeoPixel requires custom model class."
            raise AttributeError(msg)
        with torch.inference_mode():
            response, pred_masks = self.model.evaluate(
                self.tokenizer, query=prompt, images=[image_path], max_new_tokens=max_new_tokens
            )
        return response.replace("\n", " ").replace("  ", " ").strip()
