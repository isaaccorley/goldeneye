from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

import goldeneye.models.geollava._longva_inference  # noqa: F401
from goldeneye.models.base import BaseAgent
from goldeneye.models.geollava._longva_inference import (
    IMAGE_TOKEN_INDEX,
    LlavaQwenForCausalLM,
    process_images,
)
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report

_GEOLLAVA_DIR: Final[Path] = Path(__file__).parent


def _ensure_longva_registered() -> None:
    pass


class GeoLLaVA(BaseAgent):
    def __init__(
        self,
        codename: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        quantization_config: BitsAndBytesConfig | None = None,
    ) -> None:
        super().__init__(
            codename, device=device, dtype=dtype, quantization_config=quantization_config
        )
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)
        _ensure_longva_registered()

        self._process_images = process_images

        self.tokenizer = AutoTokenizer.from_pretrained(codename, use_fast=False)

        config = AutoConfig.from_pretrained(codename)
        config.model_type = "llava_qwen"

        load_kwargs: dict[str, Any] = {
            "config": config,
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["dtype"] = self.dtype
            load_kwargs["device_map"] = self.device
        self.model = LlavaQwenForCausalLM.from_pretrained(codename, **load_kwargs)
        self.model.eval()

    def _tokenize_with_image_token(self, text: str) -> torch.Tensor:
        prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split("<image>")]

        def insert_separator(X: list, sep: list) -> list:
            return [ele for sublist in zip(X, [sep] * len(X), strict=False) for ele in sublist][:-1]

        input_ids: list[int] = []
        offset = 0
        if (
            len(prompt_chunks) > 0
            and len(prompt_chunks[0]) > 0
            and prompt_chunks[0][0] == self.tokenizer.bos_token_id
        ):
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])

        return torch.tensor(input_ids, dtype=torch.long)

    def recon(
        self,
        image: str | Path | Image.Image,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 64,
    ) -> Report:
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        image_size = pil_image.size

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.device, dtype=self.dtype)
        image_processor = vision_tower.image_processor

        image_tensor = self._process_images([pil_image], image_processor, self.model.config)
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = [image_tensor[i] for i in range(image_tensor.shape[0])]
        image_tensor = [t.to(self.device, dtype=self.dtype) for t in image_tensor]

        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = self._tokenize_with_image_token(formatted_prompt).unsqueeze(0).to(self.device)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        response = output_text
        return Report(image=image, prompt=prompt, response=response)
