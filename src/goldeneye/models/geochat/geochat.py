from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from goldeneye.models.base import BaseAgent
from goldeneye.models.geochat.modeling_geochat import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    process_images,
    tokenizer_image_token,
)
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report


class GeoChat(BaseAgent):
    model: Any

    def __init__(
        self, codename: str, device: str | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__(codename, device=device, dtype=dtype)
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(codename, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            codename,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=self.device,
        )
        self._setup_tokenizer()
        self._setup_vision()
        self.model.eval()

    def _setup_tokenizer(self) -> None:
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _setup_vision(self) -> None:
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.device, dtype=self.dtype)
        self.image_processor = vision_tower.image_processor

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
        image_tensor = process_images([pil_image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        conv_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            f"USER: <image>\n{prompt} ASSISTANT:"
        )
        input_ids_result = tokenizer_image_token(conv_prompt, self.tokenizer, return_tensors="pt")
        assert isinstance(input_ids_result, torch.Tensor)
        input_ids = input_ids_result.unsqueeze(0).to(self.device)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        output_ids_trimmed = output_ids[0, input_ids.shape[1] :]
        response = self.tokenizer.decode(output_ids_trimmed, skip_special_tokens=True).strip()
        return Report(image=image, prompt=prompt, response=response)
