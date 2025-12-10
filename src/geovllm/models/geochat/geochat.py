from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from geovllm.models.base import BaseGeoVLM
from geovllm.models.geochat.modeling_geochat import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    GeoChatConfig,
    GeoChatLlamaForCausalLM,
    process_images,
    tokenizer_image_token,
)
from geovllm.models.utils import get_device, get_dtype

_ = GeoChatConfig
_ = GeoChatLlamaForCausalLM


class GeoChat(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id, device=device)
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=get_dtype(self.device),
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
        vision_tower.to(device=self.device, dtype=get_dtype(self.device))
        self.image_processor = vision_tower.image_processor

    def _load_image(self, image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def __call__(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> str:
        return self.generate(image, prompt, max_new_tokens=max_new_tokens)

    def generate(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> str:
        pil_image = self._load_image(image)
        image_tensor = process_images([pil_image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.device, dtype=get_dtype(self.device))

        conv_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            f"USER: <image>\n{prompt} ASSISTANT:"
        )
        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        output_ids_trimmed = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(output_ids_trimmed, skip_special_tokens=True).strip()
