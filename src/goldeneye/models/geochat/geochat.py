from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from goldeneye.models.base import BaseAgent
from goldeneye.models.geochat.modeling_geochat import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    GeoChatLlamaForCausalLM,
    process_images,
    tokenizer_image_token,
)
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report


class GeoChat(BaseAgent):
    model: Any

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
        self.tokenizer = AutoTokenizer.from_pretrained(codename, use_fast=False)
        config = AutoConfig.from_pretrained(codename)
        config.architectures = ["GeoChatLlamaForCausalLM"]
        config._name_or_path = codename
        if not hasattr(config, "auto_map") or config.auto_map is None:
            config.auto_map = {}
        config.auto_map.update(
            {
                "AutoModelForCausalLM": (
                    "goldeneye.models.geochat.modeling_geochat.GeoChatLlamaForCausalLM"
                ),
            }
        )
        load_kwargs: dict[str, Any] = {"config": config}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["dtype"] = self.dtype
            load_kwargs["device_map"] = self.device
        self.model = GeoChatLlamaForCausalLM.from_pretrained(codename, **load_kwargs)
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
        self._load_finetuned_vision_weights(vision_tower)
        vision_tower.to(device=self.device, dtype=self.dtype)
        self.image_processor = vision_tower.image_processor

    def _load_finetuned_vision_weights(self, vision_tower: Any) -> None:
        shard_path = hf_hub_download(self.codename, "pytorch_model-00002-of-00002.bin")
        state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
        vision_prefix = "model.vision_tower.vision_tower."
        vision_state_dict = {
            k[len(vision_prefix) :]: v for k, v in state_dict.items() if k.startswith(vision_prefix)
        }
        if not vision_state_dict or vision_tower.vision_tower is None:
            return

        pos_key = "vision_model.embeddings.position_embedding.weight"
        if pos_key in vision_state_dict:
            pos_emb = vision_state_dict.pop(pos_key)
            current_pos_emb = vision_tower.vision_tower.vision_model.embeddings.position_embedding
            if pos_emb.shape[0] != current_pos_emb.num_embeddings:
                pos_emb = self._interpolate_position_embedding(
                    pos_emb, current_pos_emb.num_embeddings
                )
            vision_state_dict[pos_key] = pos_emb

        vision_tower.vision_tower.load_state_dict(vision_state_dict, strict=False)

    def _interpolate_position_embedding(
        self, pos_emb: torch.Tensor, target_size: int
    ) -> torch.Tensor:
        seq_length, hidden_dim = pos_emb.shape
        new_seq_length = target_size
        if seq_length == new_seq_length:
            return pos_emb
        cls_token = pos_emb[:1, :]
        patch_tokens = pos_emb[1:, :]
        seq_length_1d = int((seq_length - 1) ** 0.5)
        new_seq_length_1d = int((new_seq_length - 1) ** 0.5)
        patch_tokens = patch_tokens.reshape(1, seq_length_1d, seq_length_1d, hidden_dim)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)
        patch_tokens = torch.nn.functional.interpolate(
            patch_tokens.float(), size=new_seq_length_1d, mode="bicubic", align_corners=True
        )
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(-1, hidden_dim)
        return torch.cat([cls_token, patch_tokens], dim=0)

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
