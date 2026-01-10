from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

from goldeneye.models.base import BaseAgent
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report


class GeoZero(BaseAgent):
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
        repo_id = codename
        self.processor = AutoProcessor.from_pretrained(
            repo_id, subfolder="GeoZero-8B-without-RFT", trust_remote_code=True
        )
        load_kwargs: dict[str, Any] = {
            "subfolder": "GeoZero-8B-without-RFT",
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["dtype"] = self.dtype
            load_kwargs["device_map"] = self.device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(repo_id, **load_kwargs)
        self.model.eval()

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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=True)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
        return Report(image=image, prompt=prompt, response=response)
