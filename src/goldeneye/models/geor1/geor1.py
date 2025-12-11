from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from goldeneye.models.base import BaseAgent
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report

DEFAULT_PROCESSOR = "Qwen/Qwen2.5-VL-3B-Instruct"


class GeoR1(BaseAgent):
    processor: Any
    model: Qwen2_5_VLForConditionalGeneration

    def __init__(
        self, codename: str, device: str | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__(codename, device=device, dtype=dtype)
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)
        self.processor = AutoProcessor.from_pretrained(DEFAULT_PROCESSOR, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            codename,
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs: dict[str, Any] = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        input_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        response = output_text[0].strip()
        return Report(image=image, prompt=prompt, response=response)
