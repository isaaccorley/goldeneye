from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype

DEFAULT_PROCESSOR = "Qwen/Qwen2.5-VL-3B-Instruct"


class GeoR1(BaseGeoVLM):
    processor: Any
    model: Qwen2_5_VLForConditionalGeneration

    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id, device=device)
        self.device = get_device(device)
        self.processor = AutoProcessor.from_pretrained(DEFAULT_PROCESSOR, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=get_dtype(self.device),
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

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
        with torch.inference_mode():
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
        return output_text[0].strip()
