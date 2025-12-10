from pathlib import Path

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype


class GeoR1(BaseGeoVLM):
    def __init__(
        self,
        model_id: str,
        device: str | None = None,
    ) -> None:
        super().__init__(model_id)
        self.device = get_device(device)
        base_model_id = "miniHui/Geo-R1"
        try:
            try:
                self.processor = Qwen2_5_VLProcessor.from_pretrained(
                    model_id, trust_remote_code=True
                )
            except (OSError, ValueError):
                self.processor = Qwen2VLProcessor.from_pretrained(
                    base_model_id, trust_remote_code=True
                )
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=get_dtype(self.device),
                    device_map=self.device,
                    trust_remote_code=True,
                )
            except (OSError, ValueError):
                try:
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_id,
                        dtype=get_dtype(self.device),
                        device_map=self.device,
                        trust_remote_code=True,
                    )
                except (OSError, ValueError):
                    from transformers import AutoModel

                    self.model = AutoModel.from_pretrained(
                        model_id,
                        dtype=get_dtype(self.device),
                        device_map=self.device,
                        trust_remote_code=True,
                    )
        except Exception as e:
            msg = f"Failed to load model {model_id}: {e}"
            raise RuntimeError(msg) from e
        self.model.eval()

    def _load_image(self, image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def __call__(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 512
    ) -> str:
        return self.generate(image, prompt, max_new_tokens=max_new_tokens)

    def generate(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 512
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
        image_inputs, video_inputs = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=True)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
