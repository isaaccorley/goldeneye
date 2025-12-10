from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype


class EarthDial(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id, device=device)
        self.device = get_device(device)
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=get_dtype(self.device),
                device_map=self.device,
                trust_remote_code=True,
            )
        except Exception as e:
            msg = f"Failed to load EarthDial model {model_id}: {e}"
            raise RuntimeError(msg) from e
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
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(images=[pil_image], text=prompt_text, return_tensors="pt")
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
