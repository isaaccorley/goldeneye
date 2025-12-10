from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device


class EarthGPT(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id)
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=self.device, trust_remote_code=True
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_image(self, image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def __call__(self, image: str | Path | Image.Image, prompt: str) -> str:
        return self.generate(image, prompt)

    def generate(self, image: str | Path | Image.Image, prompt: str) -> str:
        _ = self._load_image(image)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output_text[len(prompt) :].strip()
