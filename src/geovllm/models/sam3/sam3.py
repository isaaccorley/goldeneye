from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype


class SAM3(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id)
        self.device = get_device(device)
        try:
            from transformers import Sam3Model, Sam3Processor

            self.processor = Sam3Processor.from_pretrained(model_id, trust_remote_code=True)
            self.model = Sam3Model.from_pretrained(
                model_id, dtype=get_dtype(self.device), device_map=self.device, trust_remote_code=True
            )
        except (ImportError, OSError, ValueError):
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_id, dtype=get_dtype(self.device), device_map=self.device, trust_remote_code=True
            )
        self.model.eval()

    def _load_image(self, image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def __call__(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 512  # noqa: ARG002
    ) -> str:
        return self.generate(image, prompt, max_new_tokens=max_new_tokens)

    def generate(
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 512  # noqa: ARG002
    ) -> str:
        pil_image = self._load_image(image)
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        try:
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            num_objects = len(results["masks"])
            avg_score = float(results["scores"].mean().item()) if len(results["scores"]) > 0 else 0.0
            return f"Found {num_objects} object(s) matching '{prompt}' with average confidence score {avg_score:.3f}"
        except (AttributeError, KeyError, TypeError):
            return f"Processed image with prompt '{prompt}' using SAM3 model"

