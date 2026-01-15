from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from goldeneye.models.base import BaseAgent
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report


class Florence2DOTA(BaseAgent):
    """Florence2-DOTA agent for oriented object detection in aerial images.

    Based on Florence-2-large fine-tuned on DOTA for oriented object detection
    using the LMMRotate framework.

    Parameters
    ----------
    codename : str
        HuggingFace model identifier (e.g., 'Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate')
    device : str | None, optional
        Target device ('cuda', 'cpu', 'mps'), by default None (auto-detect)
    dtype : torch.dtype | None, optional
        Model dtype (torch.float16, torch.bfloat16, etc.), by default None
    quantization_config : BitsAndBytesConfig | None, optional
        Quantization config for 4-bit or 8-bit loading, by default None
    """

    processor: Any
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
        self.processor = AutoProcessor.from_pretrained(codename, trust_remote_code=True)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = self.dtype
            load_kwargs["device_map"] = self.device
        self.model = AutoModelForCausalLM.from_pretrained(codename, **load_kwargs)
        self.model.eval()

    def recon(
        self,
        image: str | Path | Image.Image,
        prompt: str = "<OD>",
        max_new_tokens: int = 1024,
    ) -> Report:
        """Run inference on an image with a task prompt.

        Parameters
        ----------
        image : str | Path | Image.Image
            Input image (file path or PIL Image)
        prompt : str, optional
            Task prompt for Florence-2. Common prompts:
            - '<OD>' for object detection
            - '<CAPTION>' for image captioning
            - '<DETAILED_CAPTION>' for detailed captioning
            - '<MORE_DETAILED_CAPTION>' for more detailed captioning
            - '<OCR>' for optical character recognition
            By default '<OD>' for object detection.
        max_new_tokens : int, optional
            Maximum number of tokens to generate, by default 1024

        Returns
        -------
        Report
            Report containing image, prompt, and generated response
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(pil_image.width, pil_image.height)
        )

        # Convert parsed answer to string representation
        if isinstance(parsed_answer, dict):
            response = str(parsed_answer.get(prompt, parsed_answer))
        else:
            response = str(parsed_answer)

        return Report(image=image, prompt=prompt, response=response)
