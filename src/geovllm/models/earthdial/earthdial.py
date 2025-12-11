import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype

INTERNVL_BASE_MODEL = "OpenGVLab/InternVL-Chat-V1-5"


class EarthDial(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id, device=device)
        self.device = get_device(device)
        old_hf_transfer = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        try:
            try:
                self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            except (OSError, FileNotFoundError, ValueError, EntryNotFoundError) as e:
                error_msg = str(e)
                if (
                    "configuration_internvl_chat" in error_msg
                    or "does not appear to have a file named" in error_msg
                ):
                    self.processor = AutoProcessor.from_pretrained(
                        INTERNVL_BASE_MODEL, trust_remote_code=True
                    )
                else:
                    raise
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=get_dtype(self.device),
                    device_map=self.device,
                    trust_remote_code=True,
                )
            except (OSError, FileNotFoundError, ValueError, EntryNotFoundError) as e:
                error_msg = str(e)
                if (
                    "configuration_internvl_chat" in error_msg
                    or "modeling_internvl_chat" in error_msg
                    or "does not appear to have a file named" in error_msg
                ):
                    base_model = AutoModelForCausalLM.from_pretrained(
                        INTERNVL_BASE_MODEL,
                        dtype=get_dtype(self.device),
                        device_map=self.device,
                        trust_remote_code=True,
                    )
                    from safetensors.torch import load_file

                    weight_index_path = hf_hub_download(
                        model_id, "model.safetensors.index.json", repo_type="model"
                    )
                    with open(weight_index_path) as f:
                        weight_index = json.load(f)
                    weight_files = [
                        hf_hub_download(model_id, f, repo_type="model")
                        for f in set(weight_index["weight_map"].values())
                    ]
                    state_dict = {}
                    for wf in weight_files:
                        state_dict.update(load_file(wf))
                    base_model.load_state_dict(state_dict, strict=False)
                    self.model = base_model
                else:
                    raise
        finally:
            if old_hf_transfer is not None:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer
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
