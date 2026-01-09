import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from goldeneye.models.base import BaseAgent
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report

INTERNVL_BASE_MODEL = "OpenGVLab/InternVL-Chat-V1-5"


class EarthDial(BaseAgent):
    def __init__(
        self, codename: str, device: str | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__(codename, device=device, dtype=dtype)
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)
        old_hf_transfer = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        try:
            try:
                self.processor = AutoProcessor.from_pretrained(codename, trust_remote_code=True)
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
                    codename,
                    dtype=self.dtype,
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
                        dtype=self.dtype,
                        device_map=self.device,
                        trust_remote_code=True,
                    )
                    from safetensors.torch import load_file

                    weight_index_path = hf_hub_download(
                        codename, "model.safetensors.index.json", repo_type="model"
                    )
                    with open(weight_index_path) as f:
                        weight_index = json.load(f)
                    weight_files = [
                        hf_hub_download(codename, f, repo_type="model")
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
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(images=[pil_image], text=prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
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
