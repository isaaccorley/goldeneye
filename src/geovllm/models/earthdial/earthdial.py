import contextlib
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype


def _ensure_internvl_available() -> None:
    if "internvl" not in sys.modules:
        internvl_parent = Path(__file__).parent
        internvl_parent_str = str(internvl_parent)
        if internvl_parent_str not in sys.path:
            sys.path.insert(0, internvl_parent_str)
        with contextlib.suppress(ImportError):
            pass


class EarthDial(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id, device=device)
        _ensure_internvl_available()
        self.device = get_device(device)
        base_model = "OpenGVLab/InternVL2-8B"
        try:
            import json

            from huggingface_hub import hf_hub_download
            from transformers import AutoConfig

            base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
            try:
                config_path = hf_hub_download(repo_id=model_id, filename="config.json")
                with open(config_path) as f:
                    model_config_dict = json.load(f)
                model_config_dict["_name_or_path"] = base_model
                if "auto_map" in model_config_dict:
                    auto_map = model_config_dict["auto_map"]
                    if "AutoModelForCausalLM" in auto_map:
                        auto_map["AutoModelForCausalLM"] = (
                            f"{base_model}--{auto_map['AutoModelForCausalLM'].split('--')[-1]}"
                        )
                model_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
                for key, value in model_config_dict.items():
                    if key not in ["_name_or_path", "auto_map", "transformers_version"]:
                        setattr(model_config, key, value)
            except Exception:
                model_config = base_config
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=model_config,
                dtype=get_dtype(self.device),
                device_map=self.device,
                trust_remote_code=True,
            )
        except Exception as e:
            msg = f"Failed to load EarthDial model {model_id}. These models require custom InternVL code files from {base_model}. Error: {e}"
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
