from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    GenerationMixin,
)
from transformers.cache_utils import DynamicCache

from goldeneye.models.base import BaseAgent
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report


def _patch_phi3_model_class() -> None:
    """Patch Phi3 classes to fix issues with transformers >= 4.50.

    Issues fixed:
    1. Phi3Model.forward computes seq_length from input_ids only, ignoring inputs_embeds.
       This causes position_ids.view(-1, 0) when using inputs_embeds.
    2. Phi3ForCausalLM.prepare_inputs_for_generation doesn't properly handle DynamicCache
       when checking if past_key_values is empty.
    """
    import inspect
    import sys

    from transformers.cache_utils import Cache

    # Find the cached HuggingFace modeling_phi3 module
    phi3_module = None
    for name, module in sys.modules.items():
        if "modeling_phi3" in name and hasattr(module, "Phi3Model"):
            phi3_module = module
            break

    if phi3_module is None:
        return  # Module not loaded yet, will be patched later

    # Patch 1: Fix Phi3Model.forward for seq_length computation
    phi3_model_class = phi3_module.Phi3Model
    original_forward = phi3_model_class.forward

    # Check if original forward accepts cache_position
    sig = inspect.signature(original_forward)
    accepts_cache_position = "cache_position" in sig.parameters

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        # Fix: compute seq_length correctly when input_ids is empty tensor [1, 0]
        # InternVL2 passes empty input_ids alongside inputs_embeds which causes
        # the original code to compute seq_length=0 even when inputs_embeds exists
        if input_ids is not None and input_ids.shape[1] > 0:
            seq_length = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]
        else:
            seq_length = 0

        # Fix position_ids shape if needed
        if position_ids is not None and seq_length > 0 and position_ids.shape[-1] != seq_length:
            # Trim or regenerate position_ids to match seq_length
            if position_ids.shape[-1] > seq_length:
                position_ids = position_ids[:, -seq_length:]
            else:
                # Generate new position_ids
                device = inputs_embeds.device if inputs_embeds is not None else "cpu"
                position_ids = torch.arange(seq_length, device=device).unsqueeze(0)

        # Pass None instead of empty input_ids to force the original code
        # to use the inputs_embeds path
        if input_ids is not None and input_ids.shape[1] == 0:
            input_ids = None

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        if accepts_cache_position:
            kwargs["cache_position"] = cache_position

        return original_forward(self, **kwargs)

    phi3_model_class.forward = patched_forward

    # Patch 2: Fix Phi3ForCausalLM.prepare_inputs_for_generation for DynamicCache
    phi3_causal_lm_class = phi3_module.Phi3ForCausalLM
    original_prepare_inputs = phi3_causal_lm_class.prepare_inputs_for_generation

    def patched_prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Check if past_key_values is empty (works for both list/tuple and Cache objects)
        past_is_empty = past_key_values is None
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_is_empty = past_key_values.get_seq_length() == 0
            elif hasattr(past_key_values, "__len__"):
                past_is_empty = len(past_key_values) == 0

        # If inputs_embeds is provided and past is empty, use inputs_embeds
        # This is the first generation step for vision-language models
        if inputs_embeds is not None and past_is_empty:
            # Create position_ids for the full inputs_embeds sequence
            position_ids = kwargs.get("position_ids")
            if attention_mask is not None and position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            return {
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }

        # Otherwise use the original implementation
        return original_prepare_inputs(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    phi3_causal_lm_class.prepare_inputs_for_generation = patched_prepare_inputs_for_generation


def _get_usable_length(  # noqa: D417
    self: DynamicCache,
    new_seq_len: int,  # noqa: ARG001
    layer_idx: int = 0,
) -> int:
    """Return the usable cache length for a specific layer.

    The InternVL2 phi3 code adds this value to kv_seq_len, so we need to return
    the cache length for the specific layer, not the max across all layers.

    Parameters
    ----------
    self : DynamicCache
        The cache instance.
    new_seq_len : int
        The new sequence length (unused, kept for API compatibility).
    layer_idx : int, optional
        The layer index, by default 0.

    Returns
    -------
    int
        The usable cache length for the specified layer.
    """
    return self.get_seq_length(layer_idx)


# Patch DynamicCache for transformers >= 4.50 compatibility with InternVL2 code
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())  # type: ignore[attr-defined]
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: self.get_max_cache_shape()  # type: ignore[attr-defined]


INTERNVL2_BASE = "OpenGVLab/InternVL2-4B"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: set[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios_sorted = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, set(target_ratios_sorted), orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _load_image_for_internvl(
    image: Image.Image, input_size: int = 448, max_num: int = 12
) -> torch.Tensor:
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


class EarthDial(BaseAgent):
    def __init__(
        self,
        codename: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        quantization_config: BitsAndBytesConfig | None = None,
    ) -> None:
        """Initialize EarthDial model.

        Parameters
        ----------
        codename : str
            HuggingFace model ID
        device : str | None, optional
            Target device
        dtype : torch.dtype | None, optional
            Model dtype
        quantization_config : BitsAndBytesConfig | None, optional
            Quantization config for 4/8-bit loading
        """
        super().__init__(
            codename, device=device, dtype=dtype, quantization_config=quantization_config
        )
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)
        old_hf_transfer = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(INTERNVL2_BASE, trust_remote_code=True)
            load_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            else:
                load_kwargs["torch_dtype"] = self.dtype

            self.model = AutoModel.from_pretrained(INTERNVL2_BASE, **load_kwargs)

            # Fix for transformers >= 4.50: Phi3ForCausalLM doesn't inherit GenerationMixin
            lm = self.model.language_model
            if not hasattr(lm, "generate"):
                lm.__class__ = type(lm.__class__.__name__, (lm.__class__, GenerationMixin), {})
            if lm.generation_config is None:
                lm.generation_config = GenerationConfig()

            # Fix position_ids shape issue in Phi3Model with transformers >= 4.50
            _patch_phi3_model_class()

            # Load EarthDial weights and detect vocab size from checkpoint
            weight_files = [
                hf_hub_download(codename, f"model-0000{i}-of-00002.safetensors") for i in [1, 2]
            ]
            state_dict: dict[str, Any] = {}
            for wf in weight_files:
                state_dict.update(load_file(wf))

            # Detect vocab size from checkpoint (different variants have different sizes)
            # RGB/MS use 32035, Methane-UHI uses 32038
            embed_key = "language_model.model.embed_tokens.weight"
            if embed_key in state_dict:
                earthdial_vocab_size = state_dict[embed_key].shape[0]
            else:
                earthdial_vocab_size = 32035  # fallback to default

            self.model.language_model.resize_token_embeddings(earthdial_vocab_size)
            self.model.load_state_dict(state_dict, strict=False)
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
        """Run inference on an image.

        Parameters
        ----------
        image : str | Path | Image.Image
            Input image
        prompt : str, optional
            Text prompt
        max_new_tokens : int, optional
            Max tokens to generate

        Returns
        -------
        Report
            Model response
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        pixel_values = _load_image_for_internvl(pil_image, max_num=12)
        pixel_values = pixel_values.to(self.dtype).to(self.model.device)
        result = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config={"max_new_tokens": max_new_tokens, "do_sample": False},
        )
        # Handle both tuple (response, history) and plain string returns
        response = result[0] if isinstance(result, tuple) else result
        return Report(image=image, prompt=prompt, response=response)
