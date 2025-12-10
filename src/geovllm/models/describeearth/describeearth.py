from __future__ import annotations

import copy
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        msg = f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        raise ValueError(msg)
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            import requests

            with requests.get(image, stream=True) as response:
                response.raise_for_status()
                with BytesIO(response.content) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                import base64

                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                with BytesIO(data) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        else:
            image_obj = Image.open(image)
    if image_obj is None:
        msg = f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        raise ValueError(msg)
    image = to_rgb(image_obj)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


def extract_vision_info_dam(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or "focal_crop" in ele
                        or ele.get("type", "") in ("image", "image_url", "video", "focal_crop")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info_dam(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[Image.Image] | None, list[Image.Image] | None]:
    vision_infos = extract_vision_info_dam(conversations)
    image_inputs = []
    focal_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "focal_crop" in vision_info:
            focal = vision_info["focal_crop"]
            if isinstance(focal, str):
                focal = Image.open(focal).convert("RGB")
            focal_inputs.append(focal)
        elif "video" in vision_info:
            msg = "Video processing not implemented in simplified version"
            raise ValueError(msg)
    if len(image_inputs) == 0:
        image_inputs = None
    if len(focal_inputs) == 0:
        focal_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs, focal_inputs


def crop_aabb_bbox(
    image: Image.Image,
    pts: np.ndarray,
    T_large: int = 224,
    T_small: int = 112,
) -> Image.Image:
    W, H = image.size
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    bw, bh = x_max - x_min, y_max - y_min
    if bw >= T_large or bh >= T_large:
        crop = image.crop((x_min, y_min, x_max, y_max))
    elif bw < T_small and bh < T_small:
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        half = T_small / 2
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        x0, y0 = min(x0, W - T_small), min(y0, H - T_small)
        crop = image.crop((x0, y0, x0 + T_small, y0 + T_small))
    else:
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        half = T_large / 2
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        x0, y0 = min(x0, W - T_large), min(y0, H - T_large)
        crop = image.crop((x0, y0, x0 + T_large, y0 + T_large))
    return crop.resize((224, 224), Image.LANCZOS)


class DescribeEarth(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id, device=device)
        self.device = get_device(device)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_id, trust_remote_code=True)
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
        self,
        image: str | Path | Image.Image,
        prompt: str,
        max_new_tokens: int = 64,
        bbox: Optional[np.ndarray] = None,
    ) -> str:
        pil_image = self._load_image(image)
        if bbox is not None:
            focal = crop_aabb_bbox(pil_image, bbox)
            category_text = ""
            prompt_text = (
                f"Please describe the object {category_text}in the bounding box in the original image <image>, "
                f"where the bounding box is defined by the coordinates: "
                f"(x_left_top: {bbox[0][0]:.1f}, y_left_top: {bbox[0][1]:.1f}, "
                f"x_right_top: {bbox[1][0]:.1f}, y_right_top: {bbox[1][1]:.1f}, "
                f"x_right_bottom: {bbox[2][0]:.1f}, y_right_bottom: {bbox[2][1]:.1f}, "
                f"x_left_bottom: {bbox[3][0]:.1f}, y_left_bottom: {bbox[3][1]:.1f}). "
                f"The corresponding cropped region is shown in the focal image <focal_crop>."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "focal_crop", "focal_crop": focal},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        else:
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
        image_inputs, video_inputs, focal_inputs = process_vision_info_dam(messages)
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if focal_inputs is not None:
            processor_kwargs["focal_crop"] = focal_inputs
        inputs = self.processor(**processor_kwargs).to(self.model.device)
        with torch.inference_mode():
            gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False)
        gen_ids_trim = [oid[len(iid) :] for iid, oid in zip(inputs.input_ids, gen_ids, strict=True)]
        return self.processor.batch_decode(
            gen_ids_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
