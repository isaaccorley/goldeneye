from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

from goldeneye.models.base import BaseAgent
from goldeneye.models.utils import get_device, get_dtype
from goldeneye.report import Report


def chat_batch(
    prompts: list[str],
    imgs: list[Image.Image],
    processor: Qwen2_5_VLProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    max_new_tokens: int = 1024,
) -> list[str]:
    inputs = processor(
        text=prompts,
        images=imgs,
        return_tensors="pt",
        padding="longest",
    ).to(model.device)
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    outputs = []
    input_lens = inputs["input_ids"].shape[1]
    gen_ids_trimmed = gen_ids[:, input_lens:]
    for i in range(len(gen_ids_trimmed)):
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            msg = "Processor does not have tokenizer attribute"
            raise AttributeError(msg)
        outputs.append(tokenizer.decode(gen_ids_trimmed[i], skip_special_tokens=True).strip())
    return outputs


def cut_image(
    image: Image.Image, bbox: list[int] | list[float], min_size: int = 512
) -> Image.Image:
    x1, y1, x2, y2 = map(int, bbox)
    width, height = x2 - x1, y2 - y1
    if width < min_size or height < min_size:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        new_x1 = center_x - min_size // 2
        new_y1 = center_y - min_size // 2
        new_x2 = new_x1 + min_size
        new_y2 = new_y1 + min_size
        if new_x1 < 0:
            new_x2 += -new_x1
            new_x1 = 0
        if new_y1 < 0:
            new_y2 += -new_y1
            new_y1 = 0
        if new_x2 > image.width:
            new_x1 -= new_x2 - image.width
            new_x2 = image.width
        if new_y2 > image.height:
            new_y1 -= new_y2 - image.height
            new_y2 = image.height
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(image.width, new_x1 + min_size)
        new_y2 = min(image.height, new_y1 + min_size)
        return image.crop((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
    else:
        cropped = image.crop((x1, y1, x2, y2))
        return cropped


def extract_bbox(completion_content: str, scale: float) -> list[list[float]]:
    pattern = r'"bbox_2d"\s*:\s*\[(.*?)\]'
    matches = re.findall(pattern, completion_content, re.DOTALL)
    bboxes = []
    for m in matches:
        try:
            nums = [float(x.strip()) for x in m.split(",")]
            bbox = [num * scale for num in nums]
            bboxes.append(bbox)
        except ValueError:
            continue
    return bboxes


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return image


PREFIX = """<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>"""

INSTRUCTION = """
You are an intelligent remote sensing analyst. Given a natural language question about a satellite image, generate a structured reasoning answer as follows:

1. <think> ... </think>
   - Provide a neutral one-sentence description of the whole image scene.
   - Cropping task: "This question is asking about <short intent>, therefore I need to crop the image to examine the surroundings of the mentioned target."
   - Non-cropping task: "This question is asking about <short intent>, therefore I need to analyze the entire image without cropping."
   - Include:
     * Question Intent: describe the type of question (object category, spatial relation, count, etc.) and needed visual info.
     * Localization Strategy:
       - Cropping: approximate referent object location in natural language (no coordinates).
       - Non-cropping: strategy to detect all relevant objects.
     * Reasoning Result:
       - Cropping: output exactly one JSON-formatted bbox for the referent: [{"bbox_2d": [x_min,y_min,x_max,y_max], "label": "<short description>"}]
       - Non-cropping: summarize how detected objects will be used to produce the count.

2. <think> ... </think> (only when saw the cropped image)
   - Explain how to reason step by step from the referent (or detected objects) to the final answer.

3. <answer> ... </answer>
   - Your final answer, use a single word or phrase.

Rules:
- Always return exactly one <answer> block, for tasks that need cropping, you can provide the bounding box of the object you are interested, after given the cropped image, you can generate another <think> block to find the answer.
- For cropping tasks, also include a bounding box in <stage_2_reasoning> block
- If unsure about localization, make a best guess—never say uncertain.
<|im_end|><|im_start|>assistant
"""  # noqa: E501

DEFAULT_PROCESSOR = "Qwen/Qwen2.5-VL-3B-Instruct"


class ZoomEarth(BaseAgent):
    processor: Qwen2_5_VLProcessor
    model: Qwen2_5_VLForConditionalGeneration

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
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            DEFAULT_PROCESSOR, trust_remote_code=True
        )
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["dtype"] = self.dtype
            load_kwargs["device_map"] = self.device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(codename, **load_kwargs)
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
        scale = max(1, max(pil_image.width, pil_image.height) / 1024)
        resized_image = resize_image(pil_image)
        full_prompt = PREFIX + prompt + INSTRUCTION
        output1 = chat_batch(
            [full_prompt],
            [resized_image],
            self.processor,
            self.model,
            max_new_tokens=max_new_tokens,
        )[0]
        bboxs = extract_bbox(output1, scale)
        if bboxs:
            bbox_float = bboxs[0]
            bbox = [int(x) for x in bbox_float]
            if isinstance(image, (str, Path)):
                image_bbox = Image.open(image).convert("RGB")
            else:
                image_bbox = image.convert("RGB")
            image_bbox = resize_image(cut_image(image_bbox, bbox))
            new_prompt = (
                PREFIX
                + prompt
                + INSTRUCTION
                + output1.split("<answer>")[0]
                + "<|vision_start|><|image_pad|><|vision_end|>"
            )
            output2 = chat_batch(
                [new_prompt],
                [resized_image, image_bbox],
                self.processor,
                self.model,
                max_new_tokens=max_new_tokens,
            )[0]
            response = output1.split("<answer>")[0] + output2
        else:
            response = output1
        return Report(image=image, prompt=prompt, response=response)
