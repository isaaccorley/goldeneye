import re
from pathlib import Path

from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device, get_dtype


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
        padding="longest",  # type: ignore[arg-type]
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


class ZoomEarth(BaseGeoVLM):
    processor: Qwen2_5_VLProcessor
    model: Qwen2_5_VLForConditionalGeneration

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
        self, image: str | Path | Image.Image, prompt: str, max_new_tokens: int = 64
    ) -> str:
        pil_image = self._load_image(image)
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
            image_bbox = self._load_image(image)
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
            return output1.split("<answer>")[0] + output2
        return output1
