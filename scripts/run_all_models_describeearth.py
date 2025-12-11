from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image, ImageDraw, ImageFont

import geovllm
from geovllm.datasets import stream_de_dataset

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("geovllm.run_all_models")


class ModelResult(NamedTuple):
    model: str
    prompt: str
    response: str | None
    error: str | None


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def fetch_image(split: str = "train") -> Image.Image:
    for sample in stream_de_dataset(split=split):
        image = sample.get("image")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
    raise RuntimeError("No image found in DE-Dataset stream.")


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join((*current, word))
        width = font.getbbox(candidate)[2] - font.getbbox(candidate)[0]
        if width <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def load_fonts(size: int) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    font_paths = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        ("arial.ttf", "arialbd.ttf"),
    ]
    for regular_path, bold_path in font_paths:
        try:
            regular = ImageFont.truetype(regular_path, size)
            bold = ImageFont.truetype(bold_path, size)
            return regular, bold
        except OSError:
            continue
    default = ImageFont.load_default()
    return default, default


def run_model(
    model_name: str, image: Image.Image, prompt: str, device: str, max_new_tokens: int
) -> ModelResult:
    LOGGER.info("Loading %s", model_name)
    try:
        model = geovllm.load_model(model_name, device=device)
    except Exception as exc:  # noqa: BLE001
        return ModelResult(model_name, prompt, None, f"load_failed: {exc}")
    try:
        with torch.inference_mode():
            response = model(image, prompt, max_new_tokens=max_new_tokens)
        error = None
    except Exception as exc:  # noqa: BLE001
        response = None
        error = f"inference_failed: {exc}"
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ModelResult(model_name, prompt, response, error)


def run_models(
    image: Image.Image, prompt: str, device: str, max_new_tokens: int
) -> list[ModelResult]:
    return [
        run_model(model_name, image, prompt, device, max_new_tokens)
        for model_name in geovllm.list_models()
    ]


def build_lines(
    prompt: str,
    results: Iterable[ModelResult],
    font: ImageFont.ImageFont,
    bold: ImageFont.ImageFont,
    max_width: int,
) -> list[tuple[str, ImageFont.ImageFont]]:
    lines: list[tuple[str, ImageFont.ImageFont]] = []
    lines.extend([(line, bold) for line in wrap_text(f"Prompt: {prompt}", bold, max_width)])
    for result in results:
        text = result.response if result.error is None else result.error
        payload = f"{result.model}: {text}"
        lines.extend([(line, font) for line in wrap_text(payload, font, max_width)])
        lines.append(("", font))
    return lines


def render_panel(
    image: Image.Image,
    prompt: str,
    results: Iterable[ModelResult],
    output_path: Path,
    max_width: int = 1200,
) -> Path:
    target_width = min(max_width, image.width)
    target_height = int(image.height * (target_width / image.width))
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    regular_font, bold_font = load_fonts(18)
    margin = 20
    line_spacing = 8
    text_lines = build_lines(prompt, results, regular_font, bold_font, target_width)

    def line_height(entry: tuple[str, ImageFont.ImageFont]) -> int:
        text, font = entry
        return font.getbbox(text or " ")[3] - font.getbbox(text or " ")[1]

    text_height = sum(line_height(line) + line_spacing for line in text_lines)
    total_height = target_height + text_height + margin * 2 + line_spacing
    canvas = Image.new("RGB", (target_width + margin * 2, total_height), "white")
    canvas.paste(resized, (margin, margin))
    draw = ImageDraw.Draw(canvas)
    y = margin + target_height + line_spacing
    for text, font in text_lines:
        draw.text((margin, y), text, fill="black", font=font)
        y += line_height((text, font)) + line_spacing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main() -> None:
    device = select_device()
    prompt = "Describe the key objects, land cover, and context in this satellite image."
    image = fetch_image(split="train")
    LOGGER.info("Fetched DE-Dataset sample: %s", image.size)
    results = run_models(image, prompt, device, max_new_tokens=64)
    output_path = Path("/home/ubuntu/github/geovllm/all_models_describeearth.png")
    render_panel(image, prompt, results, output_path)
    successes = sum(1 for result in results if result.error is None)
    LOGGER.info("Completed %s/%s models. Output saved to %s", successes, len(results), output_path)


if __name__ == "__main__":
    main()
