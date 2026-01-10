from __future__ import annotations

from pathlib import Path
from typing import Any

import supervision as sv
from PIL import Image, ImageDraw, ImageFont

from goldeneye.report import Report


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _wrap_text(text: str, max_width: int, font: Any) -> list[str]:
    """Wrap text to fit within max_width pixels.

    Parameters
    ----------
    text : str
        The text string to wrap
    max_width : int
        Maximum width in pixels for each line
    font : Any
        PIL font object used to measure text width

    Returns
    -------
    list[str]
        List of wrapped text lines
    """
    words = text.split()
    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = font.getbbox(test_line)
        if bbox[2] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines if lines else [""]


def render(
    report: Report,
    show_confidence: bool = True,
    show_caption: bool = True,
    reticle_color: str = "#FFD700",
    max_size: int = 800,
    caption_padding: int = 20,
    font_size: int = 14,
) -> Image.Image:
    """Render annotated image with optional bounding boxes and caption.

    Parameters
    ----------
    report : Report
        The report containing image, prompt, and response
    show_confidence : bool, optional
        Whether to show confidence scores on detections, by default True
    show_caption : bool, optional
        Whether to show the response caption below the image, by default True
    reticle_color : str, optional
        Hex color for bounding boxes, by default "#FFD700" (gold)
    max_size : int, optional
        Maximum dimension for the output image, by default 800
    caption_padding : int, optional
        Padding around the caption text, by default 20
    font_size : int, optional
        Font size for the caption text, by default 14

    Returns
    -------
    Image.Image
        Annotated image with optional caption
    """
    image = _load_image(report.image)
    resolution_wh = report.image_size or image.size

    detections = sv.Detections.from_vlm(
        vlm=sv.VLM.QWEN_3_VL,
        result=report.response,
        resolution_wh=resolution_wh,
    )

    annotated_image = image.copy()
    annotated_image = annotate_image(
        image=annotated_image,
        detections=detections,
        show_confidence=show_confidence,
        reticle_color=reticle_color,
    )
    annotated_image.thumbnail((max_size, max_size))

    if not show_caption or not report.response:
        return annotated_image

    # Add caption below the image
    font = ImageFont.load_default(size=font_size)

    # Wrap text to fit image width
    text_max_width = annotated_image.width - (2 * caption_padding)
    wrapped_lines = _wrap_text(report.response, text_max_width, font)

    # Limit to reasonable number of lines
    max_lines = 8
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
        wrapped_lines[-1] = wrapped_lines[-1][:50] + "..."

    # Calculate caption area height
    line_height = font_size + 4
    caption_height = len(wrapped_lines) * line_height + (2 * caption_padding)

    # Create new image with space for caption
    new_height = annotated_image.height + caption_height
    result = Image.new("RGB", (annotated_image.width, new_height), color=(255, 255, 255))
    result.paste(annotated_image, (0, 0))

    # Draw caption text
    draw = ImageDraw.Draw(result)
    y_offset = annotated_image.height + caption_padding
    for line in wrapped_lines:
        draw.text((caption_padding, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += line_height

    return result


def annotate_image(
    image: Image.Image,
    detections: sv.Detections,
    show_confidence: bool = True,
    reticle_color: str = "#FFD700",
) -> Image.Image:
    import numpy as np

    scene = np.array(image)
    box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex(reticle_color))
    scene = box_annotator.annotate(scene=scene, detections=detections)

    if show_confidence and len(detections) > 0:
        labels = []
        for i in range(len(detections)):
            conf = detections.confidence
            confidence = float(conf[i]) if conf is not None else 0.0
            class_name = (
                detections.data.get("class_name", [None] * len(detections))[i]
                if detections.data
                else None
            )
            if class_name:
                labels.append(f"{class_name} {confidence:.0%}")
            else:
                labels.append(f"{confidence:.0%} Match")
        label_annotator = sv.LabelAnnotator()
        scene = label_annotator.annotate(scene=scene, detections=detections, labels=labels)

    return Image.fromarray(scene)
