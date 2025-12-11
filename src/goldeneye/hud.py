from __future__ import annotations

from pathlib import Path

import supervision as sv
from PIL import Image

from goldeneye.report import Report


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def render(
    report: Report,
    show_confidence: bool = True,
    reticle_color: str = "gold",
) -> Image.Image:
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
    annotated_image.thumbnail((800, 800))
    return annotated_image


def annotate_image(
    image: Image.Image,
    detections: sv.Detections,
    show_confidence: bool = True,
    reticle_color: str = "gold",
) -> Image.Image:
    import numpy as np

    scene = np.array(image)
    box_annotator = sv.BoxAnnotator(color=reticle_color)
    scene = box_annotator.annotate(scene=scene, detections=detections)

    if show_confidence and len(detections) > 0:
        labels = []
        for i in range(len(detections)):
            confidence = (
                float(detections.confidence[i]) if detections.confidence is not None else 0.0
            )
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
