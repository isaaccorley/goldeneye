from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from geovllm.models.base import BaseGeoVLM
from geovllm.models.utils import get_device

_BPE_VOCAB_URL = (
    "https://huggingface.co/spaces/LanguageBind/LanguageBind/resolve/main/"
    "open_clip/bpe_simple_vocab_16e6.txt.gz"
)


def _ensure_bpe_vocab() -> None:
    """Ensure BPE vocabulary file exists, downloading if necessary."""
    import sam3.model_builder as mb

    bpe_path = os.path.join(
        os.path.dirname(mb.__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
    )
    bpe_path = os.path.normpath(bpe_path)

    if os.path.exists(bpe_path):
        return

    os.makedirs(os.path.dirname(bpe_path), exist_ok=True)

    try:
        import urllib.request

        print(f"Downloading BPE vocabulary file to {bpe_path}...")
        urllib.request.urlretrieve(_BPE_VOCAB_URL, bpe_path)
        print("✓ BPE vocabulary file downloaded successfully")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download BPE vocabulary file from {_BPE_VOCAB_URL}: {e}"
        ) from e


class SAM3(BaseGeoVLM):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        super().__init__(model_id, device=device)
        self.device = get_device(device)

        _ensure_bpe_vocab()
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        if self.device and "cuda" in str(self.device):
            self.model = self.model.to(self.device)

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
        _ = image, prompt, max_new_tokens
        return ""

    def referring_segmentation(
        self, image: str | Path | Image.Image, prompt: str
    ) -> tuple[list, list]:
        pil_image = self._load_image(image)
        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        scores = output["scores"]

        masks_list = masks.tolist() if masks is not None else []
        scores_list = scores.tolist() if scores is not None else []

        return masks_list, scores_list
