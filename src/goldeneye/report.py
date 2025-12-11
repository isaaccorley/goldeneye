from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class Report:
    image: str | Path | Image.Image
    prompt: str
    response: str
    image_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.image_size is None:
            if isinstance(self.image, Image.Image):
                self.image_size = self.image.size
            elif isinstance(self.image, (str, Path)):
                img = Image.open(self.image)
                self.image_size = img.size
                img.close()
