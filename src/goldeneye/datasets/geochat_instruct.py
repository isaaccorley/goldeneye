from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download, snapshot_download

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

REPO_ID = "MBZUAI/GeoChat_Instruct"
JSON_FILENAME = "GeoChat_Instruct.json"


class GeoChatInstructDataset:
    def __init__(
        self,
        json_path: str | Path | None = None,
        image_dir: str | Path | None = None,
        download_json: bool = True,
    ) -> None:
        self._data: list[dict] | None = None
        self.image_dir = Path(image_dir) if image_dir else None

        if json_path:
            self.json_path = Path(json_path)
        elif download_json:
            self.json_path = Path(
                hf_hub_download(repo_id=REPO_ID, filename=JSON_FILENAME, repo_type="dataset")
            )
        else:
            raise ValueError("Either json_path or download_json=True required")

    @property
    def data(self) -> list[dict]:
        if self._data is None:
            with open(self.json_path) as f:
                self._data = json.load(f)
        return self._data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def __iter__(self) -> Iterator[dict]:
        yield from self.data

    def get_image_path(self, idx: int) -> Path | None:
        if self.image_dir is None:
            return None
        abs_path = self.image_dir / self.data[idx]["image"]
        return abs_path if abs_path.exists() else None

    def load_image(self, idx: int) -> Image.Image:
        from PIL import Image

        path = self.get_image_path(idx)
        if path is None:
            raise FileNotFoundError(f"Image not found for sample {idx}")
        return Image.open(path).convert("RGB")

    def get_prompt(self, idx: int) -> str:
        for turn in self.data[idx]["conversations"]:
            if turn["from"] == "human":
                return turn["value"].replace("<image>\n", "").strip()
        return ""

    def get_response(self, idx: int) -> str:
        for turn in self.data[idx]["conversations"]:
            if turn["from"] == "gpt":
                return turn["value"]
        return ""

    @staticmethod
    def download_images(output_dir: str | Path, cache_dir: str | Path | None = None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=["images_part*"],
                cache_dir=cache_dir,
            )
        )

        part_files = sorted(snapshot_path.glob("images_part*"))
        if not part_files:
            raise FileNotFoundError(f"No image parts found in {snapshot_path}")

        cat_cmd = ["cat", *[str(p) for p in part_files]]
        tar_cmd = ["tar", "-xf", "-", "-C", str(output_dir)]

        with (
            subprocess.Popen(cat_cmd, stdout=subprocess.PIPE) as cat_proc,
            subprocess.Popen(tar_cmd, stdin=cat_proc.stdout) as tar_proc,
        ):
            if cat_proc.stdout:
                cat_proc.stdout.close()
            tar_proc.communicate()

        if tar_proc.returncode != 0:
            raise RuntimeError(f"tar extraction failed with code {tar_proc.returncode}")

        return output_dir

    def stats(self) -> dict:
        from collections import Counter

        sources = Counter()
        for item in self.data:
            parts = item["image"].split("/")
            sources[parts[0] if len(parts) > 1 else "root"] += 1

        return {"total_samples": len(self.data), "sources": dict(sources)}


def stream_geochat_instruct(
    json_path: str | Path | None = None, chunk_size: int = 1000
) -> Iterator[list[dict]]:
    if json_path is None:
        json_path = Path(
            hf_hub_download(repo_id=REPO_ID, filename=JSON_FILENAME, repo_type="dataset")
        )

    with open(json_path) as f:
        data = json.load(f)

    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]
