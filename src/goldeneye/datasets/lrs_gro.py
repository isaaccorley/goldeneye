import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image


def _extract_tar_images(
    repo_id: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    stream: bool = False,
) -> Path:
    """Extract images from multipart tar files.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID
    cache_dir : str | None, optional
        Directory to cache extracted images, by default None
    revision : str | None, optional
        Git revision to use, by default None
    stream : bool, optional
        If True, stream tar parts instead of downloading fully, by default False

    Returns
    -------
    Path
        Path to directory containing extracted images
    """
    if cache_dir is None:
        from huggingface_hub import constants

        cache_dir = str(constants.HF_HUB_CACHE)
    cache_path = Path(cache_dir) / "datasets" / repo_id.replace("/", "_")
    images_dir = cache_path / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        return images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    part_suffixes = [
        "aa",
        "ab",
        "ac",
        "ad",
        "ae",
        "af",
        "ag",
        "ah",
        "ai",
        "aj",
        "ak",
        "al",
        "am",
        "an",
        "ao",
        "ap",
        "aq",
        "ar",
        "as",
        "at",
        "au",
        "av",
        "aw",
        "ax",
        "ay",
        "az",
    ]
    if stream:
        import shutil

        combined_tar_path = cache_path / "images_combined.tar"
        if not combined_tar_path.exists() or combined_tar_path.stat().st_size == 0:
            if combined_tar_path.exists():
                combined_tar_path.unlink()
            print("Downloading and combining tar parts...")
            total_parts = 0
            with open(combined_tar_path, "wb") as combined_tar:
                for suffix in part_suffixes:
                    try:
                        print(f"  Downloading part {suffix}...", end=" ", flush=True)
                        tar_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=f"images.tar.part-{suffix}",
                            cache_dir=str(cache_path),
                            revision=revision,
                            force_download=False,
                            repo_type="dataset",
                        )
                        part_size = Path(tar_path).stat().st_size
                        with open(tar_path, "rb") as part_file:
                            shutil.copyfileobj(part_file, combined_tar, length=8192)
                        total_parts += 1
                        print(f"✓ ({part_size / 1024 / 1024:.1f} MB)")
                    except Exception:
                        if suffix == "aa":
                            raise
                        print("✗ (stopped)")
                        break
            combined_size_mb = combined_tar_path.stat().st_size / 1024 / 1024
            print(f"Combined {total_parts} parts into {combined_size_mb:.1f} MB")
        if combined_tar_path.exists() and combined_tar_path.stat().st_size > 0:
            if not any(images_dir.iterdir()):
                print("Extracting images from tar file...")
                with tarfile.open(combined_tar_path, "r") as tar:
                    tar.extractall(images_dir)
                print(f"✓ Extracted images to {images_dir}")
            else:
                print(
                    f"✓ Images already extracted ({len(list(images_dir.rglob('*.*')))} files found)"
                )
        else:
            msg = "Failed to download tar parts or combined tar is empty"
            raise RuntimeError(msg)
    else:
        for suffix in part_suffixes:
            try:
                tar_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"images.tar.part-{suffix}",
                    cache_dir=str(cache_path),
                    revision=revision,
                )
                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(images_dir)
            except Exception:
                if suffix == "aa":
                    raise
                break
    return images_dir


def load_lrs_gro(
    split: str | None = None,
    streaming: bool = True,
    cache_dir: str | None = None,
    extract_images: bool = True,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the LRS-GRO dataset.

    Parameters
    ----------
    split : str | None, optional
        Dataset split to load (e.g., "test", "sft", "rl").
        If None, loads all splits, by default None
    streaming : bool, optional
        If True, stream the dataset without downloading it entirely, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None
    extract_images : bool, optional
        If True, extract images from tar files. If False, only load JSONL metadata, by default True

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset

    Examples
    --------
    >>> from goldeneye.datasets import load_lrs_gro
    >>> # Load test split
    >>> dataset = load_lrs_gro(split="test", streaming=False)
    >>> # Load all splits
    >>> dataset = load_lrs_gro(streaming=False)
    >>> # Stream without downloading
    >>> dataset = load_lrs_gro(split="test", streaming=True)
    """
    repo_id = "HappyBug/LRS-GRO"
    if streaming:
        return load_dataset(
            repo_id,
            data_files="*.jsonl",
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        )
    if extract_images:
        images_dir = _extract_tar_images(repo_id, cache_dir=cache_dir, stream=True)
    else:
        images_dir = None
    dataset = load_dataset(
        repo_id,
        data_files="*.jsonl",
        split=split,
        streaming=False,
        cache_dir=cache_dir,
    )
    if images_dir is not None and isinstance(dataset, (Dataset, DatasetDict)):
        dataset = _add_images_to_dataset(dataset, images_dir)
    return dataset


def _add_images_to_dataset(
    dataset: Dataset | DatasetDict, images_dir: Path
) -> Dataset | DatasetDict:
    """Add image paths to dataset.

    Parameters
    ----------
    dataset : Dataset | DatasetDict
        Dataset to add images to
    images_dir : Path
        Directory containing extracted images

    Returns
    -------
    Dataset | DatasetDict
        Dataset with image paths added
    """
    if isinstance(dataset, DatasetDict):
        result_dict = {k: _add_images_to_dataset(v, images_dir) for k, v in dataset.items()}
        return DatasetDict(result_dict)  # type: ignore[arg-type]

    def _load_image(example: dict[str, Any]) -> dict[str, Any]:
        image_name = example.get("image_name", "")
        if image_name:
            image_path = images_dir / image_name
            if image_path.exists():
                example["image"] = str(image_path)
        return example

    return dataset.map(_load_image)


def stream_lrs_gro(
    split: str = "test", extract_images: bool = True, cache_dir: str | None = None
) -> Iterator[dict[str, Any]]:
    """Stream the LRS-GRO dataset sample by sample.

    This function streams the dataset without downloading it entirely to disk.
    Each sample is downloaded on-demand as you iterate.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "test"
    extract_images : bool, optional
        If True, extract images from tar files when needed, by default True
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Yields
    ------
    dict
        A single sample from the dataset with keys:
        - question_id: str
        - image_name: str
        - image: PIL.Image (if extract_images=True)
        - category: str
        - question: str
        - ground_truth: str
        - type: str (global/object/regional)
        - label: str | None
        - cut: bool
        - bbox: list[int] | None
        - split: str

    Examples
    --------
    >>> from goldeneye.datasets import stream_lrs_gro
    >>> # Stream samples one at a time
    >>> for sample in stream_lrs_gro(split="test"):
    ...     image = sample.get("image")
    ...     question = sample["question"]
    ...     # Process sample without loading entire dataset
    ...     break  # Process just first sample
    """
    repo_id = "HappyBug/LRS-GRO"
    images_dir = None
    if extract_images:
        images_dir = _extract_tar_images(repo_id, cache_dir=cache_dir, stream=True)
    jsonl_file = hf_hub_download(
        repo_id=repo_id,
        filename=f"data/{split}-00000-of-00001.jsonl.jsonl",
        cache_dir=cache_dir,
        repo_type="dataset",
    )
    with open(jsonl_file) as f:
        for line in f:
            sample = json.loads(line)
            if images_dir is not None:
                image_name = sample.get("image_name", "")
                if image_name:
                    image_path = images_dir / "image" / image_name
                    if not image_path.exists():
                        image_path = images_dir / image_name
                    if image_path.exists():
                        sample["image"] = Image.open(image_path).convert("RGB")
            yield sample
