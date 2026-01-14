"""SARLANG-1M - SAR Image-Language Dataset.

https://huggingface.co/datasets/YiminJimmy/SARLANG-1M

SARLANG-1M contains 1M+ SAR (Synthetic Aperture Radar) images
for vision-language model training.
"""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


def load_sarlang(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Any:
    """Load the SARLANG-1M dataset.

    SARLANG-1M contains 1M+ SAR (Synthetic Aperture Radar) images
    for vision-language model training.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset without downloading, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Any
        Dataset with columns: image

    Examples
    --------
    >>> from goldeneye.datasets import load_sarlang
    >>> dataset = load_sarlang(split="train", streaming=True)
    >>> sample = next(iter(dataset))
    """
    return load_dataset(
        "YiminJimmy/SARLANG-1M",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def stream_sarlang(split: str = "train") -> Iterator[dict]:
    """Stream the SARLANG-1M dataset sample by sample.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"

    Yields
    ------
    dict
        A single sample with image
    """
    ds = load_sarlang(split=split, streaming=True)
    yield from ds
