"""Landsat Captions - Landsat Remote Sensing Captioning Dataset.

https://huggingface.co/datasets/Kornberg/landsat_captions

Landsat captions dataset for remote sensing image-caption pairs
from Landsat imagery with captions and segmentation annotations.
"""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


def load_landsat_captions(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Any:
    """Load the Landsat Captions dataset.

    Landsat captions dataset for remote sensing image-caption pairs
    from Landsat imagery with captions and segmentation annotations.

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
        Dataset with columns: input, target, captions, binary_captions, etc.

    Examples
    --------
    >>> from goldeneye.datasets import load_landsat_captions
    >>> dataset = load_landsat_captions(split="train")
    >>> sample = dataset[0]
    >>> captions = sample["captions"]
    """
    return load_dataset(
        "Kornberg/landsat_captions",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def stream_landsat_captions(split: str = "train") -> Iterator[dict]:
    """Stream the Landsat Captions dataset sample by sample.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"

    Yields
    ------
    dict
        A single sample with input, target, captions, etc.
    """
    ds = load_landsat_captions(split=split, streaming=True)
    yield from ds
