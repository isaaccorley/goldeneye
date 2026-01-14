"""GeoText-1652 - Text-to-Image Geo-Localization Dataset.

https://huggingface.co/datasets/GeoText/GeoText-1652

GeoText-1652 contains 1652 geographic locations with satellite and
street-view images for text-to-image geo-localization research.
"""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


def load_geotext1652(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Any:
    """Load the GeoText-1652 dataset.

    GeoText-1652 contains geographic locations with satellite and
    street-view images for text-to-image geo-localization.

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
    >>> from goldeneye.datasets import load_geotext1652
    >>> dataset = load_geotext1652(split="train")
    >>> sample = dataset[0]
    """
    return load_dataset(
        "GeoText/GeoText-1652",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def stream_geotext1652(split: str = "train") -> Iterator[dict]:
    """Stream the GeoText-1652 dataset sample by sample.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"

    Yields
    ------
    dict
        A single sample from the dataset
    """
    ds = load_geotext1652(split=split, streaming=True)
    yield from ds
