"""RSTeller - Large-Scale Remote Sensing Image-Text Dataset.

https://huggingface.co/datasets/SlytherinGe/RSTeller

RSTeller is a large-scale remote sensing image-text dataset containing
1M+ image-text pairs for vision-language model training.
"""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


def load_rsteller(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Any:
    """Load the RSTeller dataset.

    RSTeller contains 1M+ remote sensing image-text pairs
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
        The loaded dataset with image-text pairs

    Examples
    --------
    >>> from goldeneye.datasets import load_rsteller
    >>> dataset = load_rsteller(split="train", streaming=True)
    >>> sample = next(iter(dataset))
    """
    return load_dataset(
        "SlytherinGe/RSTeller",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def stream_rsteller(split: str = "train") -> Iterator[dict]:
    """Stream the RSTeller dataset sample by sample.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"

    Yields
    ------
    dict
        A single sample from the dataset
    """
    ds = load_rsteller(split=split, streaming=True)
    yield from ds
