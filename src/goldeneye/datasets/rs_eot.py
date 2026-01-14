"""RS-EoT-4K - Remote Sensing Chain-of-Thought Reasoning Dataset.

https://huggingface.co/datasets/ShaoRun/RS-EoT-4K

RS-EoT-4K contains 4K remote sensing VQA samples with chain-of-thought
reasoning annotations. Each sample includes query, response with reasoning,
and corresponding image.
"""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


def load_rs_eot(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Any:
    """Load the RS-EoT-4K dataset.

    RS-EoT-4K contains remote sensing VQA samples with chain-of-thought
    reasoning annotations for improved interpretability.

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
        Dataset with columns: query, response, image

    Examples
    --------
    >>> from goldeneye.datasets import load_rs_eot
    >>> dataset = load_rs_eot(split="train")
    >>> sample = dataset[0]
    >>> query = sample["query"]
    >>> response = sample["response"]
    """
    return load_dataset(
        "ShaoRun/RS-EoT-4K",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def stream_rs_eot(split: str = "train") -> Iterator[dict]:
    """Stream the RS-EoT-4K dataset sample by sample.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"

    Yields
    ------
    dict
        A single sample with query, response, image
    """
    ds = load_rs_eot(split=split, streaming=True)
    yield from ds
