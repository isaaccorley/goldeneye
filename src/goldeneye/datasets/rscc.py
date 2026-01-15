"""RSCC - Remote Sensing Change Caption Dataset.

https://huggingface.co/datasets/BiliSakura/RSCC

RSCC is a large-scale Remote Sensing Change Caption dataset containing
62.3K bi-temporal image pairs with change captions. Built from xBD and EBD
datasets for temporal image understanding.

Configs:
- "benchmark": Benchmark evaluation split (split="benchmark")
- "EBD": EBD (Emergency Building Damage) subset
"""

from typing import Any, Literal

from datasets import load_dataset


def load_rscc(
    config: Literal["benchmark", "EBD"] = "benchmark",
    split: str = "benchmark",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Any:
    """Load the RSCC (Remote Sensing Change Caption) dataset.

    RSCC contains 62.3K bi-temporal image pairs with change captions,
    designed for remote sensing change captioning and temporal understanding.

    Note: Due to xBD licenses, xBD images must be obtained separately from
    https://www.xview2.org/

    Parameters
    ----------
    config : RSCCConfig, optional
        Dataset configuration. Options: "benchmark", "EBD".
        By default "benchmark"
    split : str, optional
        Dataset split to load, by default "benchmark"
    streaming : bool, optional
        If True, stream the dataset without downloading, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Any
        Dataset with columns: pre_image, post_image, change_caption

    Examples
    --------
    >>> from goldeneye.datasets import load_rscc
    >>> dataset = load_rscc(config="benchmark", split="benchmark")
    >>> sample = next(iter(dataset))
    """
    return load_dataset(
        "BiliSakura/RSCC",
        config,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
