"""EVAttrs-95K - EV Charging Station Attributes Dataset.

https://huggingface.co/datasets/liarzone/EVAttrs-95K

EVAttrs-95K contains 95K street-view images with EV charging station
attribute annotations for vision-language tasks.
"""

from typing import Any

from datasets import load_dataset


def load_evattrs(
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Any:
    """Load the EVAttrs-95K dataset.

    EVAttrs-95K contains street-view images with EV charging station
    attribute annotations.

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
        Dataset with columns: img_id, objs

    Examples
    --------
    >>> from goldeneye.datasets import load_evattrs
    >>> dataset = load_evattrs(split="train")
    >>> sample = dataset[0]
    """
    return load_dataset(
        "liarzone/EVAttrs-95K",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
