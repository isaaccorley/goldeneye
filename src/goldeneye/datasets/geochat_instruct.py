"""GeoChat Instruct - Instruction Tuning Dataset for Remote Sensing.

https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct

A 318K instruction tuning dataset for remote sensing, used to train the
GeoChat model. Combines LRBEN, NWPU_captions, SOTA, SIOR, and FAST datasets.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_geochat_instruct(
    split: Literal["train"] = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the GeoChat Instruct dataset.

    A 318K instruction tuning dataset for remote sensing, used to train the
    GeoChat model. Combines LRBEN, NWPU_captions, SOTA, SIOR, and FAST datasets.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset

    Examples
    --------
    >>> from goldeneye.datasets import load_geochat_instruct
    >>> dataset = load_geochat_instruct(split="train")
    """
    return load_dataset(
        "MBZUAI/GeoChat_Instruct",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
