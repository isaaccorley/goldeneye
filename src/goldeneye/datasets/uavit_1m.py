"""UAVIT-1M - UAV Instruction Tuning Dataset.

https://huggingface.co/datasets/ZhanYang-nwpu/UAVIT-1M

UAVIT-1M is the largest UAV instruction tuning dataset with 1.24M
instructions covering 789K multi-scene low-altitude UAV images.
Supports 11 image-level and region-level tasks.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_uavit_1m(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the UAVIT-1M dataset.

    UAVIT-1M is the largest UAV instruction tuning dataset with 1.24M
    instructions covering 789K multi-scene low-altitude UAV images.
    Supports 11 image-level and region-level tasks.

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
    >>> from goldeneye.datasets import load_uavit_1m
    >>> dataset = load_uavit_1m(split="train")
    >>> sample = dataset[0]
    """
    return load_dataset(
        "ZhanYang-nwpu/UAVIT-1M",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
