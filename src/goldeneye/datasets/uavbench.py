"""UAVBench - UAV Vision-Language Benchmark.

https://huggingface.co/datasets/ZhanYang-nwpu/UAVBench

UAVBench is a comprehensive benchmark for evaluating MLLMs on
low-altitude UAV vision-language tasks with 966K samples across
43 test units and 10 tasks.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_uavbench(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the UAVBench dataset.

    UAVBench is a comprehensive benchmark for evaluating MLLMs on
    low-altitude UAV vision-language tasks with 966K samples across
    43 test units and 10 tasks.

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
    >>> from goldeneye.datasets import load_uavbench
    >>> dataset = load_uavbench(split="train")
    >>> sample = dataset[0]
    """
    return load_dataset(
        "ZhanYang-nwpu/UAVBench",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
