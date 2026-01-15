"""GeoChat Bench - Evaluation Benchmark for GeoChat.

https://huggingface.co/datasets/MBZUAI/GeoChat-Bench

Evaluation benchmark for GeoChat, including LRBEN, HRBEN, AID, UCMerced,
and custom evaluation datasets for region captioning, visual grounding,
and grounding description tasks.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_geochat_bench(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the GeoChat Bench evaluation dataset.

    Evaluation benchmark for GeoChat, including LRBEN, HRBEN, AID, UCMerced,
    and custom evaluation datasets for region captioning, visual grounding,
    and grounding description tasks.

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
    >>> from goldeneye.datasets import load_geochat_bench
    >>> dataset = load_geochat_bench(split="train")
    """
    return load_dataset(
        "MBZUAI/GeoChat-Bench",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
