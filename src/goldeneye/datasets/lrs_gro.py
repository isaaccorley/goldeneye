"""LRS-GRO - Large RS Grounding Dataset.

https://huggingface.co/datasets/HappyBug/LRS-GRO

LRS-GRO combines scene-level VQA with object grounding through
bounding box annotations.
"""

from __future__ import annotations

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_lrs_gro(
    split: Literal["test", "rl", "sft"] = "test",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the LRS-GRO (Large RS Grounding) dataset.

    LRS-GRO combines scene-level VQA with object grounding through
    bounding box annotations.

    Parameters
    ----------
    split : {"test", "rl", "sft"}, optional
        Dataset split to load, by default "test"
    streaming : bool, optional
        If True, stream the dataset, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset with columns: question, ground_truth, bbox, category

    Examples
    --------
    >>> from goldeneye.datasets import load_lrs_gro
    >>> dataset = load_lrs_gro(split="test")
    >>> sample = dataset[0]
    >>> question = sample["question"]
    >>> bbox = sample["bbox"]
    """
    return load_dataset(
        "HappyBug/LRS-GRO",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
