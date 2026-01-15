"""LRS-VQA - Large Remote Sensing VQA.

https://huggingface.co/datasets/ll-13/LRS-VQA

LRS-VQA contains 7.3K QA pairs across 1.6K high-resolution images
(1K to 27K pixels). Designed to evaluate LVLM perception on large RS images.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_lrs_vqa(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the LRS-VQA (Large Remote Sensing VQA) dataset.

    LRS-VQA contains 7.3K QA pairs across 1.6K high-resolution images
    (1K to 27K pixels). Designed to evaluate LVLM perception on large RS images.

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
        The loaded dataset with VQA columns

    Examples
    --------
    >>> from goldeneye.datasets import load_lrs_vqa
    >>> dataset = load_lrs_vqa(split="train")
    >>> sample = dataset[0]
    >>> question = sample["question"]
    >>> answer = sample["answer"]
    """
    return load_dataset(
        "ll-13/LRS-VQA",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
