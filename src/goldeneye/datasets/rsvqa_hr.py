"""RSVQA-HR (High Resolution) - High Resolution Remote Sensing VQA.

https://huggingface.co/datasets/cpratikaki/RSVQA-HR_qwen_finetuning

High-resolution remote sensing VQA dataset formatted for Qwen fine-tuning.
Contains 358K image-question-answer samples from various RS scenes.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_rsvqa_hr(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the RSVQA-HR (High Resolution) dataset.

    High-resolution remote sensing VQA dataset formatted for Qwen fine-tuning.
    Contains 358K image-question-answer samples from various RS scenes.

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
        The loaded dataset with columns: image, messages

    Examples
    --------
    >>> from goldeneye.datasets import load_rsvqa_hr
    >>> dataset = load_rsvqa_hr(split="train")
    >>> sample = dataset[0]
    >>> image = sample["image"]
    >>> messages = sample["messages"]  # Chat format
    """
    return load_dataset(
        "cpratikaki/RSVQA-HR_qwen_finetuning",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
