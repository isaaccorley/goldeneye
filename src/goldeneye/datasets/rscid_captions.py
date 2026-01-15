"""RSCID-Captions - Remote Sensing Image Caption and VQA Dataset.

https://huggingface.co/datasets/mthandazo/rscid_captions_vqa_dataset

RSCID-Captions contains remote sensing images with captions,
formatted for VQA and captioning tasks.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_rscid_captions(
    split: Literal["train", "test"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the RSCID-Captions dataset.

    RSCID-Captions contains remote sensing images with captions,
    formatted for VQA and captioning tasks.

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
        The loaded dataset with columns: image, caption

    Examples
    --------
    >>> from goldeneye.datasets import load_rscid_captions
    >>> dataset = load_rscid_captions(split="train")
    >>> sample = dataset[0]
    >>> image = sample["image"]
    >>> caption = sample["caption"]
    """
    return load_dataset(
        "mthandazo/rscid_captions_vqa_dataset",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
