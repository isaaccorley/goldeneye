"""NWPU-Captions - NWPU-RESISC45 Image Captions.

https://huggingface.co/datasets/KhangTruong/NWPU-Caption

NWPU-Captions contains images from the NWPU-RESISC45 scene classification
dataset. 45 scene classes with ~700 images each.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_nwpu_captions(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the NWPU-Captions dataset.

    NWPU-Captions contains images from the NWPU-RESISC45 scene classification
    dataset. 45 scene classes with ~700 images each.

    Note: This dataset is in webdataset format with keys: jpg, __key__, __url__.
    Streaming mode is recommended.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset, by default True (recommended for webdataset)
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset with keys: jpg (image bytes), __key__, __url__

    Examples
    --------
    >>> from goldeneye.datasets import load_nwpu_captions
    >>> dataset = load_nwpu_captions(split="train", streaming=True)
    >>> for sample in dataset:
    ...     jpg_bytes = sample["jpg"]  # Image bytes
    ...     key = sample["__key__"]    # Unique identifier
    ...     break
    """
    return load_dataset(
        "KhangTruong/NWPU-Caption",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
