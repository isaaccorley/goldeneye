"""RSICD - Remote Sensing Image Captioning Dataset.

https://huggingface.co/datasets/arampacha/rsicd

RSICD contains 10,921 remote sensing images with 5 captions per image.
Images cover various land-use types and scenes.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_rsicd(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the RSICD (Remote Sensing Image Captioning Dataset).

    RSICD contains 10,921 remote sensing images with 5 captions per image.
    Images cover various land-use types and scenes.

    Parameters
    ----------
    split : str, optional
        Dataset split to load. Options: "train", "valid", "test", by default "train"
    streaming : bool, optional
        If True, stream the dataset without downloading, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset with columns: image, filename, captions

    Examples
    --------
    >>> from goldeneye.datasets import load_rsicd
    >>> dataset = load_rsicd(split="train")
    >>> sample = dataset[0]
    >>> image = sample["image"]  # PIL Image
    >>> captions = sample["captions"]  # list of 5 captions
    """
    return load_dataset(
        "arampacha/rsicd",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
