"""Sydney-Captions - Sydney Aerial Image Captions.

https://huggingface.co/datasets/isaaccorley/Sydney-Captions

Sydney Aerial Image dataset contains 613 images captured over Sydney, Australia.
Note: The HuggingFace version contains images only (captions not included).
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_sydney_captions(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the Sydney-Captions dataset.

    Sydney Aerial Image dataset contains 613 images captured over Sydney,
    Australia. Note: The HuggingFace version contains images only
    (captions not included).

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
        The loaded dataset with columns: image (PIL Image)

    Examples
    --------
    >>> from goldeneye.datasets import load_sydney_captions
    >>> dataset = load_sydney_captions(split="train")
    >>> sample = dataset[0]
    >>> image = sample["image"]  # PIL Image
    """
    return load_dataset(
        "isaaccorley/Sydney-Captions",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
