"""RS5M - Large-scale Remote Sensing Image-Text Dataset.

https://huggingface.co/datasets/omlab/RS5M

RS5M is a large-scale remote sensing image-text dataset with 7.25M samples.
Suitable for CLIP-style pre-training or large-scale fine-tuning.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_rs5m(
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the RS5M dataset.

    RS5M is a large-scale remote sensing image-text dataset with 7.25M samples.
    Suitable for CLIP-style pre-training or large-scale fine-tuning.

    Note: This is a very large dataset. Streaming mode is strongly recommended.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset (recommended), by default True
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset with keys: caption, img_content, img_name

    Examples
    --------
    >>> from goldeneye.datasets import load_rs5m
    >>> dataset = load_rs5m(split="train", streaming=True)
    >>> for sample in dataset:
    ...     caption = sample["caption"]
    ...     img_bytes = sample["img_content"]
    ...     break
    """
    return load_dataset(
        "omlab/RS5M",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
