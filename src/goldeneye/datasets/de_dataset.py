"""DE-Dataset - DescribeEarth Dataset.

https://huggingface.co/datasets/earth-insights/DE-Dataset

The DE-Dataset is a large-scale dataset with 25 categories and 261,806 annotated
instances, providing detailed descriptions of object attributes, relationships,
and contexts for remote sensing images.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_de_dataset(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the DE-Dataset (DescribeEarth Dataset).

    The DE-Dataset is a large-scale dataset with 25 categories and 261,806
    annotated instances, providing detailed descriptions of object attributes,
    relationships, and contexts for remote sensing images.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset without downloading it entirely. Note: webdataset
        format may require downloading the tar.gz file locally for non-streaming mode,
        by default True
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset

    Examples
    --------
    >>> from goldeneye.datasets import load_de_dataset
    >>> # Stream dataset (preferred - no full download required)
    >>> dataset = load_de_dataset(split="train", streaming=True)
    >>> # Load entire dataset (downloads tar.gz file)
    >>> dataset = load_de_dataset(split="train", streaming=False)
    """
    return load_dataset(
        "earth-insights/DE-Dataset",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
