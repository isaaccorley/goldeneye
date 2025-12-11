from collections.abc import Iterator
from io import BytesIO
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from PIL import Image


def load_de_dataset(
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the DE-Dataset (DescribeEarth Dataset).

    The DE-Dataset is a large-scale dataset with 25 categories and 261,806 annotated
    instances, providing detailed descriptions of object attributes, relationships,
    and contexts for remote sensing images.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset without downloading it entirely. Note: webdataset format
        may require downloading the tar.gz file locally for non-streaming mode, by default True
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
    repo_id = "earth-insights/DE-Dataset"
    if streaming:
        return load_dataset(
            repo_id,
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        )
    return load_dataset(
        repo_id,
        split=split,
        streaming=False,
        cache_dir=cache_dir,
    )


def stream_de_dataset(
    split: str = "train", cache_dir: str | None = None
) -> Iterator[dict[str, Any]]:
    """Stream the DE-Dataset sample by sample.

    This function streams the dataset without downloading it entirely to disk.
    Each sample is fetched on-demand as you iterate.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Yields
    ------
    dict[str, Any]
        A single sample from the dataset in webdataset format with keys:
        - jpg: PIL.Image - The remote sensing image
        - __key__: str - Unique identifier/key for the sample
        - __url__: str - URL to the source tar.gz file
        Note: The dataset is in webdataset format. For descriptions and annotations,
        you may need to load additional metadata files separately.

    Examples
    --------
    >>> from goldeneye.datasets import stream_de_dataset
    >>> # Stream samples one at a time
    >>> for sample in stream_de_dataset(split="train"):
    ...     image = sample.get("jpg")
    ...     key = sample.get("__key__", "")
    ...     # Process sample without loading entire dataset
    ...     break  # Process just first sample
    """
    dataset = load_de_dataset(split=split, streaming=True, cache_dir=cache_dir)
    for sample_raw in dataset:
        if not isinstance(sample_raw, dict):
            continue
        sample = dict(sample_raw.items())
        if "jpg" in sample:
            img = sample["jpg"]
            if isinstance(img, bytes):
                sample["image"] = Image.open(BytesIO(img)).convert("RGB")
            elif isinstance(img, Image.Image):
                sample["image"] = img.convert("RGB")
            else:
                sample["image"] = img
        yield sample
