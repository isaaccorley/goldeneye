from collections.abc import Iterator

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_xlrs_bench(
    split: str = "train", streaming: bool = False, cache_dir: str | None = None
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the XLRS-Bench-lite dataset.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset without downloading it entirely, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset
        The loaded dataset

    Examples
    --------
    >>> from goldeneye.datasets import load_xlrs_bench
    >>> # Load entire dataset (downloads to disk)
    >>> dataset = load_xlrs_bench(split="train", streaming=False)
    >>> # Stream dataset (no download required)
    >>> dataset = load_xlrs_bench(split="train", streaming=True)
    """
    return load_dataset(
        "initiacms/XLRS-Bench-lite",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def stream_xlrs_bench(split: str = "train") -> Iterator[dict]:
    """Stream the XLRS-Bench-lite dataset sample by sample.

    This function streams the dataset without downloading it entirely to disk.
    Each sample is downloaded on-demand as you iterate.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"

    Yields
    ------
    dict
        A single sample from the dataset

    Examples
    --------
    >>> from goldeneye.datasets import stream_xlrs_bench
    >>> # Stream samples one at a time
    >>> for sample in stream_xlrs_bench(split="train"):
    ...     image = sample["image"]
    ...     question = sample["question"]
    ...     # Process sample without loading entire dataset
    ...     break  # Process just first sample
    """
    dataset = load_xlrs_bench(split=split, streaming=True)
    yield from dataset
