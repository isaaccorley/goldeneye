"""XLRS-Bench-lite - Cross-Linguistic Remote Sensing Benchmark.

https://huggingface.co/datasets/initiacms/XLRS-Bench-lite

A benchmark for evaluating VLMs on remote sensing images with
multilingual questions and answers.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_xlrs_bench(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the XLRS-Bench-lite dataset.

    A benchmark for evaluating VLMs on remote sensing images with
    multilingual questions and answers.

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
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
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
