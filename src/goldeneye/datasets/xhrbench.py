"""XHRBench - Ultra-High-Resolution RS Benchmark.

https://huggingface.co/datasets/FelixKAI/XHRBench

XHRBench evaluates MLLMs on ultra-high-resolution RS images (4K to 300MP).
Includes MCQ, open-ended VQA, and image captioning tasks across
9 perception categories and 4 reasoning types.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_xhrbench(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the XHRBench (Ultra-High-Resolution RS Benchmark) dataset.

    XHRBench evaluates MLLMs on ultra-high-resolution RS images (4K to 300MP).
    Includes MCQ, open-ended VQA, and image captioning tasks across
    9 perception categories and 4 reasoning types.

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
        The loaded dataset

    Examples
    --------
    >>> from goldeneye.datasets import load_xhrbench
    >>> dataset = load_xhrbench(split="train")
    >>> sample = dataset[0]
    >>> question = sample["question"]
    """
    return load_dataset(
        "FelixKAI/XHRBench",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
