"""VRSBench - Versatile Remote Sensing Benchmark.

https://huggingface.co/datasets/xiang709/VRSBench

VRSBench contains 29.6K images with comprehensive annotations including
captions, object detections, referring expressions, and 123K QA pairs.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_vrsbench(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the VRSBench (Versatile Remote Sensing Benchmark) dataset.

    VRSBench contains 29.6K images with comprehensive annotations including
    captions, object detections, referring expressions, and 123K QA pairs.

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
        The loaded dataset with columns: caption, objects, qa_pairs, image

    Examples
    --------
    >>> from goldeneye.datasets import load_vrsbench
    >>> dataset = load_vrsbench(split="train")
    >>> sample = dataset[0]
    >>> caption = sample["caption"]
    >>> qa_pairs = sample["qa_pairs"]  # list of QA dicts
    """
    return load_dataset(
        "xiang709/VRSBench",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
