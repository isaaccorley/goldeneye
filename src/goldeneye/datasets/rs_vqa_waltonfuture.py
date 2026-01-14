"""Remote Sensing VQA (WaltonFuture).

https://huggingface.co/datasets/WaltonFuture/remote-sensing-VQA

A VQA dataset for remote sensing images with image-question-answer triplets.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_rs_vqa(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the Remote Sensing VQA dataset.

    A VQA dataset for remote sensing images with image-question-answer triplets.

    Parameters
    ----------
    split : str, optional
        Dataset split to load. Options: "train", "test", by default "train"
    streaming : bool, optional
        If True, stream the dataset, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset with columns: images (list), problem, answer

    Examples
    --------
    >>> from goldeneye.datasets import load_rs_vqa
    >>> dataset = load_rs_vqa(split="train")
    >>> sample = dataset[0]
    >>> images = sample["images"]  # list of PIL Images
    >>> problem = sample["problem"]  # question text
    >>> answer = sample["answer"]  # answer text
    """
    return load_dataset(
        "WaltonFuture/remote-sensing-VQA",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
