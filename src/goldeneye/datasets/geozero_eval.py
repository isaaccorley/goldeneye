"""GeoZero Eval Datasets.

https://huggingface.co/datasets/hjvsl/GeoZero_Eval_Datasets

Evaluation dataset for the GeoZero model with conversation format
and associated images (~26K samples).
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_geozero_eval(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the GeoZero Eval dataset.

    Evaluation dataset for the GeoZero model with conversation format
    and associated images.

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
        The loaded dataset with columns: conversations, images

    Examples
    --------
    >>> from goldeneye.datasets import load_geozero_eval
    >>> dataset = load_geozero_eval(split="train")
    >>> sample = dataset[0]
    >>> conversations = sample["conversations"]
    >>> images = sample["images"]
    """
    return load_dataset(
        "hjvsl/GeoZero_Eval_Datasets",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
