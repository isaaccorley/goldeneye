"""FAIR1M Caption - Fine-grained Object Recognition Captions.

https://huggingface.co/datasets/blanchon/FAIR1M_Small_Caption

FAIR1M Caption contains 22K remote sensing images with captions describing
fine-grained objects like aircraft, ships, vehicles, etc.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_fair1m_caption(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the FAIR1M Caption dataset.

    FAIR1M Caption contains 22K remote sensing images with captions describing
    fine-grained objects like aircraft, ships, vehicles, etc.

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
        The loaded dataset with columns: image, text

    Examples
    --------
    >>> from goldeneye.datasets import load_fair1m_caption
    >>> dataset = load_fair1m_caption(split="train")
    >>> sample = dataset[0]
    >>> image = sample["image"]
    >>> caption = sample["text"]
    """
    return load_dataset(
        "blanchon/FAIR1M_Small_Caption",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
