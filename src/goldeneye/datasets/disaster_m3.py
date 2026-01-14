"""DisasterM3 - Multi-hazard Disaster Assessment Dataset.

https://huggingface.co/datasets/Kingdrone-Junjue/DisasterM3

DisasterM3 is a comprehensive VLM disaster assessment dataset with
27K bi-temporal satellite images and 123K instruction pairs across
5 continents, covering 36 historical disaster events in 10 categories.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_disaster_m3(
    split: str = "test",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the DisasterM3 dataset.

    DisasterM3 is a comprehensive VLM disaster assessment dataset with
    27K bi-temporal satellite images and 123K instruction pairs across
    5 continents, covering 36 historical disaster events in 10 categories.

    Features:
    - Multi-hazard: 10 natural and man-made disaster types
    - Multi-sensor: Optical and SAR imagery for post-disaster scenes
    - Multi-task: 9 disaster-related perception and reasoning tasks

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "test"
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
    >>> from goldeneye.datasets import load_disaster_m3
    >>> dataset = load_disaster_m3(split="test")
    >>> sample = dataset[0]
    """
    return load_dataset(
        "Kingdrone-Junjue/DisasterM3",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
