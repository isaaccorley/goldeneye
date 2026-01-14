"""GeoPixInstruct - Pixel-level RS Instruction Dataset.

https://huggingface.co/datasets/Norman-ou/GeoPixInstruct-Anno

GeoPixInstruct contains 58K pixel-level RS instructions across 6 splits
for training models on referring segmentation and related tasks.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_geopix_instruct(
    split: str = "train.SIOR_T",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the GeoPixInstruct dataset.

    GeoPixInstruct contains 58K pixel-level RS instructions across 6 splits
    for training models on referring segmentation and related tasks.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train.SIOR_T"
        Available splits: train.SIOR_T, train.RRSIS_T, etc.
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
    >>> from goldeneye.datasets import load_geopix_instruct
    >>> dataset = load_geopix_instruct(split="train.SIOR_T")
    >>> sample = dataset[0]
    """
    return load_dataset(
        "Norman-ou/GeoPixInstruct-Anno",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
