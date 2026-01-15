"""VHM VersaD - Versatile RS Pre-training Dataset.

https://huggingface.co/datasets/FitzPC/VHM_VersaD

VHM VersaD is a large-scale remote sensing image-text dataset with
4.1M samples for pre-training versatile RS VLMs. Used as pre-training
data for VHM and MF-RSVLM models.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_vhm_versad(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the VHM VersaD dataset.

    VHM VersaD is a large-scale remote sensing image-text dataset with
    4.1M samples for pre-training versatile RS VLMs. Used as pre-training
    data for VHM and MF-RSVLM models.

    Note: This is a very large dataset. Streaming mode is strongly recommended.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset (recommended), by default True
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset with webdataset format (jpg, __key__, __url__)

    Examples
    --------
    >>> from goldeneye.datasets import load_vhm_versad
    >>> dataset = load_vhm_versad(split="train", streaming=True)
    >>> for sample in dataset:
    ...     jpg_bytes = sample["jpg"]
    ...     key = sample["__key__"]
    ...     break
    """
    return load_dataset(
        "FitzPC/VHM_VersaD",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
