"""UCM-Captions - UC Merced Land Use Captions.

https://huggingface.co/datasets/cpratikaki/UCMcaptions_finetuning

UC Merced Land Use Captions contains images from the UC Merced Land Use dataset
with human-generated captions. Originally 2,100 images (21 classes x 100 images)
with 5 captions each, expanded to 10,500 samples.
"""

from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_ucm_captions(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the UCM-Captions dataset.

    UC Merced Land Use Captions contains images from the UC Merced Land Use
    dataset with human-generated captions. Originally 2,100 images
    (21 classes x 100 images) with 5 captions each, expanded to 10,500 samples.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset without downloading, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset with columns: image, caption

    Examples
    --------
    >>> from goldeneye.datasets import load_ucm_captions
    >>> dataset = load_ucm_captions(split="train")
    >>> sample = dataset[0]
    >>> image = sample["image"]
    >>> caption = sample["caption"]
    """
    return load_dataset(
        "cpratikaki/UCMcaptions_finetuning",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
