"""PreGRES - Pre-training for Generalized Referring Expression Segmentation.

https://huggingface.co/datasets/jquenum/PreGRES

PreGRES contains image-conversation pairs for pre-training referring
expression segmentation models.
"""

from typing import Any, Literal

from datasets import load_dataset


def load_pregres(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Any:
    """Load the PreGRES dataset.

    PreGRES contains image-conversation pairs for pre-training referring
    expression segmentation models.

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
    Any
        Dataset with columns: id, image, conversations

    Examples
    --------
    >>> from goldeneye.datasets import load_pregres
    >>> dataset = load_pregres(split="train")
    >>> sample = dataset[0]
    >>> conversations = sample["conversations"]
    """
    return load_dataset(
        "jquenum/PreGRES",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
