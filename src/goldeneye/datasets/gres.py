"""GRES - Generalized Referring Expression Segmentation Dataset.

https://huggingface.co/datasets/jquenum/GRES

GRES contains text expressions with segmentation masks for generalized
referring expression segmentation tasks.
"""

from typing import Any

from datasets import load_dataset


def load_gres(
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Any:
    """Load the GRES dataset.

    GRES contains text expressions with segmentation masks for generalized
    referring expression segmentation tasks.

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
        Dataset with columns: text, is_sentence, shapes

    Examples
    --------
    >>> from goldeneye.datasets import load_gres
    >>> dataset = load_gres(split="train")
    >>> sample = dataset[0]
    >>> text = sample["text"]
    """
    return load_dataset(
        "jquenum/GRES",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
