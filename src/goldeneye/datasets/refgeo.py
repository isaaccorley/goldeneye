"""RefGeo - Referring Expression Comprehension for Remote Sensing.

https://huggingface.co/datasets/erenzhou/refGeo

RefGeo contains remote sensing images with referring expressions
and bounding box/polygon annotations for grounding tasks.
"""

from typing import Any

from datasets import load_dataset


def load_refgeo(
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Any:
    """Load the RefGeo dataset.

    RefGeo contains remote sensing images with referring expressions
    and bounding box/polygon annotations for grounding tasks.

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
        Dataset with columns: question_id, image_id, bbox, poly, question

    Examples
    --------
    >>> from goldeneye.datasets import load_refgeo
    >>> dataset = load_refgeo(split="train")
    >>> sample = dataset[0]
    >>> bbox = sample["bbox"]
    >>> question = sample["question"]
    """
    return load_dataset(
        "erenzhou/refGeo",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
