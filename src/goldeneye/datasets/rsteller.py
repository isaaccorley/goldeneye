"""RSTeller - Large-Scale Remote Sensing Image-Text Dataset.

https://huggingface.co/datasets/SlytherinGe/RSTeller

RSTeller is a large-scale remote sensing image-text dataset containing
1M+ image-text pairs for vision-language model training.
"""

import ast
from typing import Any, Literal

from datasets import load_dataset


def _parse_json_column(example: dict) -> dict:
    """Parse the json column from string to dict.

    Parameters
    ----------
    example : dict
        A single example from the dataset

    Returns
    -------
    dict
        The example with parsed metadata
    """
    if "json" in example and example["json"] is not None:
        if isinstance(example["json"], str):
            try:
                example["metadata"] = ast.literal_eval(example["json"])
            except (ValueError, SyntaxError):
                example["metadata"] = None
        else:
            example["metadata"] = example["json"]
    return example


def load_rsteller(
    split: Literal["train"] = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Any:
    """Load the RSTeller dataset.

    RSTeller contains 1M+ remote sensing image-text pairs
    for vision-language model training.

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
        Dataset with columns: __key__, __url__, json, jpg, metadata
        The metadata dict contains 'annotations' (list of captions) and
        'metadata' (image info like patch_id and image_name).

    Examples
    --------
    >>> from goldeneye.datasets import load_rsteller
    >>> dataset = load_rsteller(split="train", streaming=True)
    >>> sample = next(iter(dataset))
    >>> annotations = sample["metadata"]["annotations"]
    >>> caption = annotations[0]["text"]
    """
    ds = load_dataset(
        "SlytherinGe/RSTeller",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
    return ds.map(_parse_json_column)
