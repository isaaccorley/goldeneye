"""ChangeChat-87k - Change Detection Conversation Dataset.

https://huggingface.co/datasets/isaaccorley/ChangeChat-87k

ChangeChat-87k contains 87K bi-temporal image pairs with change detection
conversations for vision-language understanding.
"""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


def load_changechat(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Any:
    """Load the ChangeChat-87k dataset.

    ChangeChat-87k contains bi-temporal image pairs with change detection
    conversations for vision-language understanding.

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
        Dataset with columns: id, image, changeflag, conversations

    Examples
    --------
    >>> from goldeneye.datasets import load_changechat
    >>> dataset = load_changechat(split="train")
    >>> sample = dataset[0]
    >>> conversations = sample["conversations"]
    """
    return load_dataset(
        "isaaccorley/ChangeChat-87k",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def stream_changechat(split: str = "train") -> Iterator[dict]:
    """Stream the ChangeChat-87k dataset sample by sample.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"

    Yields
    ------
    dict
        A single sample with id, image, changeflag, conversations
    """
    ds = load_changechat(split=split, streaming=True)
    yield from ds
