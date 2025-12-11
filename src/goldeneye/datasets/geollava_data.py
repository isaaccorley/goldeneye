"""GeoLLaVA-Data dataset loader.

GeoLLaVA-Data is a remote sensing visual question answering dataset containing:
- SuperRS-VQA (avg. 8376x8378 resolution)
- HighRS-VQA (avg. 2000x1912 resolution)

Total: 81,367 ultra-high-resolution image-text pairs for supervised fine-tuning.

Reference: https://huggingface.co/datasets/initiacms/GeoLLaVA-Data
Paper: https://arxiv.org/abs/2505.21375
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterator

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

DATASET_ID = "initiacms/GeoLLaVA-Data"


class ConversationTurn(TypedDict):
    """A single turn in a conversation."""

    from_: str  # "human" or "gpt"
    value: str


class GeoLLaVASample(TypedDict):
    """A sample from the GeoLLaVA-Data dataset."""

    image: list[str]  # List of image paths
    conversations: list[ConversationTurn]  # Conversation turns
    id: int  # Sample ID


def load_geollava_data(
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the GeoLLaVA-Data dataset.

    GeoLLaVA-Data contains ultra-high-resolution remote sensing VQA data
    with multiple-choice questions, object detection, spatial reasoning,
    and image description tasks.

    Parameters
    ----------
    split : str, optional
        Dataset split to load, by default "train"
    streaming : bool, optional
        If True, stream the dataset without downloading it entirely, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset

    Examples
    --------
    >>> from goldeneye.datasets import load_geollava_data
    >>> # Load entire dataset (downloads to disk)
    >>> dataset = load_geollava_data(split="train", streaming=False)
    >>> # Stream dataset (no download required)
    >>> dataset = load_geollava_data(split="train", streaming=True)
    >>> # Access a sample
    >>> sample = dataset[0]
    >>> print(sample["image"])  # List of image paths
    >>> print(sample["conversations"])  # Conversation turns
    """
    return load_dataset(
        DATASET_ID,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def stream_geollava_data(split: str = "train") -> Iterator[dict]:
    """Stream the GeoLLaVA-Data dataset sample by sample.

    This function streams the dataset without downloading it entirely to disk.
    Each sample is downloaded on-demand as you iterate.

    Parameters
    ----------
    split : str, optional
        Dataset split to stream, by default "train"

    Yields
    ------
    dict
        A single sample from the dataset containing:
        - image: list[str] - List of image paths
        - conversations: list[dict] - Conversation turns with "from" and "value" keys
        - id: int - Sample ID

    Examples
    --------
    >>> from goldeneye.datasets import stream_geollava_data
    >>> # Stream samples one at a time
    >>> for sample in stream_geollava_data(split="train"):
    ...     image_paths = sample["image"]
    ...     conversations = sample["conversations"]
    ...     # Extract question and answer
    ...     question = conversations[0]["value"]  # Human turn
    ...     answer = conversations[1]["value"]  # GPT turn
    ...     break  # Process just first sample
    """
    dataset = load_geollava_data(split=split, streaming=True)
    yield from dataset


def parse_conversation(sample: dict) -> tuple[str, str, list[str]]:
    """Parse a GeoLLaVA-Data sample into question, answer, and image paths.

    Parameters
    ----------
    sample : dict
        A sample from the GeoLLaVA-Data dataset

    Returns
    -------
    tuple[str, str, list[str]]
        A tuple of (question, answer, image_paths)

    Examples
    --------
    >>> from goldeneye.datasets import load_geollava_data, parse_conversation
    >>> dataset = load_geollava_data(split="train", streaming=True)
    >>> sample = next(iter(dataset))
    >>> question, answer, image_paths = parse_conversation(sample)
    """
    conversations = sample["conversations"]
    image_paths = sample["image"]

    question = ""
    answer = ""

    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human":
            question = value
        elif role == "gpt":
            answer = value

    return question, answer, image_paths
