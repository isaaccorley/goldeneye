"""EarthDial-Dataset - Remote Sensing Evaluation Dataset.

https://huggingface.co/datasets/akshaydudhane/EarthDial-Dataset

EarthDial-Dataset is a curated collection of evaluation-only datasets
for remote sensing and Earth observation downstream tasks.
Supports multiple task configs: Classification, GeoChat_Bench, Detection,
Region_captioning, Image_captioning.

Note: Each config has different splits. Use streaming to discover splits.
"""

from collections.abc import Iterator
from typing import Any, Literal

from datasets import load_dataset

EarthDialConfig = Literal[
    "Classification",
    "GeoChat_Bench",
    "Detection",
    "Region_captioning",
    "Image_captioning",
]


def load_earthdial_dataset(
    config: EarthDialConfig = "Image_captioning",
    split: str | None = None,
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Any:
    """Load the EarthDial-Dataset.

    EarthDial-Dataset contains evaluation-only datasets for remote sensing
    and Earth observation tasks. All datasets are for inference/testing only.

    Parameters
    ----------
    config : EarthDialConfig, optional
        Task configuration to load. Options:
        - "Classification": AID_sample, UCM_sample, BigEarthNet, WHU
        - "GeoChat_Bench": GeoChat benchmark evaluation
        - "Detection": Object detection evaluation
        - "Region_captioning": Region captioning evaluation
        - "Image_captioning": sydney_Captions, UCM_Captions, RSICD
        By default "Image_captioning"
    split : str | None, optional
        Dataset split to load, by default None (returns all splits)
    streaming : bool, optional
        If True, stream the dataset without downloading, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Any
        The loaded dataset with task-specific columns

    Examples
    --------
    >>> from goldeneye.datasets import load_earthdial_dataset
    >>> dataset = load_earthdial_dataset(config="Image_captioning",
    ...                                   split="RSICD_Captions")
    >>> sample = next(iter(dataset))
    """
    return load_dataset(
        "akshaydudhane/EarthDial-Dataset",
        config,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def stream_earthdial_dataset(
    config: EarthDialConfig = "Image_captioning",
    split: str | None = None,
) -> Iterator[dict]:
    """Stream the EarthDial-Dataset sample by sample.

    Parameters
    ----------
    config : EarthDialConfig, optional
        Task configuration to load, by default "Image_captioning"
    split : str | None, optional
        Dataset split to stream, by default None

    Yields
    ------
    dict
        A single sample from the dataset
    """
    ds = load_earthdial_dataset(config=config, split=split, streaming=True)
    yield from ds
