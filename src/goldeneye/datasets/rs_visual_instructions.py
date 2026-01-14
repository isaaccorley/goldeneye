"""AdaptLLM Remote Sensing Visual Instructions.

https://huggingface.co/datasets/AdaptLLM/remote-sensing-visual-instructions

A large-scale visual instruction tuning dataset (36.4K samples) combining
image captioning and synthetic VQA tasks from multiple RS captioning datasets:
NWPU-Captions, RSICD, RSITMD, Sydney-captions, and UCM-captions.
"""

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_rs_visual_instructions(
    config: str = "image_caption_and_synthetic_task",
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load the AdaptLLM Remote Sensing Visual Instructions dataset.

    A large-scale visual instruction tuning dataset (36.4K samples) combining
    image captioning and synthetic VQA tasks from multiple RS captioning
    datasets: NWPU-Captions, RSICD, RSITMD, Sydney-captions, and UCM-captions.

    Parameters
    ----------
    config : str, optional
        Dataset configuration, by default "image_caption_and_synthetic_task"
    split : str, optional
        Dataset split to load, by default "train"
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
    >>> from goldeneye.datasets import load_rs_visual_instructions
    >>> dataset = load_rs_visual_instructions(split="train")
    >>> for sample in dataset:
    ...     # Process visual instruction sample
    ...     break
    """
    return load_dataset(
        "AdaptLLM/remote-sensing-visual-instructions",
        name=config,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
