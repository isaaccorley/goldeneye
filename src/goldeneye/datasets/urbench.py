"""UrBench - Urban Scene Understanding Benchmark.

https://huggingface.co/datasets/bczhou/UrBench

UrBench is a comprehensive benchmark for urban scene understanding
with multiple task configurations including scene recognition,
object grounding, counting, and more.
"""

from typing import Any, Literal

from datasets import load_dataset

UrBenchConfig = Literal[
    "camera-localization",
    "city-retrieval",
    "counting",
    "image-retrieval",
    "object-attribute-recognition",
    "object-grounding",
    "object-matching",
    "orientation",
    "road-understanding",
    "role-based-reasoning",
    "scene-comparison",
    "scene-recognition",
    "traffic-sign-reasoning",
    "visual-prompt-reasoning",
]


def load_urbench(
    config: UrBenchConfig = "scene-recognition",
    split: str = "test",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Any:
    """Load the UrBench dataset.

    UrBench is a comprehensive benchmark for urban scene understanding
    with multiple task configurations.

    Parameters
    ----------
    config : UrBenchConfig, optional
        Task configuration to load, by default "scene-recognition"
    split : str, optional
        Dataset split to load, by default "test"
    streaming : bool, optional
        If True, stream the dataset without downloading, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None

    Returns
    -------
    Any
        Dataset with task-specific columns including metric, task_type, etc.

    Examples
    --------
    >>> from goldeneye.datasets import load_urbench
    >>> dataset = load_urbench(config="scene-recognition", split="test")
    >>> sample = dataset[0]
    """
    return load_dataset(
        "bczhou/UrBench",
        config,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
