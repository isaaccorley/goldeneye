from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig

from goldeneye.models.base import BaseAgent

_AGENT_REGISTRY: dict[str, str] = {
    "GeoZero": "hjvsl/GeoZero",
    "GeoLLaVA-8K": "initiacms/GeoLLaVA-8K",
    "Geo-R1-3B-GRPO-REC-5shot": "Geo-R1/Geo-R1-3B-GRPO-REC-5shot",
    "Geo-R1-3B-GRPO-GRES-5shot": "Geo-R1/Geo-R1-3B-GRPO-GRES-5shot",
    "Geo-R1-3B-GRPO-OVD-5shot": "Geo-R1/Geo-R1-3B-GRPO-OVD-5shot",
    "Geo-R1-3B-GRPO-OVD-10shot": "Geo-R1/Geo-R1-3B-GRPO-OVD-10shot",
    "Geo-R1-3B-GRPO-REC-1shot": "Geo-R1/Geo-R1-3B-GRPO-REC-1shot",
    "Geo-R1-3B-GRPO-GRES-1shot": "Geo-R1/Geo-R1-3B-GRPO-GRES-1shot",
    "Geo-R1-3B-GRPO-GRES-10shot": "Geo-R1/Geo-R1-3B-GRPO-GRES-10shot",
    "Geo-R1-3B-GRPO-REC-10shot": "Geo-R1/Geo-R1-3B-GRPO-REC-10shot",
    "EarthDial-4B-RGB": "akshaydudhane/EarthDial_4B_RGB",
    "EarthDial-4B-MS": "akshaydudhane/EarthDial_4B_MS",
    "EarthDial-4B-Methane-UHI": "akshaydudhane/EarthDial_4B_Methane_UHI",
    "geochat-7B": "MBZUAI/geochat-7B",
    "ZoomEarth-3B": "HappyBug/ZoomEarth-3B",
    "DescribeEarth": "earth-insights/DescribeEarth",
}

_AGENT_CLASS_PATHS: dict[str, tuple[str, str]] = {
    "GeoZero": ("goldeneye.models.geozero", "GeoZero"),
    "GeoLLaVA-8K": ("goldeneye.models.geollava", "GeoLLaVA"),
    "Geo-R1-3B-GRPO-REC-5shot": ("goldeneye.models.geor1", "GeoR1"),
    "Geo-R1-3B-GRPO-GRES-5shot": ("goldeneye.models.geor1", "GeoR1"),
    "Geo-R1-3B-GRPO-OVD-5shot": ("goldeneye.models.geor1", "GeoR1"),
    "Geo-R1-3B-GRPO-OVD-10shot": ("goldeneye.models.geor1", "GeoR1"),
    "Geo-R1-3B-GRPO-REC-1shot": ("goldeneye.models.geor1", "GeoR1"),
    "Geo-R1-3B-GRPO-GRES-1shot": ("goldeneye.models.geor1", "GeoR1"),
    "Geo-R1-3B-GRPO-GRES-10shot": ("goldeneye.models.geor1", "GeoR1"),
    "Geo-R1-3B-GRPO-REC-10shot": ("goldeneye.models.geor1", "GeoR1"),
    "EarthDial-4B-RGB": ("goldeneye.models.earthdial", "EarthDial"),
    "EarthDial-4B-MS": ("goldeneye.models.earthdial", "EarthDial"),
    "EarthDial-4B-Methane-UHI": ("goldeneye.models.earthdial", "EarthDial"),
    "geochat-7B": ("goldeneye.models.geochat", "GeoChat"),
    "ZoomEarth-3B": ("goldeneye.models.zoomearth", "ZoomEarth"),
    "DescribeEarth": ("goldeneye.models.describe_earth", "DescribeEarth"),
}


def _get_agent_class(codename: str) -> type[BaseAgent]:
    import importlib

    module_path, class_name = _AGENT_CLASS_PATHS[codename]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def assets() -> list[str]:
    return list(_AGENT_REGISTRY.keys())


def dispatch_agent(
    codename: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    quantization_config: BitsAndBytesConfig | None = None,
) -> BaseAgent:
    """Dispatch a geospatial VLM agent by codename.

    Parameters
    ----------
    codename : str
        Model identifier from assets()
    device : str | None, optional
        Target device ('cuda', 'cpu', 'mps'), by default None (auto-detect)
    dtype : torch.dtype | None, optional
        Model dtype (torch.float16, torch.bfloat16, etc.), by default None
    quantization_config : BitsAndBytesConfig | None, optional
        Quantization config for 4-bit or 8-bit loading, by default None

    Returns
    -------
    BaseAgent
        Loaded model agent ready for inference
    """
    if codename not in _AGENT_REGISTRY:
        msg = f"Model {codename} not found. Available models: {assets()}"
        raise ValueError(msg)

    hf_model_id = _AGENT_REGISTRY[codename]
    agent_class = _get_agent_class(codename)
    return agent_class(
        hf_model_id, device=device, dtype=dtype, quantization_config=quantization_config
    )
