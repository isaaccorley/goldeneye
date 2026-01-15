from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig

from goldeneye.models.base import BaseAgent
from goldeneye.models.describe_earth import DescribeEarth
from goldeneye.models.earthdial import EarthDial
from goldeneye.models.florence2_dota import Florence2DOTA
from goldeneye.models.geochat import GeoChat
from goldeneye.models.geollava import GeoLLaVA
from goldeneye.models.geor1 import GeoR1
from goldeneye.models.geozero import GeoZero
from goldeneye.models.rscovlm import RSCoVLM
from goldeneye.models.zoomearth import ZoomEarth

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
    "EarthDial": "akshaydudhane/EarthDial_4B_RGB",
    "GeoChat": "MBZUAI/geochat-7B",
    "ZoomEarth": "HappyBug/ZoomEarth-3B",
    "DescribeEarth": "earth-insights/DescribeEarth",
    # RSCoVLM variants (Qwen2.5-VL-7B based)
    "RSCoVLM-7B": "Qingyun/RSCoVLM-7B-2512",
    "RSCoVLM-det-7B": "Qingyun/RSCoVLM-det-7B-2512",
    # Florence2-DOTA (Florence-2 based)
    "Florence2-DOTA": "Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate",
    # GeoChat-based variants
    "GeoChat-UAV": "ZhanYang-nwpu/GeoChat-UAV",
    "SkySenseGPT": "ll-13/SkySenseGPT-7B-CLIP-ViT",
}

_AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "GeoZero": GeoZero,
    "GeoLLaVA-8K": GeoLLaVA,
    "Geo-R1-3B-GRPO-REC-5shot": GeoR1,
    "Geo-R1-3B-GRPO-GRES-5shot": GeoR1,
    "Geo-R1-3B-GRPO-OVD-5shot": GeoR1,
    "Geo-R1-3B-GRPO-OVD-10shot": GeoR1,
    "Geo-R1-3B-GRPO-REC-1shot": GeoR1,
    "Geo-R1-3B-GRPO-GRES-1shot": GeoR1,
    "Geo-R1-3B-GRPO-GRES-10shot": GeoR1,
    "Geo-R1-3B-GRPO-REC-10shot": GeoR1,
    "EarthDial": EarthDial,
    "GeoChat": GeoChat,
    "ZoomEarth": ZoomEarth,
    "DescribeEarth": DescribeEarth,
    # RSCoVLM variants (Qwen2.5-VL-7B based)
    "RSCoVLM-7B": RSCoVLM,
    "RSCoVLM-det-7B": RSCoVLM,
    # Florence2-DOTA (Florence-2 based)
    "Florence2-DOTA": Florence2DOTA,
    # GeoChat-based variants (reuse GeoChat class)
    "GeoChat-UAV": GeoChat,
    "SkySenseGPT": GeoChat,
}


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
    agent_class = _AGENT_CLASSES[codename]
    return agent_class(
        hf_model_id,
        device=device,
        dtype=dtype,
        quantization_config=quantization_config,
    )
