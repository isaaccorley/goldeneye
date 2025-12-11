import torch

from goldeneye.models.base import BaseAgent
from goldeneye.models.describe_earth import DescribeEarth
from goldeneye.models.earthdial import EarthDial
from goldeneye.models.geochat import GeoChat
from goldeneye.models.geollava import GeoLLaVA
from goldeneye.models.geopixel import GeoPixel
from goldeneye.models.geor1 import GeoR1
from goldeneye.models.geozero import GeoZero
from goldeneye.models.sam3 import SAM3
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
    "EarthDial-4B-RGB": "akshaydudhane/EarthDial_4B_RGB",
    "EarthDial-4B-MS": "akshaydudhane/EarthDial_4B_MS",
    "EarthDial-4B-Methane-UHI": "akshaydudhane/EarthDial_4B_Methane_UHI",
    "geochat-7B": "MBZUAI/geochat-7B",
    "GeoPixel-7B-RES": "MBZUAI/GeoPixel-7B-RES",
    "GeoPixel-7B": "MBZUAI/GeoPixel-7B",
    "SAM3": "facebook/sam3",
    "ZoomEarth-3B": "HappyBug/ZoomEarth-3B",
    "DescribeEarth": "earth-insights/DescribeEarth",
}

_AGENT_CLASSES: dict[str, type[BaseAgent] | None] = {
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
    "EarthDial-4B-RGB": EarthDial,
    "EarthDial-4B-MS": EarthDial,
    "EarthDial-4B-Methane-UHI": EarthDial,
    "geochat-7B": GeoChat,
    "GeoPixel-7B-RES": GeoPixel,
    "GeoPixel-7B": GeoPixel,
    "SAM3": SAM3,
    "ZoomEarth-3B": ZoomEarth,
    "DescribeEarth": DescribeEarth,
}


def assets() -> list[str]:
    return list(_AGENT_REGISTRY.keys())


def dispatch_agent(
    codename: str, device: str | None = None, dtype: torch.dtype | None = None
) -> BaseAgent:
    if codename not in _AGENT_REGISTRY:
        msg = f"Agent {codename} not found. Available agents: {assets()}"
        raise ValueError(msg)

    hf_model_id = _AGENT_REGISTRY[codename]
    agent_class = _AGENT_CLASSES[codename]
    if agent_class is None:
        msg = f"Agent {codename} is not supported"
        raise ImportError(msg)
    return agent_class(hf_model_id, device=device, dtype=dtype)
