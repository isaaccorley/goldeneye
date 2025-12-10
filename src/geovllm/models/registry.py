from geovllm.models.base import BaseGeoVLM
from geovllm.models.describeearth import DescribeEarth
from geovllm.models.earthdial import EarthDial
from geovllm.models.earthgpt import EarthGPT
from geovllm.models.geochat import GeoChat
from geovllm.models.geollava import GeoLLaVA
from geovllm.models.geopixel import GeoPixel
from geovllm.models.geor1 import GeoR1
from geovllm.models.geozero import GeoZero
from geovllm.models.sam3 import SAM3
from geovllm.models.zoomearth import ZoomEarth

_MODEL_REGISTRY: dict[str, str] = {
    "GeoZero": "hjvsl/GeoZero",
    "GeoLLaVA-8K": "initiacms/GeoLLaVA-8K",
    "Geo-R1": "miniHui/Geo-R1",
    "Geo-R1-7B-GRPO-REC-10shot": "Geo-R1/Geo-R1-7B-GRPO-REC-10shot",
    "Geo-R1-3B-GRPO-REC-5shot": "Geo-R1/Geo-R1-3B-GRPO-REC-5shot",
    "Geo-R1-3B-GRPO-GRES-5shot": "Geo-R1/Geo-R1-3B-GRPO-GRES-5shot",
    "Geo-R1-3B-GRPO-OVD-5shot": "Geo-R1/Geo-R1-3B-GRPO-OVD-5shot",
    "Geo-R1-3B-GRPO-OVD-10shot": "Geo-R1/Geo-R1-3B-GRPO-OVD-10shot",
    "Geo-R1-3B-GRPO-REC-1shot": "Geo-R1/Geo-R1-3B-GRPO-REC-1shot",
    "Geo-R1-3B-GRPO-GRES-1shot": "Geo-R1/Geo-R1-3B-GRPO-GRES-1shot",
    "Geo-R1-3B-GRPO-GRES-10shot": "Geo-R1/Geo-R1-3B-GRPO-GRES-10shot",
    "Geo-R1-3B-GRPO-REC-10shot": "Geo-R1/Geo-R1-3B-GRPO-REC-10shot",
    "EarthGPT": "Pruz0/EarthGPT",
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

_MODEL_CLASSES: dict[str, type[BaseGeoVLM]] = {
    "GeoZero": GeoZero,
    "GeoLLaVA-8K": GeoLLaVA,
    "Geo-R1": GeoR1,
    "Geo-R1-7B-GRPO-REC-10shot": GeoR1,
    "Geo-R1-3B-GRPO-REC-5shot": GeoR1,
    "Geo-R1-3B-GRPO-GRES-5shot": GeoR1,
    "Geo-R1-3B-GRPO-OVD-5shot": GeoR1,
    "Geo-R1-3B-GRPO-OVD-10shot": GeoR1,
    "Geo-R1-3B-GRPO-REC-1shot": GeoR1,
    "Geo-R1-3B-GRPO-GRES-1shot": GeoR1,
    "Geo-R1-3B-GRPO-GRES-10shot": GeoR1,
    "Geo-R1-3B-GRPO-REC-10shot": GeoR1,
    "EarthGPT": EarthGPT,
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


def list_models() -> list[str]:
    return list(_MODEL_REGISTRY.keys())


def load_model(model_name: str, device: str | None = None) -> BaseGeoVLM:
    if model_name not in _MODEL_REGISTRY:
        msg = f"Model {model_name} not found. Available models: {list_models()}"
        raise ValueError(msg)

    hf_model_id = _MODEL_REGISTRY[model_name]
    model_class = _MODEL_CLASSES[model_name]
    return model_class(hf_model_id, device=device)
