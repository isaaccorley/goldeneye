from typing import Literal

import torch
from transformers import BitsAndBytesConfig


def get_device(device: str | None = None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str | None, dtype: torch.dtype | None = None) -> torch.dtype:
    if dtype is not None:
        return dtype
    if device == "mps":
        return torch.float16
    return torch.bfloat16


def create_quantization_config(bits: Literal[4, 8]) -> BitsAndBytesConfig:
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
