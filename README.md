# goldeneye

[![PyPI version](https://badge.fury.io/py/goldeneye.svg)](https://badge.fury.io/py/goldeneye)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple unified interface for geospatial vision-language models. Run any supported geospatial VLM with just a few lines of code.

## Installation

```bash
pip install goldeneye
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install goldeneye
```

## Quick Start

```python
import goldeneye

# List available models
print(goldeneye.assets())

# Load and run a model
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot")
response = model("path/to/satellite_image.jpg", "What is shown in this image?")
print(response)
```

## Supported Models

| Model                      | Size | HuggingFace ID                           | Notes                              |
| -------------------------- | ---- | ---------------------------------------- | ---------------------------------- |
| Geo-R1-3B-GRPO-REC-1shot   | 3B   | `Geo-R1/Geo-R1-3B-GRPO-REC-1shot`        | Referring expression comprehension |
| Geo-R1-3B-GRPO-REC-5shot   | 3B   | `Geo-R1/Geo-R1-3B-GRPO-REC-5shot`        | Referring expression comprehension |
| Geo-R1-3B-GRPO-REC-10shot  | 3B   | `Geo-R1/Geo-R1-3B-GRPO-REC-10shot`       | Referring expression comprehension |
| Geo-R1-3B-GRPO-GRES-1shot  | 3B   | `Geo-R1/Geo-R1-3B-GRPO-GRES-1shot`       | Generalized RES                    |
| Geo-R1-3B-GRPO-GRES-5shot  | 3B   | `Geo-R1/Geo-R1-3B-GRPO-GRES-5shot`       | Generalized RES                    |
| Geo-R1-3B-GRPO-GRES-10shot | 3B   | `Geo-R1/Geo-R1-3B-GRPO-GRES-10shot`      | Generalized RES                    |
| Geo-R1-3B-GRPO-OVD-5shot   | 3B   | `Geo-R1/Geo-R1-3B-GRPO-OVD-5shot`        | Open vocabulary detection          |
| Geo-R1-3B-GRPO-OVD-10shot  | 3B   | `Geo-R1/Geo-R1-3B-GRPO-OVD-10shot`       | Open vocabulary detection          |
| ZoomEarth-3B               | 3B   | `HappyBug/ZoomEarth-3B`                  | Zoom-in reasoning                  |
| DescribeEarth              | 3B   | `earth-insights/DescribeEarth`           | RS image captioning                |
| EarthDial-4B-RGB           | 4B   | `akshaydudhane/EarthDial_4B_RGB`         | RGB imagery                        |
| EarthDial-4B-MS            | 4B   | `akshaydudhane/EarthDial_4B_MS`          | Multispectral imagery              |
| EarthDial-4B-Methane-UHI   | 4B   | `akshaydudhane/EarthDial_4B_Methane_UHI` | Methane/UHI detection              |
| geochat-7B                 | 7B   | `MBZUAI/geochat-7B`                      | Grounded RS VLM                    |
| GeoLLaVA-8K                | 7B   | `initiacms/GeoLLaVA-8K`                  | Long-context RS VLM                |
| GeoZero                    | 8B   | `hjvsl/GeoZero`                          | General geospatial reasoning       |

### Memory Requirements

- **3B models** (GeoR1, ZoomEarth, DescribeEarth): ~6GB VRAM at fp16
- **4B models** (EarthDial): ~8GB VRAM at fp16
- **7B models** (GeoChat, GeoLLaVA): ~14GB VRAM at fp16
- **8B models** (GeoZero): ~16GB VRAM at fp16

## Usage

### Basic Usage

```python
import goldeneye
from PIL import Image

model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot")

# Use with file path
response = model("satellite_image.jpg", "Describe this image.")

# Use with PIL Image
image = Image.open("satellite_image.jpg")
response = model(image, "What type of land use is shown here?")

# Control response length
response = model(image, "Describe this image in detail.", max_new_tokens=256)
```

### Device and Dtype Selection

```python
import torch
import goldeneye

# Auto-detect device (default)
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot")

# Force specific device
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot", device="cuda")
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot", device="mps")  # Apple Silicon
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot", device="cpu")

# Specify dtype
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot", dtype=torch.bfloat16)
```

### Quantization (Reduce Memory Usage)

```python
import torch
import goldeneye
from transformers import BitsAndBytesConfig

# 8-bit quantization
config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model = goldeneye.dispatch_agent("geochat-7B", quantization_config=config_8bit)

# 4-bit quantization
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = goldeneye.dispatch_agent("GeoZero", quantization_config=config_4bit)
```

### Benchmark Datasets

Stream geospatial benchmark datasets directly:

```python
import goldeneye
from goldeneye.datasets import stream_xlrs_bench, stream_de_dataset

model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot")

# XLRS-Bench-lite
for sample in stream_xlrs_bench(split="train"):
    response = model(sample["image"], sample.get("question", "Describe this image."))
    break

# DE-Dataset (DescribeEarth)
for sample in stream_de_dataset(split="train"):
    response = model(sample["image"], "Describe this satellite image.")
    break
```

## Development

```bash
git clone https://github.com/isaaccorley/goldeneye.git
cd goldeneye
uv sync --all-extras
uv run pre-commit install
```

Run tests:

```bash
uv run pytest -vvv
```

## Citation

If you use goldeneye in your research, please cite the relevant model papers:

- **Geo-R1**: [Unlocking VLM Geospatial Reasoning with Cross-View Reinforcement Learning](https://arxiv.org/abs/2510.00072)
- **GeoChat**: [Grounded Large Vision-Language Model for Remote Sensing](https://arxiv.org/abs/2311.15826)
- **EarthDial**: [A Multi-Turn Multi-Modal Dataset & Benchmark for Earth Observation](https://arxiv.org/abs/2501.10724)

## License

MIT License - see [LICENSE](LICENSE) for details.
