<p align="center">
  <img src="assets/goldeneye-logo-vertical.png" alt="goldeneye logo" width="400">
</p>

[![PyPI version](https://badge.fury.io/py/goldeneye.svg)](https://badge.fury.io/py/goldeneye)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`goldeneye` is a simple and growing unified interface for geospatial vision-language models. Run any supported geospatial VLM with just a few lines of code.

## Installation

```bash
pip install goldeneye
```

## Quick Start

```python
import goldeneye

# List available agents (models)
print(goldeneye.assets())

# Dispatch an agent for collecting intel
model = goldeneye.dispatch_agent("DescribeEarth")
report = model.recon("assets/sample.jpg", "Describe this image.")
print(report)

# Report(
#    image='assets/sample.jpg',
#
#    prompt='Describe this image.',
#
#    response='The image depicts an aerial view of a
#    residential area surrounded by dense greenery,
#    likely trees and shrubs. The houses are
#    scattered across the landscape, with varying
#    sizes and designs, some featuring pitched roofs
#    and others flat-roofed structures. The roads
#    are visible as light-colored lines
#    crisscrossing the area, connecting'
# )
```

<p align="center">
  <img src="assets/sample.jpg" alt="sample satellite image" width="400">
</p>

## Supported Models

<details open>
<summary>Click to expand model list (7 models)</summary>

| Model                   | Size | Paper                                                                                                                                | Code                                                      |
| ----------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| **DescribeEarth**       | 3B   | [DescribeEarth: A Global Vision-Language Dataset for Aerial and Satellite Image Captioning](https://arxiv.org/abs/2509.25654v1)      | [github](https://github.com/earth-insights/DescribeEarth) |
| **ZoomEarth**           | 3B   | [ZoomEarth: A Unified Remote Sensing Framework for Multi-scale Vision-Language Tasks](https://arxiv.org/abs/2511.12267)              | [github](https://github.com/earth-insights/ZoomEarth)     |
| **EarthDial**           | 4B   | [EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](https://arxiv.org/abs/2501.10724)                     | [github](https://github.com/akshaydudhane16/EarthDial)    |
| **GeoChat**             | 7B   | [GeoChat: Grounded Large Vision-Language Model for Remote Sensing](https://arxiv.org/abs/2311.15826)                                 | [github](https://github.com/mbzuai-oryx/GeoChat)          |
| **GeoLLaVA-8K**         | 7B   | [GeoLLaVA-8K: A Large Vision-Language Model for High-Resolution Remote Sensing Applications](https://arxiv.org/abs/2505.21375)       | [github](https://github.com/MiliLab/GeoLLaVA-8K)          |
| **GeoZero**             | 8B   | [GeoZero: Zero-shot Geospatial Reasoning with Multimodal LLMs](https://arxiv.org/abs/2511.22645)                                     | [github](https://github.com/MiliLab/GeoZero)              |
| **Geo-R1 (8 variants)** | 3B   | [Geo-R1: Unleashing the Power of Reinforcement Learning in Generalist Geospatial Foundation Model](https://arxiv.org/abs/2510.00072) | [github](https://github.com/om-ai-lab/Geo-R1)             |

</details>

### Memory Requirements

- **3B models** (GeoR1, ZoomEarth, DescribeEarth): ~6GB VRAM at fp16
- **4B models** (EarthDial): ~8GB VRAM at fp16
- **7B models** (GeoChat, GeoLLaVA): ~14GB VRAM at fp16
- **8B models** (GeoZero): ~16GB VRAM at fp16

## Usage

```python
import torch
import goldeneye
from PIL import Image
from transformers import BitsAndBytesConfig

# Load a model (auto-detects device)
model = goldeneye.dispatch_agent("DescribeEarth")

# Or specify device/dtype
model = goldeneye.dispatch_agent("DescribeEarth", device="cuda", dtype=torch.bfloat16)

# Or use quantization for larger models
config = BitsAndBytesConfig(load_in_8bit=True)
model = goldeneye.dispatch_agent("GeoChat", quantization_config=config)

# Run inference with file path or PIL Image
report = model.recon("satellite_image.jpg", "Describe this image.")
report = model.recon(Image.open("satellite_image.jpg"), "Describe this image.", max_new_tokens=256)
```

### Benchmark Datasets

<details open>
<summary>Click to expand dataset list (2 datasets)</summary>

| Dataset             | Samples | Paper                                                                                                                           | HuggingFace                                                                            |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **DE-Dataset**      | ~321k   | [DescribeEarth: A Global Vision-Language Dataset for Aerial and Satellite Image Captioning](https://arxiv.org/abs/2509.25654v1) | [earth-insights/DE-Dataset](https://huggingface.co/datasets/earth-insights/DE-Dataset) |
| **XLRS-Bench-lite** | ~2.8k   | [XLRS-Bench: A Benchmark for Cross-Lingual Visual Reasoning in Remote Sensing](https://arxiv.org/abs/2503.23771)                | [initiacms/XLRS-Bench-lite](https://huggingface.co/datasets/initiacms/XLRS-Bench-lite) |

</details>

```python
from goldeneye.datasets import stream_de_dataset

# Stream samples from geospatial benchmarks
for sample in stream_de_dataset(split="train"):
    report = model.recon(sample["image"], "Describe this satellite image.")
    break
```

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for development setup and guidelines.
