# goldeneye

> **WIP**: This project is under active development. APIs may change.

Simple unified interface for geospatial vision-language models. Test any supported geospatial VLM with just one line of code.

## Installation

```bash
pip install goldeneye
```

Or using `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install goldeneye
```

## Quick Start

```python
import goldeneye

model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot")
response = model("path/to/image.jpg", "What is shown in this satellite image?")
print(response)
```

## Supported Models

| Model Family           | Size | Architecture  | HuggingFace ID                 | Notes                           |
| ---------------------- | ---- | ------------- | ------------------------------ | ------------------------------- |
| GeoR1 (8 variants)     | 3B   | Qwen2.5-VL-3B | `Geo-R1/Geo-R1-3B-GRPO-*`      | REC/GRES/OVD tasks, 1/5/10-shot |
| ZoomEarth              | 3B   | Qwen2.5-VL-3B | `HappyBug/ZoomEarth-3B`        | Zoom-in reasoning               |
| DescribeEarth          | 3B   | Qwen2.5-VL-3B | `earth-insights/DescribeEarth` | RS image captioning             |
| EarthDial (3 variants) | 4B   | InternVL2     | `akshaydudhane/EarthDial_4B_*` | RGB/MS/Methane-UHI              |
| GeoChat                | 7B   | LLaMA         | `MBZUAI/geochat-7B`            | Grounded RS VLM                 |
| GeoLLaVA               | 7B   | LongVA-7B     | `initiacms/GeoLLaVA-8K`        | Long-context RS VLM             |
| GeoZero                | 8B   | Qwen3-VL      | `hjvsl/GeoZero`                | Largest text model              |

### Memory Requirements

- **3B models** (GeoR1, ZoomEarth, DescribeEarth): ~6GB VRAM at fp16
- **4B models** (EarthDial): ~8GB VRAM at fp16
- **7B models** (GeoChat, GeoLLaVA): ~14GB VRAM at fp16, 8-bit recommended for 16GB GPUs
- **8B models** (GeoZero): ~16GB VRAM at fp16

## Usage

### List Available Models

```python
import goldeneye

print(goldeneye.assets())
```

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

### Device Selection

```python
import goldeneye

# Auto-detect (default)
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot")

# Force specific device
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot", device="cuda")  # NVIDIA GPU
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot", device="mps")  # Apple Silicon
model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot", device="cpu")  # CPU
```

### Benchmark Datasets

#### XLRS-Bench-lite

```python
import goldeneye
from goldeneye.datasets import stream_xlrs_bench

model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot")

for sample in stream_xlrs_bench(split="train"):
    image = sample["image"]
    question = sample.get("question", "Describe this image.")
    response = model(image, question)
    print(f"Q: {question}\nA: {response}\n")
    break  # Process just first sample
```

#### DE-Dataset (DescribeEarth)

```python
import goldeneye
from goldeneye.datasets import stream_de_dataset

model = goldeneye.dispatch_agent("DescribeEarth")

for sample in stream_de_dataset(split="train"):
    image = sample["image"]
    key = sample.get("__key__", "")
    response = model(image, "Describe what you see in this satellite image.")
    print(f"Key: {key}\nResponse: {response}\n")
    break  # Process just first sample
```

## Development

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras
uv run pre-commit install
```

## Model Information

- **Geo-R1**: [Paper](https://arxiv.org/abs/2510.00072) - Unlocking VLM Geospatial Reasoning with Cross-View Reinforcement Learning
- **geochat-7B**: [Paper](https://arxiv.org/abs/2311.15826) - Grounded Large Vision-Language Model for Remote Sensing
