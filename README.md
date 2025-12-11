# geovllm

Simple unified interface for geospatial vision-language models. Test any supported geospatial VLM with just one line of code.

## Installation

```bash
pip install geovllm
```

Or using `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install geovllm
```

## Quick Start

```python
import geovllm

model = geovllm.load_model("Geo-R1-3B-GRPO-REC-5shot")
response = model("path/to/image.jpg", "What is shown in this satellite image?")
print(response)
```

## Supported Models

- **GeoZero** (`hjvsl/GeoZero`)
- **GeoLLaVA-8K** (`initiacms/GeoLLaVA-8K`) - Based on LongVA-7B
- **Geo-R1-3B** - Qwen2.5-VL-3B based geospatial reasoning models (8 variants: REC/GRES/OVD with 1/5/10-shot)
- **EarthDial-4B** - InternVL2-based models (RGB, MS, Methane-UHI variants)
- **geochat-7B** (`MBZUAI/geochat-7B`) - Grounded Large Vision Language Model for Remote Sensing
- **GeoPixel-7B** (`MBZUAI/GeoPixel-7B`) - Pixel grounding model for RS-GCG task
- **GeoPixel-7B-RES** (`MBZUAI/GeoPixel-7B-RES`) - Pixel grounding model for RRSIS task
- **ZoomEarth-3B** (`HappyBug/ZoomEarth-3B`) - Qwen2.5-VL based zoom-in reasoning model
- **DescribeEarth** (`earth-insights/DescribeEarth`) - Remote sensing image captioning model

## Usage

### List Available Models

```python
import geovllm

print(geovllm.list_models())
```

### Basic Usage

```python
import geovllm
from PIL import Image

model = geovllm.load_model("Geo-R1-3B-GRPO-REC-5shot")

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
import geovllm

# Auto-detect (default)
model = geovllm.load_model("Geo-R1-3B-GRPO-REC-5shot")

# Force specific device
model = geovllm.load_model("Geo-R1-3B-GRPO-REC-5shot", device="cuda")  # NVIDIA GPU
model = geovllm.load_model("Geo-R1-3B-GRPO-REC-5shot", device="mps")  # Apple Silicon
model = geovllm.load_model("Geo-R1-3B-GRPO-REC-5shot", device="cpu")  # CPU
```

### Pixel Grounding with GeoPixel

GeoPixel models support pixel-level segmentation:

```python
import geovllm
import numpy as np
from PIL import Image

model = geovllm.load_model("GeoPixel-7B")
image = Image.open("satellite_image.jpg")

# Get text response with segmentation masks
response, masks = model.generate_with_masks(
    image, "Segment all buildings in this image.", max_new_tokens=128
)

# Masks are numpy arrays (H, W) with values 0 or 1
for i, mask in enumerate(masks):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(f"mask_{i}.png")
```

### Benchmark Datasets

#### XLRS-Bench-lite

```python
import geovllm
from geovllm.datasets import stream_xlrs_bench

model = geovllm.load_model("Geo-R1-3B-GRPO-REC-5shot")

for sample in stream_xlrs_bench(split="train"):
    image = sample["image"]
    question = sample.get("question", "Describe this image.")
    response = model(image, question)
    print(f"Q: {question}\nA: {response}\n")
    break  # Process just first sample
```

#### DE-Dataset (DescribeEarth)

```python
import geovllm
from geovllm.datasets import stream_de_dataset

model = geovllm.load_model("DescribeEarth")

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
- **GeoPixel-7B**: [Paper](https://arxiv.org/abs/2501.13925) - Pixel Grounding Large Multimodal Models in Remote Sensing
