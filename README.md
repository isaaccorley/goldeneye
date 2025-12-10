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

# Load any supported model
model = geovllm.load_model("GeoR1")

# Use with an image and prompt
response = model("path/to/image.jpg", "What is shown in this satellite image?")
print(response)
```

## Supported Models

- **GeoZero** (`hjvsl/GeoZero`)
- **GeoLLaVA-8K** (`initiacms/GeoLLaVA-8K`) - Based on LongVA-7B
- **Geo-R1** (`miniHui/Geo-R1`) - Qwen2.5-VL based reasoning model
- **EarthGPT** (`Pruz0/EarthGPT`) - GPT2-based geospatial model
- **geochat-7B** (`MBZUAI/geochat-7B`) - Grounded Large Vision Language Model for Remote Sensing
- **GeoPixel-7B** (`MBZUAI/GeoPixel-7B`) - Pixel grounding model for RS-GCG task
- **GeoPixel-7B-RES** (`MBZUAI/GeoPixel-7B-RES`) - Pixel grounding model for RRSIS task
- **SAM3** (`facebook/sam3`) - Segment Anything Model 3 for promptable concept segmentation

## Usage

### List Available Models

```python
import geovllm

print(geovllm.list_models())
# ['GeoZero', 'GeoLLaVA-8K', 'Geo-R1', 'EarthGPT', 'geochat-7B', 'GeoPixel-7B-RES', 'GeoPixel-7B']
```

### Load and Use a Model

```python
import geovllm
from PIL import Image

# Load model (automatically uses GPU if available)
model = geovllm.load_model("GeoR1")

# Use with file path
response = model("satellite_image.jpg", "Describe this image.")

# Use with PIL Image
image = Image.open("satellite_image.jpg")
response = model(image, "What type of land use is shown here?")

# Or use the generate method explicitly
response = model.generate(image, "Analyze the urban development in this area.")
```

### Specify Device

```python
import geovllm

# Force CPU usage
model = geovllm.load_model("GeoR1", device="cpu")

# Force CUDA (NVIDIA GPU)
model = geovllm.load_model("GeoR1", device="cuda")

# Force MPS (Apple Silicon GPU)
model = geovllm.load_model("GeoR1", device="mps")
```

**Note**: By default, models automatically detect and use the best available device:

- CUDA (if NVIDIA GPU available)
- MPS (if Apple Silicon Mac with macOS 12.3+)
- CPU (fallback)

### Testing with Benchmark Dataset

Test your models on the [XLRS-Bench-lite](https://huggingface.co/datasets/initiacms/XLRS-Bench-lite) benchmark dataset. The dataset can be streamed sample-by-sample without downloading the entire dataset:

```python
import geovllm
from geovllm.datasets import stream_xlrs_bench

model = geovllm.load_model("GeoR1")

# Stream dataset (no download required - samples fetched on-demand)
for sample in stream_xlrs_bench(split="train"):
    image = sample["image"]
    question = sample.get("question", "Describe this image.")

    response = model(image, question)
    print(f"Q: {question}")
    print(f"A: {response}\n")

    # Process just a few samples
    break
```

You can also load the entire dataset (downloads to disk):

```python
from geovllm.datasets import load_xlrs_bench

# Download entire dataset
dataset = load_xlrs_bench(split="train", streaming=False)
print(f"Dataset size: {len(dataset)}")

# Or stream without downloading
dataset = load_xlrs_bench(split="train", streaming=True)
```

## Development

We use [uv](https://docs.astral.sh/uv/), [ty](https://docs.astral.sh/ty/), and [pre-commit](https://pre-commit.com/) to make this project easy to use.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras
uv run pre-commit install

# optionally run pre-commit hooks manually
uv run pre-commit run --all-files
```

## Model Information

- **Geo-R1**: [Paper](https://arxiv.org/abs/2510.00072) - Unlocking VLM Geospatial Reasoning with Cross-View Reinforcement Learning
- **geochat-7B**: [Paper](https://arxiv.org/abs/2311.15826) - Grounded Large Vision-Language Model for Remote Sensing
- **GeoPixel-7B**: [Paper](https://arxiv.org/abs/2501.13925) - Pixel Grounding Large Multimodal Models in Remote Sensing
