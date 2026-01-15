# Geospatial VLM Models

A curated list of geospatial/remote sensing vision-language models available on HuggingFace.

## Model Status Legend

- ✅ **Integrated** - Available in `goldeneye` via `load_agent()`
- 🔧 **TODO** - Not yet integrated, needs implementation
- ⚠️ **Partial** - Partially working or needs fixes

______________________________________________________________________

## Integrated Models (in goldeneye)

| Model          | HuggingFace ID                                 | Size | Status        | Notes                                    |
| -------------- | ---------------------------------------------- | ---- | ------------- | ---------------------------------------- |
| GeoR1          | `Geo-R1/Geo-R1-3B-GRPO-*`                      | 3B   | ✅ Integrated | Qwen2.5-VL based, geospatial reasoning   |
| GeoZero        | `hjvsl/GeoZero`                                | 7B   | ✅ Integrated | Zero-shot geospatial VLM                 |
| GeoChat        | `MBZUAI/geochat-7B`                            | 7B   | ✅ Integrated | MBZUAI's geospatial chat model           |
| GeoLLaVA       | `initiacms/GeoLLaVA-8K`                        | 7B   | ✅ Integrated | LLaVA-based geospatial model             |
| EarthDial      | `akshaydudhane/EarthDial_4B_RGB`               | 4B   | ✅ Integrated | Earth observation VLM                    |
| DescribeEarth  | `earth-insights/DescribeEarth`                 | 7B   | ✅ Integrated | Image description model                  |
| ZoomEarth      | `HappyBug/ZoomEarth-3B`                        | 3B   | ✅ Integrated | Multi-resolution RS model                |
| RSCoVLM-7B     | `Qingyun/RSCoVLM-7B-2512`                      | 7B   | ✅ Integrated | Qwen2.5-VL based, multi-task RS VLM      |
| RSCoVLM-det-7B | `Qingyun/RSCoVLM-det-7B-2512`                  | 7B   | ✅ Integrated | Detection-focused RSCoVLM variant        |
| Florence2-DOTA | `Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate` | 0.9B | ✅ Integrated | Florence-2 for oriented object detection |
| GeoChat-UAV    | `ZhanYang-nwpu/GeoChat-UAV`                    | 7B   | ✅ Integrated | UAV-specific GeoChat variant             |
| SkySenseGPT    | `ll-13/SkySenseGPT-7B-CLIP-ViT`                | 7B   | ✅ Integrated | Fine-grained RS understanding            |

______________________________________________________________________

## Models to Integrate

| Model       | HuggingFace ID               | Size | Status  | Notes                                           |
| ----------- | ---------------------------- | ---- | ------- | ----------------------------------------------- |
| VHM         | `FitzPC/vhm_7B`              | 7B   | 🔧 TODO | Versatile RS VLM. Trained on VHM_VersaD (4.1M). |
| TEOChat     | `jirvin16/TEOChat`           | 7B   | 🔧 TODO | Temporal Earth Observation chat.                |
| SegEarth-R1 | `earth-insights/SegEarth-R1` | 7B   | 🔧 TODO | Geospatial pixel reasoning.                     |
| CCExpert    | Various                      | 7B   | 🔧 TODO | Change captioning expert.                       |

______________________________________________________________________

## Not Supported Models

These models cannot be loaded with standard HuggingFace transformers and require custom code from their original repositories.

| Model           | HuggingFace ID                    | Size | Reason                                              |
| --------------- | --------------------------------- | ---- | --------------------------------------------------- |
| GeoPix          | `Norman-ou/GeoPix-ft-sior_rsicap` | 7B   | Custom architecture with mask predictor             |
| UniGeoSeg       | `nishuo1999/UniGeoSeg`            | 2B   | Custom `llava_phi` model type not in transformers   |
| EarthMind       | `sy1998/EarthMind-4B`             | 4B   | SA2VA-Chat custom architecture                      |
| EarthMind-Multi | `sy1998/EarthMind4B_multi`        | 4B   | SA2VA-Chat custom architecture                      |
| MF-RSVLM        | `FelixKAI/mfrsvlm-7b_sft`         | 7B   | Custom `mfrsvlm` model type not in transformers     |
| EagleVision-1B  | `liarzone/EagleVision-1B`         | 1B   | Hybrid detector+LLM, not a pure VLM                 |
| EagleVision-2B  | `liarzone/EagleVision-2B`         | 2B   | Hybrid detector+LLM, not a pure VLM                 |
| EagleVision-4B  | `liarzone/EagleVision-4B`         | 4B   | Hybrid detector+LLM, not a pure VLM                 |
| EagleVision-7B  | `liarzone/EagleVision-7B`         | 7B   | Hybrid detector+LLM, not a pure VLM                 |
| RSUniVLM        | `isaaccorley/RSUniVLM`            | 2B   | Custom `llava_qwen_gmoe` architecture               |
| LISAt-7b        | `jquenum/LISAt-7b`                | 7B   | LISA architecture with SAM decoder for segmentation |
| LISAt_PRE-7b    | `jquenum/LISAt_PRE-7b`            | 7B   | LISA architecture with SAM decoder for segmentation |
| MiniGPTv2-UAV   | `ZhanYang-nwpu/MiniGPTv2-UAV`     | 7B   | MiniGPT-v2 custom architecture                      |
| LLaVA1.5-UAV    | `ZhanYang-nwpu/LLaVA1.5-UAV`      | 7B   | Custom `llava_llama` model type not in transformers |
| GeoGround       | `erenzhou/GeoGround`              | 7B   | Missing `model_type` in config, LLaVA-1.5 with LoRA |

______________________________________________________________________

## Model Details

### VHM (Versatile Height Model)

- **HuggingFace**: [`FitzPC/vhm_7B`](https://huggingface.co/FitzPC/vhm_7B)
- **Architecture**: PyTorch, VHM architecture
- **Training Data**: VHM_VersaD (4.1M samples), VHM_dataset_sft
- **Tasks**: Remote sensing image understanding
- **Downloads**: ~127/month

### TEOChat

- **HuggingFace**: [`jirvin16/TEOChat`](https://huggingface.co/jirvin16/TEOChat)
- **Paper**: [arXiv:2410.06234](https://arxiv.org/abs/2410.06234)
- **Architecture**: Based on LLaVA
- **Training Data**: TEOChatlas (500K+ samples from fMoW, xBD, S2Looking, QFabric)
- **Tasks**: Temporal Earth observation, change detection, multi-temporal understanding

### SegEarth-R1

- **HuggingFace**: [`earth-insights/SegEarth-R1`](https://huggingface.co/earth-insights)
- **Paper**: [arXiv:2504.09644](https://arxiv.org/abs/2504.09644)
- **Training Data**: EarthReason dataset
- **Tasks**: Geospatial pixel reasoning, segmentation

### GeoPix

- **HuggingFace**: [`Norman-ou/GeoPix-ft-sior_rsicap`](https://huggingface.co/Norman-ou/GeoPix-ft-sior_rsicap)
- **Paper**: [arXiv:2501.06828](https://arxiv.org/abs/2501.06828)
- **GitHub**: [Norman-Ou/GeoPix](https://github.com/Norman-Ou/GeoPix)
- **Training Data**: GeoPixInstruct-Anno (58K samples)
- **Tasks**: Pixel-level image understanding, referring segmentation

### RSCoVLM

- **HuggingFace**: [`Qingyun/RSCoVLM-7B-2512`](https://huggingface.co/Qingyun/RSCoVLM-7B-2512)
- **Paper**: [arXiv:2511.21272](https://arxiv.org/abs/2511.21272)
- **GitHub**: [VisionXLab/RSCoVLM](https://github.com/VisionXLab/RSCoVLM)
- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Training Data**: remote-sensing-sft-data (comprehensive RS collection)
- **Tasks**: Multi-task RS understanding, ultra-high-resolution reasoning, oriented object detection

### LMMRotate

- **HuggingFace**: [`Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate`](https://huggingface.co/Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate)
- **Paper**: [arXiv:2501.09720](https://arxiv.org/abs/2501.09720) (IGARSS 2025 Oral)
- **GitHub**: [Li-Qingyun/mllm-mmrotate](https://github.com/Li-Qingyun/mllm-mmrotate)
- **Base Model**: Florence-2-large
- **Training Data**: lmmrotate-sft-data (DOTA, DIOR-R, FAIR1M, SRSDD, RSAR)
- **Tasks**: Oriented object detection in aerial images

### EarthMind

- **HuggingFace**: [`sy1998/EarthMind-4B`](https://huggingface.co/sy1998/EarthMind-4B)
- **Architecture**: SA2VA-based, 4B parameters
- **Training Data**: EarthMind-data
- **Tasks**: RS reasoning, segmentation, multi-task understanding

### MF-RSVLM

- **HuggingFace**: [`FelixKAI/mfrsvlm-7b_sft`](https://huggingface.co/FelixKAI/mfrsvlm-7b_sft)
- **Paper**: [arXiv:2512.24022](https://arxiv.org/abs/2512.24022)
- **GitHub**: [opendatalab/MF-RSVLM](https://github.com/opendatalab/MF-RSVLM)
- **Architecture**: CLIP ViT-L/14 + 2-layer MLP + Vicuna-7B
- **Training Data**: VHM_VersaD (1.4M pretrain), RSVLM-SFT (instruction tuning)
- **Tasks**: RS image understanding, VQA, captioning

### SkySenseGPT

- **HuggingFace**: [`ll-13/SkySenseGPT-7B-CLIP-ViT`](https://huggingface.co/ll-13/SkySenseGPT-7B-CLIP-ViT)
- **GitHub**: [Luo-Z13/SkySenseGPT](https://github.com/Luo-Z13/SkySenseGPT)
- **Training Data**: FIT-RS (1.8M instruction samples)
- **Tasks**: Fine-grained RS understanding, object relationships

### GeoChat-UAV / UAVMLLM

- **HuggingFace**: [`ZhanYang-nwpu/GeoChat-UAV`](https://huggingface.co/ZhanYang-nwpu/GeoChat-UAV)
- **Collection**: [UAVMLLM](https://huggingface.co/collections/ZhanYang-nwpu/uavmllm)
- **Training Data**: UAVIT-1M (1.24M instructions, 789K UAV images)
- **Tasks**: Low-altitude UAV image understanding, 11 image/region-level tasks

### GeoPixel

- **HuggingFace**: [MBZUAI/geopixel collection](https://huggingface.co/collections/MBZUAI/geopixel-67b6e1e441250814d06f2043)
- **Paper**: [arXiv:2501.13925](https://arxiv.org/abs/2501.13925) (ICML 2025)
- **GitHub**: [mbzuai-oryx/GeoPixel](https://github.com/mbzuai-oryx/GeoPixel)
- **Training Data**: GeoPixelD (53K phrases, 600K objects)
- **Tasks**: Pixel-level grounding, fine-grained RS understanding

### LHRS-Bot-Nova

- **HuggingFace**: [`LHRS/LHRS-Bot-Nova`](https://huggingface.co/LHRS/LHRS-Bot-Nova)
- **Paper**: ECCV 2024
- **GitHub**: [NJU-LHRS/LHRS-Bot](https://github.com/NJU-LHRS/LHRS-Bot)
- **Architecture**: LLaMA2-7B-Chat based
- **Training Data**: LHRS_Data benchmark
- **Tasks**: RS image understanding, VQA

### Falcon (RS-VLM)

- **GitHub**: [TianHuiLab/Falcon](https://github.com/TianHuiLab/Falcon)
- **Paper**: arXiv 2025
- **Tasks**: Multi-scale RS understanding

### SARChat

- **GitHub**: [JimmyMa99/SARChat](https://github.com/JimmyMa99/SARChat)
- **Paper**: arXiv 2025
- **Tasks**: SAR image understanding, conversational AI

### DisasterM3 VLM

- **HuggingFace**: [`Kingdrone-Junjue/DisasterM3`](https://huggingface.co/datasets/Kingdrone-Junjue/DisasterM3)
- **Paper**: [arXiv:2505.21089](https://arxiv.org/abs/2505.21089) (NeurIPS 2025)
- **GitHub**: [Junjue-Wang/DisasterM3](https://github.com/Junjue-Wang/DisasterM3)
- **Tasks**: Disaster damage assessment, multi-task VLM (9 tasks, 10 disaster types)

______________________________________________________________________

## Adding a New Model

To add a new model to goldeneye:

1. Create `src/goldeneye/models/<name>/` directory
1. Implement agent class inheriting from `BaseAgent`
1. Implement `recon()` method for inference
1. Register in `registry.py`
1. Add unit tests in `tests/test_models.py`

See [AGENTS.md](../AGENTS.md) for detailed instructions.

______________________________________________________________________

## Related Resources

- [Datasets Documentation](datasets.md)
- [HuggingFace Remote Sensing Models](https://huggingface.co/models?other=remote-sensing)
- [Awesome Remote Sensing VLMs](https://github.com/topics/remote-sensing-vlm)
