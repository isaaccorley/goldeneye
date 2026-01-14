# Geospatial VLM Models

A curated list of geospatial/remote sensing vision-language models available on HuggingFace.

## Model Status Legend

- ✅ **Integrated** - Available in `goldeneye` via `load_agent()`
- 🔧 **TODO** - Not yet integrated, needs implementation
- ⚠️ **Partial** - Partially working or needs fixes

______________________________________________________________________

## Integrated Models (in goldeneye)

| Model         | HuggingFace ID      | Size | Status        | Notes                                  |
| ------------- | ------------------- | ---- | ------------- | -------------------------------------- |
| GeoR1         | `Carkham/GeoR1-7B`  | 7B   | ✅ Integrated | Qwen2.5-VL based, geospatial reasoning |
| GeoZero       | Various             | 7B   | ✅ Integrated | Zero-shot geospatial VLM               |
| GeoChat       | `MBZUAI/geochat-7B` | 7B   | ✅ Integrated | MBZUAI's geospatial chat model         |
| GeoLLaVA      | Various             | 7B   | ✅ Integrated | LLaVA-based geospatial model           |
| EarthDial     | Various             | 7B   | ✅ Integrated | Earth observation VLM                  |
| DescribeEarth | Various             | 7B   | ✅ Integrated | Image description model                |
| ZoomEarth     | Various             | 7B   | ✅ Integrated | Multi-resolution RS model              |

______________________________________________________________________

## Models to Integrate

| Model           | HuggingFace ID                                 | Size | Status  | Notes                                                            |
| --------------- | ---------------------------------------------- | ---- | ------- | ---------------------------------------------------------------- |
| VHM             | `FitzPC/vhm_7B`                                | 7B   | 🔧 TODO | Versatile RS VLM. Trained on VHM_VersaD (4.1M).                  |
| TEOChat         | `jirvin16/TEOChat`                             | 7B   | 🔧 TODO | Temporal Earth Observation chat.                                 |
| SegEarth-R1     | `earth-insights/SegEarth-R1`                   | 7B   | 🔧 TODO | Geospatial pixel reasoning.                                      |
| CCExpert        | Various                                        | 7B   | 🔧 TODO | Change captioning expert.                                        |
| GeoPix          | `Norman-ou/GeoPix-ft-sior_rsicap`              | -    | 🔧 TODO | Pixel-level RS MLLM. Referring segmentation.                     |
| UniGeoSeg       | `nishuo1999/UniGeoSeg`                         | 2B   | 🔧 TODO | Unified geo segmentation. LLaVA-Phi based.                       |
| SkySenseGPT     | `ll-13/SkySenseGPT-7B-CLIP-ViT`                | 7B   | 🔧 TODO | RS VLM with fine-grained understanding.                          |
| GeoChat-UAV     | `ZhanYang-nwpu/GeoChat-UAV`                    | 7B   | 🔧 TODO | UAV-specific GeoChat. UAVMLLM collection.                        |
| RSCoVLM         | `Qingyun/RSCoVLM-7B-2512`                      | 7B   | 🔧 TODO | Multi-task RS VLM (Qwen2.5-VL based). Detection + understanding. |
| RSCoVLM-det     | `Qingyun/RSCoVLM-det-7B-2512`                  | 7B   | 🔧 TODO | Detection-only version of RSCoVLM.                               |
| LMMRotate       | `Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate` | 0.9B | 🔧 TODO | Florence-2 for oriented object detection.                        |
| EarthMind       | `sy1998/EarthMind-4B`                          | 4B   | 🔧 TODO | RS reasoning with segmentation. SA2VA architecture.              |
| EarthMind-Multi | `sy1998/EarthMind4B_multi`                     | 4B   | 🔧 TODO | Multi-task EarthMind variant.                                    |
| MF-RSVLM        | `FelixKAI/mfrsvlm-7b_sft`                      | 7B   | 🔧 TODO | Feature fusion RS VLM. CLIP + Vicuna-7B.                         |

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
