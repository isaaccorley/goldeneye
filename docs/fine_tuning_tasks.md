# Fine-tuning Task Support Matrix

This document outlines the distinct fine-tuning tasks available across the goldeneye datasets.

## Task Definitions

| Task | Description |
|------|-------------|
| **Captioning** | Generate natural language descriptions for images |
| **VQA** | Visual Question Answering - answer questions about images |
| **MCQ** | Multiple Choice Questions - select correct answer from options |
| **Grounding** | Locate objects/regions given text descriptions (bbox/polygon output) |
| **Referring Expression** | Understand referring expressions to identify objects |
| **Change Detection** | Describe or detect changes between bi-temporal image pairs |
| **Scene Classification** | Classify the scene type of an image |
| **Object Detection** | Detect and localize objects with bounding boxes |
| **Conversation** | Multi-turn dialogue about images |
| **Chain-of-Thought** | Reasoning with step-by-step explanations |
| **Segmentation** | Pixel-level segmentation masks |

## Dataset Task Matrix

| Dataset | Captioning | VQA | MCQ | Grounding | Change Det. | Scene Class. | Conversation | CoT | Samples | Streaming |
|---------|:----------:|:---:|:---:|:---------:|:-----------:|:------------:|:------------:|:---:|--------:|:---------:|
| **changechat** | | | | | ✅ | | ✅ | | 87K | ✅ |
| **de-dataset** | ✅ | | | ✅ | | | | | 262K | ✅ |
| **disaster-m3** | | | | | ✅ | | | | 27K | ✅ |
| **earthdial** | ✅ | | | ✅ | | ✅ | | | varies | ✅ |
| **evattrs** | | | | | | | | | 95K | ✅ |
| **fair1m-caption** | ✅ | | | | | | | | 22K | ✅ |
| **geochat-bench** | | ✅ | | ✅ | | ✅ | | | varies | ✅ |
| **geochat-instruct** | ✅ | ✅ | | ✅ | | ✅ | ✅ | | 318K | ❌ |
| **geopix-instruct** | | | | ✅ | | | ✅ | | 58K | ✅ |
| **geotext1652** | | | | | | | | | 1.6K | ✅ |
| **geozero-eval** | | ✅ | | ✅ | | | ✅ | | varies | ✅ |
| **gres** | | | | | | | | | varies | ✅ |
| **landsat-captions** | ✅ | | | | | | | | varies | ✅ |
| **lrs-gro** | | ✅ | | | | ✅ | | | varies | ✅ |
| **lrs-vqa** | | ✅ | | | | | | | 7.3K | ✅ |
| **nwpu-captions** | ✅ | | | | | ✅ | | | 31.5K | ✅ |
| **pregres** | | | | ✅ | | | ✅ | | varies | ✅ |
| **refgeo** | | | | ✅ | | | | | varies | ✅ |
| **rs-eot** | | ✅ | | | | | | ✅ | 4K | ✅ |
| **rs-visual-instructions** | ✅ | ✅ | | | | | ✅ | | 36K | ✅ |
| **rs-vqa** | | ✅ | | | | ✅ | | | varies | ✅ |
| **rs5m** | ✅ | | | | | | | | 7.25M | ✅ |
| **rscc** | | | | | ✅ | | | | 62K | ✅ |
| **rscid-captions** | ✅ | | | | | | | | varies | ✅ |
| **rsicd** | ✅ | | | | | | | | 10.9K | ✅ |
| **rsteller** | ✅ | | | | | | | | 1M+ | ❌ |
| **rsvqa-hr** | | ✅ | | | | | ✅ | | 358K | ✅ |
| **sarlang** | ✅ | | | | | | | | 1M+ | ✅ |
| **sydney-captions** | ✅ | | | | | | | | 613 | ✅ |
| **uavbench** | | ✅ | | | | | ✅ | | 966K | ✅ |
| **uavit-1m** | | ✅ | | | | | ✅ | | 1.24M | ✅ |
| **ucm-captions** | ✅ | | | | | | | | 10.5K | ✅ |
| **urbench** | | ✅ | ✅ | ✅ | | ✅ | | | varies | ✅ |
| **vhm-versad** | ✅ | | | | | | | | 4.1M | ✅ |
| **vrsbench** | ✅ | ✅ | | ✅ | | | | | 29.6K | ✅ |
| **xhrbench** | ✅ | ✅ | ✅ | | | | | | varies | ✅ |
| **xlrs-bench** | | ✅ | ✅ | | | | | | varies | ✅ |

## Task Summary

| Task | Dataset Count | Key Datasets |
|------|---------------|--------------|
| **Captioning** | 17 | rs5m, vhm-versad, rsteller, sarlang, rsicd |
| **VQA** | 16 | rsvqa-hr, uavit-1m, vrsbench, lrs-vqa |
| **Conversation** | 11 | changechat, geochat-instruct, uavbench, rsvqa-hr |
| **Grounding** | 9 | de-dataset, geochat-bench, vrsbench, refgeo |
| **Scene Classification** | 7 | earthdial, nwpu-captions, lrs-gro, rs-vqa |
| **Change Detection** | 4 | changechat, disaster-m3, rscc |
| **MCQ** | 4 | urbench, xhrbench, xlrs-bench |
| **Chain-of-Thought** | 1 | rs-eot |

## Recommended Datasets by Task

### Image Captioning
Best datasets for training image captioning models:
- **rs5m** (7.25M) - Largest, general RS captions
- **vhm-versad** (4.1M) - Large-scale pre-training
- **rsteller** (1M+) - Diverse RS scenes (no streaming)
- **rsicd** (10.9K) - High quality, 5 captions per image

### Visual Question Answering
Best datasets for VQA fine-tuning:
- **rsvqa-hr** (358K) - High resolution, Qwen format
- **uavit-1m** (1.24M) - UAV scenes, conversation format
- **vrsbench** (29.6K) - Comprehensive with 123K QA pairs
- **lrs-vqa** (7.3K) - Large high-resolution images

### Instruction Following / Conversation
Best datasets for instruction tuning:
- **geochat-instruct** (318K) - Multi-task instruction data
- **uavbench** (966K) - UAV-focused conversations
- **rs-visual-instructions** (36K) - Synthetic VQA + captions
- **changechat** (87K) - Change detection conversations

### Visual Grounding
Best datasets for grounding/referring expression:
- **de-dataset** (262K) - Detailed object descriptions
- **vrsbench** (29.6K) - Object detections + expressions
- **refgeo** - Referring expressions with bbox/polygon
- **geopix-instruct** (58K) - Pixel-level instructions

### Change Detection
Best datasets for temporal change understanding:
- **changechat** (87K) - Bi-temporal conversations
- **rscc** (62K) - Change captions (bi-temporal pairs)
- **disaster-m3** (27K) - Disaster assessment

## Data Format Notes

### Embedded Images (Ready to Use)
These datasets have PIL Images directly embedded:
- rsicd, fair1m-caption, rscid-captions, ucm-captions
- sydney-captions, rsvqa-hr, landsat-captions
- rs-eot, rs-vqa, geotext1652, earthdial

### Bytes Format (Needs Decoding)
These use webdataset format with image bytes:
- **rs5m**: `img_content` field contains JPEG bytes
- **vhm-versad**: `jpg` field contains image bytes
- **nwpu-captions**: `jpg` field contains image bytes
- **de-dataset**: `jpg` field contains PIL Image

### Path References (External Download Required)
These datasets reference external image paths:
- geochat-bench, geochat-instruct, vrsbench
- lrs-vqa, uavbench, uavit-1m, xhrbench
- geopix-instruct, pregres, refgeo, rscc

## Streaming Support

**35/37 datasets support streaming** (94.6%)

Only these require full download:
- `geochat-instruct` - Data type conversion issues
- `rsteller` - Struct casting issues

