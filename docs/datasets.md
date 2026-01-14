# Geospatial VLM Datasets for Fine-tuning

A curated list of remote sensing / geospatial vision-language datasets available on HuggingFace for fine-tuning VLMs.

## Dataset Status Legend

- ✅ **Works** - Loads with `load_dataset()` and has proper image+text columns
- ⚠️ **Partial** - Loads but missing some expected columns or needs preprocessing
- 🔧 **Needs Work** - Requires custom loader or has access restrictions
- ❌ **Not Available** - Not on HuggingFace or broken

______________________________________________________________________

## Image Captioning Datasets

| Dataset         | HuggingFace ID                         | Size   | Status           | Keys/Notes                                                                     |
| --------------- | -------------------------------------- | ------ | ---------------- | ------------------------------------------------------------------------------ |
| RSICD           | `arampacha/rsicd`                      | 10.9K  | ✅ Works         | `image`, `filename`, `captions` (list of 5)                                    |
| UCM-Captions    | `cpratikaki/UCMcaptions_finetuning`    | 10.5K  | ✅ Works         | `image`, `caption`                                                             |
| NWPU-Captions   | `KhangTruong/NWPU-Caption`             | 31.5K  | ⚠️ Partial       | Webdataset: `jpg`, `__key__`, `__url__` (captions in separate JSON)            |
| RSCID-Captions  | `mthandazo/rscid_captions_vqa_dataset` | varies | ✅ Works         | `image`, `caption`                                                             |
| RSITMD-Captions | `Hari2207/RSITMD`                      | varies | ❌ Not Available | Empty dataset on HuggingFace                                                   |
| FAIR1M Caption  | `blanchon/FAIR1M_Small_Caption`        | 22.3K  | ✅ Works         | `image`, `text`                                                                |
| RS5M            | `omlab/RS5M`                           | 7.25M  | ✅ Works         | Webdataset: `caption`, `img_content`, `img_name`, `__key__`, `__url__`         |
| Landsat30-AU    | `supermarkioner/Landsat30-AU`          | 100K+  | ✅ Works         | `image_id`, `image_path`, `caption`, `gt_caption`. Australian Landsat imagery. |
| RSTeller        | `SlytherinGe/RSTeller`                 | 1M+    | ✅ Works         | Webdataset format. Large-scale RS image-text pairs.                            |

### Captioning Datasets - Loader Status

- [x] `load_rsicd` - Implemented ✅
- [x] `load_ucm_captions` - Implemented ✅
- [x] `load_nwpu_captions` - Implemented (webdataset) ⚠️
- [x] `load_fair1m_caption` - Implemented ✅
- [x] `load_rs5m` - Implemented ✅
- [x] `load_landsat30_au` - Implemented ✅
- [x] `load_rsteller` - Implemented ✅

______________________________________________________________________

## Visual Question Answering (VQA) Datasets

| Dataset      | HuggingFace ID                        | Size         | Status        | Keys/Notes                                                                     |
| ------------ | ------------------------------------- | ------------ | ------------- | ------------------------------------------------------------------------------ |
| RS-VQA       | `WaltonFuture/remote-sensing-VQA`     | 17K          | ✅ Works      | `images` (list), `problem`, `answer`                                           |
| RSVQA-HR     | `cpratikaki/RSVQA-HR_qwen_finetuning` | 358K         | ✅ Works      | `image`, `messages` (Qwen chat format)                                         |
| XLRS-Bench   | `initiacms/XLRS-Bench-lite`           | varies       | ✅ Works      | Existing loader                                                                |
| VRSBench     | `xiang709/VRSBench`                   | 29.6K images | ✅ Works      | `caption`, `objects`, `qa_pairs`, `image`. Captions, referring, 123K QA pairs. |
| Geo3DVQA     | `mt129/Geo3DVQA`                      | 2.1K         | ⚠️ Partial    | `image`, `label` only (3D VQA with sky view factor)                            |
| LRS-VQA      | `ll-13/LRS-VQA`                       | 7.3K         | ✅ Works      | Large RS images VQA (1K-27K pixel images). 8 question types.                   |
| FloodNet VQA | `FrancescoCiccone/FloodNet_VQA`       | 7.4K         | 🔧 Needs Work | Requires access agreement                                                      |
| XHRBench     | `FelixKAI/XHRBench`                   | 1.9K         | ✅ Works      | Ultra-high-res RS benchmark (4K-300MP). MCQ, OEQ, captioning.                  |

### VQA Datasets - Loader Status

- [x] `load_rs_vqa` - Implemented ✅
- [x] `load_rsvqa_hr` - Implemented ✅
- [x] `load_xlrs_bench` - Implemented ✅
- [ ] `load_floodnet_vqa` - TODO (gated dataset)

______________________________________________________________________

## Instruction Tuning Datasets

| Dataset                | HuggingFace ID                                | Size    | Status        | Keys/Notes                                                                                                         |
| ---------------------- | --------------------------------------------- | ------- | ------------- | ------------------------------------------------------------------------------------------------------------------ |
| RS Visual Instructions | `AdaptLLM/remote-sensing-visual-instructions` | 36.4K   | ✅ Works      | `images`, `messages`                                                                                               |
| GeoChat Instruct       | `MBZUAI/GeoChat_Instruct`                     | 318K    | ✅ Works      | Instruction tuning format                                                                                          |
| GeoChat Bench          | `MBZUAI/GeoChat-Bench`                        | varies  | ✅ Works      | Evaluation benchmark                                                                                               |
| ThinkGeo               | `MBZUAI/ThinkGeo`                             | 311     | ⚠️ Partial    | Tool-augmented agent eval. `image` only in streaming                                                               |
| GeoZero Eval           | `hjvsl/GeoZero_Eval_Datasets`                 | 26K     | ✅ Works      | `conversations`, `images`. Eval dataset for GeoZero.                                                               |
| FIT-RS                 | `ll-13/FIT-RS`                                | 1.8M    | 🔧 Needs Work | Fine-grained RS instruction tuning. Semantic relationships.                                                        |
| UAVIT-1M               | `ZhanYang-nwpu/UAVIT-1M`                      | 1.24M   | ✅ Works      | UAV instruction tuning. 11 tasks, 789K images.                                                                     |
| UAVBench               | `ZhanYang-nwpu/UAVBench`                      | 1M      | ✅ Works      | UAV benchmark. 966K samples, 43 test units, 10 tasks.                                                              |
| RS SFT Data            | `Qingyun/remote-sensing-sft-data`             | varies  | 🔧 Needs Work | Comprehensive RS SFT collection. Gated access.                                                                     |
| LMMRotate SFT          | `Qingyun/lmmrotate-sft-data`                  | 100K-1M | 🔧 Needs Work | Oriented detection SFT (DOTA, DIOR-R, FAIR1M, SRSDD, RSAR). Gated.                                                 |
| GeoPixInstruct         | `Norman-ou/GeoPixInstruct-Anno`               | 58K     | ✅ Works      | Pixel-level RS instructions. 6 splits.                                                                             |
| EarthMind Data         | `sy1998/EarthMind-data`                       | varies  | 🔧 Needs Work | EarthMind training data. Viewer unavailable.                                                                       |
| RSVLM-SFT              | `FelixKAI/RSVLM-SFT`                          | varies  | 🔧 Needs Work | MF-RSVLM SFT data. Viewer unavailable.                                                                             |
| GeoSeg-Bench           | `nishuo1999/GeoSeg-Bench`                     | varies  | ⚠️ Partial    | Segmentation benchmark for UniGeoSeg. Empty readme.                                                                |
| TEOChatlas             | `jirvin16/TEOChatlas`                         | 500K+   | 🔧 Needs Work | Temporal EO data (fMoW, xBD, S2Looking, QFabric). Uses deprecated loading script, 125GB.                           |
| EarthReason            | `earth-insights/EarthReason`                  | 1-10K   | 🔧 Needs Work | Geospatial pixel reasoning. Zip files, needs custom loader for images+questions+answers.                           |
| EarthDial-Dataset      | `akshaydudhane/EarthDial-Dataset`             | 82K+    | ✅ Works      | Multi-config eval dataset. Configs: Classification, GeoChat_Bench, Detection, Region_captioning, Image_captioning. |

### Instruction Datasets - Loader Status

- [x] `load_rs_visual_instructions` - Implemented ✅
- [x] `load_geochat_instruct` - Implemented ✅
- [x] `load_geochat_bench` - Implemented ✅
- [ ] `load_teochatlas` - TODO (requires custom loader, deprecated `trust_remote_code`)
- [ ] `load_earthreason` - TODO (zip files, needs extraction + custom loader)
- [x] `load_earthdial_dataset` - Implemented ✅

______________________________________________________________________

## Change Detection / Captioning Datasets

| Dataset         | HuggingFace ID      | Size      | Status           | Keys/Notes                                                                                    |
| --------------- | ------------------- | --------- | ---------------- | --------------------------------------------------------------------------------------------- |
| LEVIR-CC        | `DoryokuX/LEVIR-CC` | 600       | ⚠️ Partial       | `image`, `label` (no captions in viewer)                                                      |
| LEVIR-CC (full) | `lcybuaa/LEVIR-CC`  | 10K pairs | 🔧 Needs Work    | Bi-temporal images + 50K captions in JSON                                                     |
| LEVIR-MCI       | `lcybuaa/LEVIR-MCI` | varies    | 🔧 Needs Work    | Bi-temporal + masks + captions                                                                |
| SYSU-CD         | `ericyu/SYSU_CD`    | varies    | ⚠️ Partial       | `imageA`, `imageB`, `label` (segmentation masks, no captions)                                 |
| RSCC            | `BiliSakura/RSCC`   | 62.3K     | ✅ Works         | Remote sensing change caption (xBD+EBD). Bi-temporal + captions. Configs: `EBD`, `benchmark`. |
| Dubai-CC        | N/A                 | N/A       | ❌ Not Available | Not found on HuggingFace                                                                      |
| MUDS            | N/A                 | N/A       | ❌ Not Available | Not found on HuggingFace                                                                      |

### Change Detection - Loader Status

- [ ] `load_levir_cc` - TODO (needs custom loader for bi-temporal pairs + JSON captions)
- [ ] `load_levir_mci` - TODO (multi-task: detection + captioning)
- [x] `load_rscc` - Implemented ✅

______________________________________________________________________

## Disaster Assessment Datasets

| Dataset                    | HuggingFace ID                          | Size       | Status        | Keys/Notes                                                                                        |
| -------------------------- | --------------------------------------- | ---------- | ------------- | ------------------------------------------------------------------------------------------------- |
| DisasterM3                 | `Kingdrone-Junjue/DisasterM3`           | 27K images | ✅ Works      | VLM disaster dataset (NeurIPS 2025). 123K instructions, 9 tasks, 10 disaster types. Multi-sensor. |
| Disaster Assessment (Qwen) | `Tushar365/disaster-assessment-qwen2vl` | 20         | ⚠️ Partial    | Very small, Qwen format                                                                           |
| xBD                        | Various                                 | varies     | 🔧 Needs Work | Disaster damage, mostly segmentation                                                              |

### Disaster - Loader Status

- [x] `load_disaster_m3` - Implemented ✅
- [ ] `load_disaster_assessment` - TODO (small dataset, may need custom preprocessing)

______________________________________________________________________

## Large-Scale Pre-training Datasets

| Dataset     | HuggingFace ID              | Size   | Status        | Keys/Notes                                                         |
| ----------- | --------------------------- | ------ | ------------- | ------------------------------------------------------------------ |
| RS5M        | `omlab/RS5M`                | 7.25M  | ✅ Works      | Image-caption pairs for CLIP-style training                        |
| DE-Dataset  | `earth-insights/DE-Dataset` | 261K   | ✅ Works      | Existing loader                                                    |
| VHM VersaD  | `FitzPC/VHM_VersaD`         | 4.1M   | ✅ Works      | Webdataset: `jpg`, `__key__`. Large-scale RS pre-training.         |
| VHM SFT     | `FitzPC/VHM_dataset_sft`    | varies | 🔧 Needs Work | SFT dataset for VHM model. Viewer unavailable.                     |
| DynamicVL   | `weihao1115/dvl_suite`      | varies | ⚠️ Partial    | Urban dynamics benchmark. `image` only in streaming. NeurIPS 2025. |
| BigEarthNet | Various                     | 519K+  | 🔧 Needs Work | Classification, no captions                                        |

______________________________________________________________________

## Grounding / Referring Expression Datasets

| Dataset   | HuggingFace ID         | Size   | Status        | Keys/Notes                                                                         |
| --------- | ---------------------- | ------ | ------------- | ---------------------------------------------------------------------------------- |
| LRS-GRO   | `HappyBug/LRS-GRO`     | varies | ✅ Works      | `question`, `ground_truth`, `bbox`, `category`. Scene-level VQA + grounding.       |
| DIOR-RSVG | `danielz01/DIOR-RSVG`  | 24.3K  | 🔧 Needs Work | Requires access agreement                                                          |
| RefSegRS  | `JessicaYuan/RefSegRS` | 8.8K   | ⚠️ Partial    | Referring segmentation. `image`, `label` only (no text expressions in HF version). |

______________________________________________________________________

## Summary

### Currently Implemented (in `goldeneye.datasets`)

```python
# Captioning
from goldeneye.datasets import (
    load_rsicd, load_ucm_captions, load_nwpu_captions, load_fair1m_caption,
    load_rs5m, load_rscid_captions, load_sydney_captions,
    load_landsat30_au, load_rsteller,
)

# VQA & Benchmarks
from goldeneye.datasets import (
    load_rs_vqa, load_rsvqa_hr, load_xlrs_bench, load_vrsbench,
    load_lrs_vqa, load_xhrbench, load_lrs_gro, load_de_dataset,
)

# Instruction Tuning
from goldeneye.datasets import (
    load_rs_visual_instructions, load_geochat_instruct, load_geochat_bench,
    load_geozero_eval, load_uavit_1m, load_uavbench, load_geopix_instruct,
    load_earthdial_dataset,
)

# Change Captioning
from goldeneye.datasets import load_rscc

# Disaster
from goldeneye.datasets import load_disaster_m3

# Pre-training
from goldeneye.datasets import load_vhm_versad
```

### Recommended for Fine-tuning

| Use Case           | Recommended Dataset                      | Size      |
| ------------------ | ---------------------------------------- | --------- |
| Image Captioning   | RSICD, FAIR1M Caption                    | 10-22K    |
| VQA                | RSVQA-HR                                 | 358K      |
| Instruction Tuning | RS Visual Instructions, GeoChat Instruct | 36-318K   |
| Pre-training       | RS5M                                     | 7.25M     |
| Change Detection   | LEVIR-CC (needs custom loader)           | 10K pairs |

### Priority TODO

1. [x] Add `load_fair1m_caption` - Implemented ✅
1. [x] Add `load_rs5m` - Implemented ✅
1. [ ] Add `load_levir_cc` - Change captioning (unique task)
1. [ ] Add `load_teochatlas` - Temporal EO (125GB, complex loader)
1. [ ] Investigate FloodNet VQA - Disaster domain (gated)
