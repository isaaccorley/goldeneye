Session Overview
Goal: Make geovllm models pip-usable with inference-only support and add smoke tests to download/load/run dummy images. Vendoring missing custom code from authors’ repos where HF lacks files. Core models: EarthGPT, EarthDial, GeoChat, GeoLLaVA-8K, GeoPixel, Geo-R1, GeoZero, SAM3.
Current blockers: Tool access intermittently failing; last attempts to run terminal/file tools errored.
Research Findings
EarthDial depends on InternVL2 custom code (e.g., modeling_internvl_chat.py). HF snapshot missing these files.
GeoChat repo: https://github.com/mbzuai-oryx/GeoChat
GeoLLaVA-8K repo: https://github.com/MiliLab/GeoLLaVA-8K
GeoPixel repo: https://github.com/mbzuai-oryx/GeoPixel
InternVL license: MIT.
Geo-R1 failure: missing Qwen2_5_VLProcessor in env.
Steps Taken
Added dummy image/prompt fixtures and smoke test (tests/test_models_inference.py) gated by GEOVLLM_RUN_INFERENCE=1, using max_new_tokens=4.
Normalized wrappers to accept max_new_tokens, default 64; BaseGeoVLM signature updated.
Vendored InternVL code for EarthDial into src/geovllm/models/earthdial/internvl/... (configs, modeling, conversation, internlm2, phi3) with MIT license. Adjusted EarthDial wrapper to add vendored package to sys.path and tweak auto_map fallback.
Fixed registry syntax error.
What Worked
EarthDial imports successfully after vendoring.
Pre-commit/ty-check previously passing after wrapper updates.
What Didn't Work
Inference smoke tests failed initially: missing custom code (EarthDial), missing processors (Geo-R1), empty output (EarthGPT), unknown model_type (GeoChat/GeoLLaVA/GeoPixel).
Terminal/tool calls recently failing; unable to clone/download further repos.
Current State
Modified files: src/geovllm/models/base.py, multiple wrappers (EarthGPT, EarthDial, GeoChat, GeoLLaVA, GeoPixel, GeoZero, GeoR1), tests/conftest.py, tests/test_models_inference.py, src/geovllm/models/earthdial/... vendored InternVL tree, src/geovllm/models/registry.py (fixed syntax).
EarthDial wrapper now imports vendored InternVL; structure under earthdial/internvl/model/....
Smoke tests exist but still fail for unaddressed models.
Next Steps
Restore tool access.
Clone/download GeoChat, GeoLLaVA-8K, GeoPixel repos; vendor minimal inference code (configs/modeling/tokenization/conversation) with license headers into each model package; update wrappers to use vendored code, avoiding trust_remote_code.
Geo-R1: vendor required Qwen2_5_VL components or fallback to AutoProcessor/AutoModel; ensure processor availability.
EarthGPT: adjust prompt/generation to avoid empty output on dummy prompt.
Re-run GEOVLLM_RUN_INFERENCE=1 uv run --dev pytest tests/test_models_inference.py.
Important Context
User okay with vendoring sizeable code as long as it works.
Keep licenses from source repos in vendored files.
Tests are gated by env var to avoid costly downloads by default.