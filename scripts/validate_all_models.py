#!/usr/bin/env python
"""Validate all models can run inference on DE-Dataset with OOM fallback.

Usage:
    python scripts/validate_all_models.py
    python scripts/validate_all_models.py --output results.json
    python scripts/validate_all_models.py --models GeoR1-3B-GRPO-REC-5shot,DescribeEarth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

import goldeneye
from goldeneye.datasets import stream_de_dataset
from goldeneye.models.utils import create_quantization_config

SKIP_MODELS: set[str] = set()

RESPONSE_KEYWORDS = [
    "building",
    "buildings",
    "house",
    "houses",
    "road",
    "roads",
    "street",
    "streets",
    "water",
    "river",
    "lake",
    "ocean",
    "sea",
    "forest",
    "trees",
    "vegetation",
    "field",
    "fields",
    "land",
    "area",
    "region",
    "city",
    "urban",
    "rural",
    "satellite",
    "aerial",
    "image",
    "scene",
    "residential",
    "industrial",
    "agricultural",
    "mountain",
    "coast",
    "bridge",
    "airport",
    "parking",
    "vehicle",
    "ship",
    "boat",
    "green",
    "blue",
    "brown",
]


@dataclass
class ModelResult:
    model: str
    status: Literal["success", "load_failed", "inference_failed", "invalid_response", "skipped"]
    precision: str = "unknown"
    response: str | None = None
    response_length: int = 0
    keywords_found: int = 0
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class ValidationReport:
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    device: str = "unknown"
    cuda_memory_gb: float = 0.0
    total_models: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)


def get_cuda_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def fetch_sample_image(split: str = "train") -> Image.Image:
    for sample in stream_de_dataset(split=split):
        image = sample.get("image")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
    msg = "No image found in DE-Dataset stream"
    raise RuntimeError(msg)


def validate_response(response: str, prompt: str) -> tuple[bool, int]:
    if not response or len(response.strip()) < 10:
        return False, 0

    response_lower = response.lower()
    keyword_count = sum(1 for kw in RESPONSE_KEYWORDS if kw in response_lower)

    if keyword_count < 2:
        return False, keyword_count

    prompt_words = set(prompt.lower().split())
    response_words = set(response_lower.split())
    overlap = len(prompt_words.intersection(response_words))
    if overlap > len(prompt_words) * 0.8:
        return False, keyword_count

    sentences = re.split(r"[.!?]+", response)
    if len(sentences) < 1 or len(sentences[0].split()) < 3:
        return False, keyword_count

    return True, keyword_count


def load_model_with_fallback(model_name: str, device: str) -> tuple[Any, str]:
    precision_order: list[tuple[str, int | None]] = [
        ("bf16", None),
        ("8bit", 8),
        ("4bit", 4),
    ]

    for precision_name, quant_bits in precision_order:
        try:
            if quant_bits is not None:
                quant_config = create_quantization_config(quant_bits)  # type: ignore[arg-type]
                model = goldeneye.dispatch_agent(
                    model_name, device=device, quantization_config=quant_config
                )
            else:
                model = goldeneye.dispatch_agent(model_name, device=device)

            return model, precision_name

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"  OOM at {precision_name}, trying lower precision...", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            return None, f"load_error: {e}"
        except Exception as e:
            return None, f"load_error: {e}"

    return None, "all_precisions_failed"


def load_and_run_with_fallback(
    model_name: str,
    device: str,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> tuple[Any, str, str | None]:
    """Load model and run inference, retrying with lower precision on OOM.

    Returns
    -------
    tuple[Any, str, str | None]
        (response, precision, error) - response is the model output,
        precision is the quantization level used, error is None on success.
    """
    precision_order: list[tuple[str, int | None]] = [
        ("bf16", None),
        ("8bit", 8),
        ("4bit", 4),
    ]

    for precision_name, quant_bits in precision_order:
        model = None
        try:
            # Load model
            if quant_bits is not None:
                quant_config = create_quantization_config(quant_bits)  # type: ignore[arg-type]
                model = goldeneye.dispatch_agent(
                    model_name, device=device, quantization_config=quant_config
                )
            else:
                model = goldeneye.dispatch_agent(model_name, device=device)

            print(f"  Loaded at {precision_name}, running inference...", flush=True)

            # Run inference
            result = model(image, prompt, max_new_tokens=max_new_tokens)

            if hasattr(result, "response"):
                response = result.response
            elif isinstance(result, dict) and "response" in result:
                response = result["response"]
            else:
                response = str(result)

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return response, precision_name, None

        except RuntimeError as e:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"  OOM at {precision_name}, trying lower precision...", flush=True)
                continue
            return None, precision_name, f"runtime_error: {e}"
        except Exception as e:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, precision_name, f"error: {e}"

    return None, "4bit", "all_precisions_failed_oom"


def test_model(
    model_name: str, image: Image.Image, prompt: str, device: str, max_new_tokens: int = 64
) -> ModelResult:
    import time

    start_time = time.time()

    if model_name in SKIP_MODELS:
        return ModelResult(
            model=model_name,
            status="skipped",
            precision="n/a",
            error="Segmentation-only model, no captioning support",
        )

    print(f"Loading {model_name}...", flush=True)

    response, precision, error = load_and_run_with_fallback(
        model_name, device, image, prompt, max_new_tokens
    )

    if error is not None:
        return ModelResult(
            model=model_name,
            status="inference_failed" if "oom" in error.lower() else "load_failed",
            precision=precision,
            error=error,
            duration_seconds=time.time() - start_time,
        )

    is_valid, keyword_count = validate_response(response, prompt)  # type: ignore[arg-type]

    if is_valid:
        return ModelResult(
            model=model_name,
            status="success",
            precision=precision,
            response=response,
            response_length=len(response),  # type: ignore[arg-type]
            keywords_found=keyword_count,
            duration_seconds=time.time() - start_time,
        )
    else:
        return ModelResult(
            model=model_name,
            status="invalid_response",
            precision=precision,
            response=response[:200] if response else None,  # type: ignore[index]
            response_length=len(response) if response else 0,  # type: ignore[arg-type]
            keywords_found=keyword_count,
            error=f"Response validation failed (keywords={keyword_count})",
            duration_seconds=time.time() - start_time,
        )


def save_report(report: ValidationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all goldeneye models")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("validation_results.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        default=None,
        help="Comma-separated list of models to test (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens for generation",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_memory = get_cuda_memory_gb()

    print(f"Device: {device}", flush=True)
    if cuda_memory > 0:
        print(f"CUDA memory: {cuda_memory:.1f} GB", flush=True)

    report = ValidationReport(
        device=device,
        cuda_memory_gb=cuda_memory,
    )

    print("Fetching DE-Dataset sample...", flush=True)
    try:
        sample_image = fetch_sample_image()
        print(f"Sample image size: {sample_image.size}", flush=True)
    except Exception as e:
        print(f"Failed to fetch sample: {e}", flush=True)
        return 1

    prompt = "Describe the key objects, land cover, and context in this satellite image."

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = goldeneye.assets()

    report.total_models = len(model_names)

    print(f"\nTesting {len(model_names)} models...\n", flush=True)

    for model_name in model_names:
        result = test_model(model_name, sample_image, prompt, device, args.max_new_tokens)
        report.results.append(asdict(result))

        if result.status == "success":
            report.successful += 1
            status_str = f"✓ {result.precision}"
        elif result.status == "skipped":
            report.skipped += 1
            status_str = "○ skipped"
        else:
            report.failed += 1
            status_str = f"✗ {result.error}"

        print(f"  {model_name}: {status_str}", flush=True)

        save_report(report, args.output)

    print(f"\n{'=' * 60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Total:      {report.total_models}", flush=True)
    print(f"Successful: {report.successful}", flush=True)
    print(f"Failed:     {report.failed}", flush=True)
    print(f"Skipped:    {report.skipped}", flush=True)
    print(f"\nResults saved to: {args.output}", flush=True)

    non_skipped_total = report.total_models - report.skipped
    if non_skipped_total > 0 and report.successful == non_skipped_total:
        return 0
    return 1 if report.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
