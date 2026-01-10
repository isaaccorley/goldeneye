#!/usr/bin/env python
"""Validate all models by running inference and measuring GPU memory.

Usage:
    python scripts/validate_all_models.py
    python scripts/validate_all_models.py --output-dir renders
    python scripts/validate_all_models.py --models GeoR1-3B-GRPO-REC-5shot,DescribeEarth
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import goldeneye
from goldeneye.models.utils import create_quantization_config


@dataclass
class ModelStats:
    """Statistics for a single model run."""

    model: str
    success: bool
    gpu_memory_gb: float = 0.0
    peak_memory_gb: float = 0.0
    duration_seconds: float = 0.0
    quantization: str = "none"
    error: str | None = None


def get_gpu_memory_gb() -> float:
    """Get current GPU memory allocated in GB.

    Returns
    -------
    float
        GPU memory in GB, or 0.0 if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


def get_peak_memory_gb() -> float:
    """Get peak GPU memory allocated in GB.

    Returns
    -------
    float
        Peak GPU memory in GB, or 0.0 if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_memory_stats() -> None:
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_and_run_model(
    model_name: str,
    image_path: Path,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
) -> tuple[goldeneye.Report | None, str]:
    """Load model and run inference.

    Parameters
    ----------
    model_name : str
        Name of the model to load
    image_path : Path
        Path to the input image
    prompt : str
        Prompt for the model
    device : str
        Device to run on
    max_new_tokens : int, optional
        Max tokens for generation, by default 256

    Returns
    -------
    tuple[goldeneye.Report | None, str]
        Report from model inference (or None on failure), and quantization type
    """
    # ZoomEarth requires 8-bit quantization to avoid OOM
    if "ZoomEarth" in model_name:
        quant_config = create_quantization_config(8)
        quant_type = "8bit"
    else:
        quant_config = None
        quant_type = "none"

    try:
        model = goldeneye.dispatch_agent(
            model_name, device=device, quantization_config=quant_config
        )
        report = model.recon(image_path, prompt, max_new_tokens=max_new_tokens)
        return report, quant_type
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        return None, quant_type
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def main() -> int:
    """Run validation on all models.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Validate goldeneye models")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("renders"),
        help="Output directory for rendered images",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        default=Path("assets/sample.jpg"),
        help="Input image path",
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
        default=256,
        help="Max new tokens for generation",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    if not args.image.exists():
        print(f"Image not found: {args.image}", flush=True)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        "Describe the key objects, land cover, "
        "and context in this satellite image."
    )

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = goldeneye.assets()

    print(f"\nTesting {len(model_names)} models...\n", flush=True)

    results: list[ModelStats] = []

    for model_name in model_names:
        gc.collect()
        reset_memory_stats()

        print(f"Loading {model_name}...", flush=True)
        start_time = time.time()

        report, quant_type = load_and_run_model(
            model_name, args.image, prompt, device, args.max_new_tokens
        )

        duration = time.time() - start_time
        peak_mem = get_peak_memory_gb()

        if report is None:
            print(f"  ✗ {model_name}: Failed", flush=True)
            results.append(
                ModelStats(
                    model=model_name,
                    success=False,
                    peak_memory_gb=peak_mem,
                    duration_seconds=duration,
                    quantization=quant_type,
                    error="inference failed",
                )
            )
            continue

        print(f"  Response: {report.response[:80]}...", flush=True)

        # Render and save
        rendered = goldeneye.hud.render(report, show_caption=True)
        output_path = args.output_dir / f"{model_name}.png"
        rendered.save(output_path)

        stats = ModelStats(
            model=model_name,
            success=True,
            peak_memory_gb=peak_mem,
            duration_seconds=duration,
            quantization=quant_type,
        )
        results.append(stats)

        print(
            f"  ✓ {model_name}: {peak_mem:.1f} GB, {duration:.1f}s",
            flush=True,
        )

    # Print summary table
    print(f"\n{'=' * 70}", flush=True)
    print("GPU MEMORY USAGE SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'Model':<35} {'Peak VRAM':<12} {'Time':<10} {'Status'}", flush=True)
    print("-" * 70, flush=True)

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    for r in results:
        status = "✓" if r.success else "✗"
        quant = f" ({r.quantization})" if r.quantization != "none" else ""
        print(
            f"{r.model:<35} {r.peak_memory_gb:>6.1f} GB    "
            f"{r.duration_seconds:>6.1f}s    {status}{quant}",
            flush=True,
        )

    print(f"\n{'=' * 70}", flush=True)
    print(f"Total: {len(results)} | Success: {successful} | Failed: {failed}")
    print(f"Renders saved to: {args.output_dir}/", flush=True)

    # Save JSON report
    json_path = args.output_dir / "memory_stats.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Memory stats saved to: {json_path}", flush=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
