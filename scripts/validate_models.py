#!/usr/bin/env python
"""Validate all models can run inference on DE-Dataset with OOM handling."""

import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

import goldeneye
from goldeneye.datasets import stream_de_dataset


def get_available_memory_gb() -> float:
    """Get available CUDA memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def estimate_model_size(model_name: str) -> float:
    """Rough estimate of model size in GB based on name."""
    if "7B" in model_name or "8K" in model_name:
        return 14.0  # 7B models ~14GB fp16
    elif "4B" in model_name:
        return 8.0   # 4B models ~8GB fp16
    elif "3B" in model_name:
        return 6.0   # 3B models ~6GB fp16
    else:
        return 4.0   # Conservative estimate


def load_model_with_fallback(model_name: str, device: str) -> tuple[Any, str]:
    """Load model with progressive fallback on OOM.
    
    Returns:
        (model, precision_used) or (None, error_msg)
    """
    available_mem = get_available_memory_gb()
    estimated_size = estimate_model_size(model_name)
    
    # Choose strategy based on memory and model size
    if available_mem < 8 or estimated_size > 10:
        # Low memory or large model - start with quantization
        precision_options = [
            ("bnb4", torch.float16, "4bit"),  # 4-bit quantization
            ("bnb8", torch.float16, "8bit"),  # 8-bit quantization
            ("fp16", torch.float16, None),    # Half precision
            ("cpu", torch.float32, None),     # CPU fallback
        ]
    elif available_mem < 16:
        # Medium memory - start with fp16
        precision_options = [
            ("fp16", torch.float16, None),    # Half precision
            ("bnb8", torch.float16, "8bit"),  # 8-bit quantization
            ("bnb4", torch.float16, "4bit"),  # 4-bit quantization
            ("cpu", torch.float32, None),     # CPU fallback
        ]
    else:
        # High memory - start with fp32
        precision_options = [
            ("fp32", torch.float32, None),    # Full precision
            ("fp16", torch.float16, None),    # Half precision
            ("bnb8", torch.float16, "8bit"),  # 8-bit quantization
            ("bnb4", torch.float16, "4bit"),  # 4-bit quantization
        ]
    
    for precision_name, dtype, quant in precision_options:
        try:
            # Skip CPU for large models
            if precision_name == "cpu" and estimated_size > 8:
                continue
                
            use_device = "cpu" if precision_name == "cpu" else device
            kwargs = {"device": use_device, "dtype": dtype}
            if quant:
                kwargs["load_in_8bit"] = quant == "8bit"
                kwargs["load_in_4bit"] = quant == "4bit"
                
            model = goldeneye.dispatch_agent(model_name, **kwargs)
            return model, f"{precision_name}@{use_device}"
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                return None, str(e)
        except Exception as e:
            return None, str(e)
    
    return None, "All loading attempts failed"


def validate_response(response: str, prompt: str) -> bool:
    """Validate response quality."""
    if not response or len(response.strip()) < 10:
        return False
        
    # Check for image-related keywords
    keywords = [
        "building", "buildings", "house", "houses", "road", "roads", "street", "streets",
        "water", "river", "lake", "ocean", "sea", "forest", "trees", "vegetation",
        "field", "fields", "land", "area", "region", "city", "urban", "rural",
        "satellite", "aerial", "image", "scene"
    ]
    
    response_lower = response.lower()
    keyword_count = sum(1 for kw in keywords if kw in response_lower)
    
    # Must contain at least 2 keywords
    if keyword_count < 2:
        return False
        
    # Not just repeating the prompt
    prompt_words = set(prompt.lower().split())
    response_words = set(response_lower.split())
    overlap = len(prompt_words.intersection(response_words))
    if overlap > len(prompt_words) * 0.8:  # >80% overlap
        return False
        
    # Not gibberish (check for reasonable sentence structure)
    sentences = re.split(r'[.!?]+', response)
    if len(sentences) < 1 or len(sentences[0].split()) < 3:
        return False
        
    return True


def test_model_inference(model_name: str, model, precision: str, sample_image: Image.Image) -> tuple[bool, str]:
    """Test inference on a sample image."""
    prompt = "Describe what you see in this satellite image."
    
    try:
        result = model(sample_image, prompt, max_new_tokens=32)
        
        if isinstance(result, dict) and "response" in result:
            response = result["response"]
        elif hasattr(result, 'response'):
            response = result.response
        else:
            response = str(result)
            
        if validate_response(response, prompt):
            return True, f"✓ Valid response ({len(response)} chars)"
        else:
            return False, f"✗ Invalid response: '{response[:100]}...'"
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, "OOM during inference"
        else:
            return False, str(e)
    except Exception as e:
        return False, str(e)


def main() -> int:
    """Main validation function."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}", flush=True)
        
        if torch.cuda.is_available():
            mem_gb = get_available_memory_gb()
            print(f"CUDA memory: {mem_gb:.1f} GB", flush=True)
        
        # Get sample from DE-Dataset
        print("Fetching DE-Dataset sample...", flush=True)
        try:
            for sample in stream_de_dataset(split="train"):
                sample_image = sample["image"]
                if isinstance(sample_image, Image.Image):
                    sample_image = sample_image.convert("RGB")
                    break
            else:
                raise RuntimeError("No image found in DE-Dataset")
        except Exception as e:
            print(f"Failed to fetch DE-Dataset sample: {e}", flush=True)
            return 1
        
        print(f"Sample image size: {sample_image.size}", flush=True)
        
        # Test all models
        models = goldeneye.assets()
        print(f"\nTesting {len(models)} models...\n", flush=True)
        
        results = []
        successful = 0
        
        for model_name in models:
            print(f"Model: {model_name}", flush=True)
            
            # Load with fallback
            model, precision = load_model_with_fallback(model_name, device)
            
            if model is None:
                results.append((model_name, False, precision))
                print(f"  Failed to load: {precision}", flush=True)
                continue
                
            # Test inference
            success, msg = test_model_inference(model_name, model, precision, sample_image)
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            results.append((model_name, success, msg))
            if success:
                successful += 1
                
            print(f"  Result: {msg}", flush=True)
            print(flush=True)
        
        # Summary
        print("SUMMARY:", flush=True)
        print(f"Total models: {len(models)}", flush=True)
        print(f"Successful: {successful}", flush=True)
        print(f"Failed: {len(models) - successful}", flush=True)
        
        if results:
            print("\nDetailed results:", flush=True)
            for name, success, msg in results:
                status = "✓" if success else "✗"
                print(f"  {status} {name}: {msg}", flush=True)
        
        return 0 if successful > 0 else 1
        
    except BrokenPipeError:
        # Handle broken pipe gracefully
        sys.stderr.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
