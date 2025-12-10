import os
import torch
from pathlib import Path

from PIL import Image

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import geovllm


def main() -> None:
    print("Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = "cpu"
    else:
        print(f"CUDA is available. Using device: cuda")
        device = "cuda"
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print("\nCreating dummy image...")
    dummy_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    dummy_path = Path("/tmp/dummy_test.png")
    dummy_image.save(dummy_path)
    print(f"Saved dummy image to {dummy_path}")

    print("\nLoading ZoomEarth model...")
    try:
        model = geovllm.load_model("ZoomEarth-3B", device=device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    print("\nRunning inference...")
    prompt = "What do you see in this image?"
    try:
        output = model(dummy_path, prompt, max_new_tokens=64)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")
        print(f"\nOutput length: {len(output)} characters")
    except Exception as e:
        print(f"Error during inference: {e}")
        raise

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
