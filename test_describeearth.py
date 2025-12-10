#!/usr/bin/env python3
from pathlib import Path

import geovllm
from PIL import Image


def create_dummy_image(path: Path, size: tuple[int, int] = (512, 512)) -> Path:
    image = Image.new("RGB", size, color=(128, 128, 128))
    image.save(path)
    return path


def main():
    print("Testing DescribeEarth model on CUDA...")

    dummy_image_path = Path("/tmp/dummy_test.png")
    create_dummy_image(dummy_image_path, size=(512, 512))
    print(f"Created dummy image: {dummy_image_path}")

    print("\nLoading DescribeEarth model on CUDA...")
    try:
        model = geovllm.load_model("DescribeEarth", device="cuda")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    prompt = "Describe what you see in this satellite image."
    print(f"\nRunning inference with prompt: '{prompt}'")
    print("This may take a while on first run (downloading model)...")

    try:
        output = model(dummy_image_path, prompt, max_new_tokens=64)
        print(f"\n✅ Inference successful!")
        print(f"Output: {output}")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
