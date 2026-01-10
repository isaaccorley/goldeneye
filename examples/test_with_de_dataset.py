#!/usr/bin/env python3
"""Example: Test DescribeEarth model with DE-Dataset.

This example shows how to stream the DE-Dataset and test it with the DescribeEarth model.
"""

import goldeneye
from goldeneye.datasets import stream_de_dataset


def main() -> None:
    print("Loading DescribeEarth model...")
    model = goldeneye.dispatch_agent("DescribeEarth", device="cuda")
    print("Model loaded!\n")

    print("Streaming DE-Dataset samples...")
    for i, sample in enumerate(stream_de_dataset(split="train")):
        if i >= 3:
            break

        image = sample.get("image")
        key = sample.get("__key__", "unknown")

        if image is None:
            print(f"Sample {i}: No image found, skipping...")
            continue

        print(f"\nSample {i}: {key}")
        print(f"Image size: {image.size}")

        prompt = "Describe what you see in this satellite image."
        print(f"Prompt: {prompt}")

        try:
            response = model(image, prompt, max_new_tokens=128)
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error during inference: {e}\n")
            continue

    print("Test completed!")


if __name__ == "__main__":
    main()
