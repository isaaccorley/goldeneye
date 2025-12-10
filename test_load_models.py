#!/usr/bin/env python3

import geovllm


def test_load_all_models():
    models = geovllm.list_models()
    print(f"Testing {len(models)} models...\n")

    results = {}
    for model_name in models:
        print(f"Testing {model_name}...", end=" ", flush=True)
        try:
            geovllm.load_model(model_name, device="cpu")
            print("✓ Success")
            results[model_name] = ("success", None)
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[model_name] = ("failed", str(e))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    success_count = sum(1 for status, _ in results.values() if status == "success")
    print(f"Successfully loaded: {success_count}/{len(models)}")
    print(f"Failed: {len(models) - success_count}/{len(models)}")

    if success_count < len(models):
        print("\nFailed models:")
        for model_name, (status, error) in results.items():
            if status == "failed":
                print(f"  - {model_name}: {error}")

    return results


if __name__ == "__main__":
    test_load_all_models()
