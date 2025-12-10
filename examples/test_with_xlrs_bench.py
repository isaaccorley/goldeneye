import geovllm
from geovllm.datasets import stream_xlrs_bench

model = geovllm.load_model("GeoR1", device="cpu")

print("Streaming XLRS-Bench-lite dataset (no download required)...")
for i, sample in enumerate(stream_xlrs_bench(split="train")):
    if i >= 3:
        break

    image = sample["image"]
    question = sample.get("question", "Describe this image.")

    print(f"\nSample {i + 1}:")
    print(f"Question: {question}")

    response = model(image, question)
    print(f"Response: {response}")
