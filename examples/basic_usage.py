import geovllm

print("Available models:", geovllm.list_models())

model = geovllm.load_model("GeoR1", device="cpu")
print(f"Loaded model: {model.model_name}")

response = model("path/to/image.jpg", "What is shown in this satellite image?")
print(f"Response: {response}")
