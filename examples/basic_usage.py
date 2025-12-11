import goldeneye

print("Available models:", goldeneye.list_models())

model = goldeneye.load_agent("GeoR1", device="cpu")
print(f"Loaded model: {model.model_name}")

response = model("path/to/image.jpg", "What is shown in this satellite image?")
print(f"Response: {response}")
