import goldeneye

print("Available models:", goldeneye.list_models())

model = goldeneye.load_agent("GeoR1", device="cpu")
print(f"Loaded agent: {model.codename}")

response = model("path/to/image.jpg", "What is shown in this satellite image?")
print(f"Response: {response}")
