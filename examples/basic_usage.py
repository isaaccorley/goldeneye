import goldeneye

print("Available models:", goldeneye.assets())

model = goldeneye.dispatch_agent("Geo-R1-3B-GRPO-REC-5shot", device="cpu")
print(f"Loaded agent: {model.codename}")

response = model("path/to/image.jpg", "What is shown in this satellite image?")
print(f"Response: {response}")
