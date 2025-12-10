#!/usr/bin/env python3
from geovllm.datasets import load_de_dataset, stream_de_dataset

def test_streaming():
    print("Testing DE-Dataset streaming...")
    try:
        dataset = load_de_dataset(split="train", streaming=True)
        print(f"✅ Streaming dataset loaded: {type(dataset)}")
        
        print("\nFetching first sample...")
        for i, sample in enumerate(dataset):
            print(f"Sample {i} keys: {list(sample.keys())}")
            if "image" in sample:
                print(f"  Image type: {type(sample['image'])}")
            if "description" in sample:
                print(f"  Description preview: {str(sample['description'])[:100]}...")
            break
        print("✅ Streaming test successful!")
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()

def test_non_streaming():
    print("\n\nTesting DE-Dataset non-streaming (download)...")
    print("Note: This will download the tar.gz file which may be large.")
    try:
        dataset = load_de_dataset(split="train", streaming=False)
        print(f"✅ Non-streaming dataset loaded: {type(dataset)}")
        if hasattr(dataset, "__len__"):
            print(f"  Dataset size: {len(dataset)}")
        print("✅ Non-streaming test successful!")
    except Exception as e:
        print(f"❌ Non-streaming test failed: {e}")
        import traceback
        traceback.print_exc()

def test_stream_function():
    print("\n\nTesting stream_de_dataset function...")
    try:
        for i, sample in enumerate(stream_de_dataset(split="train")):
            print(f"Sample {i} keys: {list(sample.keys())}")
            break
        print("✅ Stream function test successful!")
    except Exception as e:
        print(f"❌ Stream function test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_streaming()
    test_stream_function()
    # Uncomment to test non-streaming (downloads large tar.gz)
    # test_non_streaming()

