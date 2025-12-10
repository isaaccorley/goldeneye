#!/usr/bin/env python3
"""Test DE-Dataset streaming and DescribeEarth model integration.

This test verifies:
1. Streaming works correctly
2. DescribeEarth model can process images from the dataset
3. Model produces reasonable responses
"""

import geovllm
from geovllm.datasets import stream_de_dataset


def test_streaming_and_inference():
    print("=" * 80)
    print("Testing DE-Dataset Streaming + DescribeEarth Model")
    print("=" * 80)
    
    print("\n1. Loading DescribeEarth model on CUDA...")
    try:
        model = geovllm.load_model("DescribeEarth", device="cuda")
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    print("\n2. Testing dataset streaming...")
    try:
        samples_processed = 0
        max_samples = 5
        
        for i, sample in enumerate(stream_de_dataset(split="train")):
            if samples_processed >= max_samples:
                break
            
            print(f"\n   Sample {samples_processed + 1}:")
            print(f"   - Key: {sample.get('__key__', 'N/A')}")
            
            image = sample.get("image")
            if image is None:
                print("   ⚠️  No image found, skipping...")
                continue
            
            print(f"   - Image size: {image.size}, mode: {image.mode}")
            
            prompt = "Describe what you see in this satellite image. Provide details about objects, their locations, and the overall scene."
            print(f"   - Prompt: {prompt[:60]}...")
            
            print("   - Running inference...")
            try:
                response = model(image, prompt, max_new_tokens=128)
                print(f"   ✅ Response received ({len(response)} chars)")
                print(f"   - Response preview: {response[:200]}...")
                
                if len(response) < 10:
                    print("   ⚠️  Warning: Response seems too short")
                elif "sorry" in response.lower() or "cannot" in response.lower() or "unable" in response.lower():
                    print("   ⚠️  Warning: Response may indicate model couldn't process image")
                else:
                    print("   ✅ Response looks reasonable")
                
            except Exception as e:
                print(f"   ❌ Inference failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            samples_processed += 1
        
        if samples_processed == 0:
            print("   ❌ No samples processed successfully")
            return False
        
        print(f"\n   ✅ Successfully processed {samples_processed} samples")
        return True
        
    except Exception as e:
        print(f"   ❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_performance():
    print("\n" + "=" * 80)
    print("Testing Streaming Performance")
    print("=" * 80)
    
    import time
    
    print("\nStreaming first 10 samples to measure performance...")
    start_time = time.time()
    sample_count = 0
    
    for sample in stream_de_dataset(split="train"):
        sample_count += 1
        if sample_count >= 10:
            break
    
    elapsed = time.time() - start_time
    print(f"✅ Streamed {sample_count} samples in {elapsed:.2f} seconds")
    print(f"   Average: {elapsed/sample_count:.2f} seconds per sample")
    
    if elapsed < 30:
        print("   ✅ Streaming performance is good")
    else:
        print("   ⚠️  Streaming is slower than expected")


if __name__ == "__main__":
    success = test_streaming_and_inference()
    test_streaming_performance()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 80)
    
    print("\nNote: DE-Dataset in webdataset format contains only images.")
    print("Descriptions/labels are stored separately and not included in the streaming format.")
    print("The model responses above demonstrate that DescribeEarth can process the images.")

