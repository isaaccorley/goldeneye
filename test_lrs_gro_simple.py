import os
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from PIL import Image, ImageDraw, ImageFont

import geovllm
from geovllm.datasets import stream_lrs_gro


def create_visualization(
    image: Image.Image, question: str, response: str, output_path: Path
) -> None:
    """Create a visualization with image, question, and response."""
    padding = 20
    text_padding = 15
    max_width = 1200
    font_size = 16
    line_spacing = 8

    image_width = min(image.width, max_width)
    image_height = int(image.height * (image_width / image.width))
    resized_image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        bold_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            bold_font = ImageFont.truetype("arialbd.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
            bold_font = ImageFont.load_default()

    def _wrap_text(text: str, max_width: int, font: ImageFont.FreeTypeFont) -> list[str]:
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = font.getbbox(test_line)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    question_lines = _wrap_text(f"Q: {question}", image_width - 2 * text_padding, font)
    response_lines = _wrap_text(f"A: {response}", image_width - 2 * text_padding, font)

    question_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in question_lines)
    question_height += line_spacing * (len(question_lines) - 1)
    response_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in response_lines)
    response_height += line_spacing * (len(response_lines) - 1)

    total_height = image_height + padding + question_height + padding + response_height + padding
    total_width = image_width + 2 * padding

    result = Image.new("RGB", (total_width, total_height), color="white")
    result.paste(resized_image, (padding, padding))

    y_offset = image_height + padding + text_padding
    draw = ImageDraw.Draw(result)

    for line in question_lines:
        draw.text((padding + text_padding, y_offset), line, fill="black", font=bold_font)
        y_offset += font.getbbox(line)[3] - font.getbbox(line)[1] + line_spacing

    y_offset += padding - line_spacing

    for line in response_lines:
        draw.text((padding + text_padding, y_offset), line, fill="black", font=font)
        y_offset += font.getbbox(line)[3] - font.getbbox(line)[1] + line_spacing

    result.save(output_path)
    print(f"Visualization saved to {output_path}")


def main() -> None:
    print("Loading ZoomEarth model...")
    model = geovllm.load_model("ZoomEarth-3B", device="cuda")
    print("Model loaded!")

    print("\nStreaming LRS-GRO dataset (test split)...")
    print("Note: First run will download and extract images (this may take a while)...")
    sample_count = 0
    max_samples_to_check = 100

    for sample in stream_lrs_gro(split="test", extract_images=True):
        sample_count += 1
        if sample_count % 10 == 0:
            print(f"Processed {sample_count} samples...")

        if "image" not in sample:
            if sample_count >= max_samples_to_check:
                print(f"\nChecked {max_samples_to_check} samples, no images found.")
                print("Some images may be missing from the dataset. Trying a different approach...")
                break
            continue

        print(f"\n✓ Found image in sample {sample_count}!")
        print(f"Question ID: {sample.get('question_id', 'N/A')}")
        print(f"Image name: {sample.get('image_name', 'N/A')}")

        image = sample["image"]
        question = sample.get("question", "Describe this image.")
        ground_truth = sample.get("ground_truth", "")

        print(f"Question: {question}")
        print(f"Ground truth: {ground_truth}")
        print("Running inference...")

        response = model(image, question, max_new_tokens=128)
        print(f"Response: {response[:200]}...")

        output_path = Path("lrs_gro_sample_visualization.png")
        create_visualization(image, question, response, output_path)

        print(f"\n✓ Visualization saved to {output_path}")
        break

    if sample_count == 0:
        print("No samples found in dataset.")


if __name__ == "__main__":
    main()
