"""Script to analyze all goldeneye datasets and document their structure.

Handles:
- Direct PIL images
- Raw bytes that need decoding (RS5M)
- Image paths (documents but notes images need separate download)
- Skips first N samples for datasets with paper figures
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

# Create output directory for sample images
OUTPUT_DIR = Path("dataset_samples")
OUTPUT_DIR.mkdir(exist_ok=True)


# Dataset configurations
# skip_samples: number of samples to skip (e.g., paper figures)
# bytes_image_key: key containing raw image bytes to decode
# has_direct_images: False if images are paths that need external loading
# image_note: additional notes about image access
DATASET_CONFIG: dict[str, dict] = {
    "sarlang": {
        "skip_samples": 2,  # First 2 are paper figures
        "note": "First 2 samples are paper figures, actual SAR images start at sample 2",
    },
    "rs5m": {
        "bytes_image_key": "img_content",
        "bytes_decode": {"caption": "utf-8", "img_name": "utf-8"},
        "note": "Image stored as raw bytes in 'img_content', decode with PIL.Image.open(BytesIO(bytes))",
    },
    "disaster-m3": {
        "note": "Only contains binary masks (grayscale mode=L), actual RGB images stored externally",
        "has_direct_images": False,
    },
    # Datasets with image paths only
    "geochat-bench": {
        "has_direct_images": False,
        "note": "Image column contains paths, need to download images separately from MBZUAI/GeoChat_Bench repo",
    },
    "vrsbench": {
        "has_direct_images": False,
        "note": "Image column contains filenames, images in Images_train.zip/Images_val.zip",
    },
    "xhrbench": {
        "has_direct_images": False,
        "note": "Images stored in 'images' list with 'path' field, need external download",
    },
    "lrs-vqa": {
        "has_direct_images": False,
        "note": "Image column contains paths like 'LRS_VQA/image/15565.tif'",
    },
    "rs-visual-instructions": {
        "has_direct_images": False,
        "note": "Images stored as paths in 'images' list",
    },
    "uavit-1m": {
        "has_direct_images": False,
        "note": "Image column contains paths like 'ERA/train/Harvesting/...'",
    },
    "uavbench": {
        "has_direct_images": False,
        "note": "Image column contains paths",
    },
    "geopix-instruct": {
        "has_direct_images": False,
        "note": "image_path column contains paths like 'SIOR/JPEGImages-trainval/05114.jpg'",
    },
    "rscc": {
        "has_direct_images": False,
        "note": "pre_image and post_image contain absolute paths (need local dataset)",
    },
    "changechat": {
        "has_direct_images": False,
        "note": "Image column contains list of paths like ['train/A/train_000001.png', 'train/B/...']",
    },
    "refgeo": {
        "has_direct_images": False,
        "note": "image_id column contains IDs, not direct images",
    },
    "pregres": {
        "has_direct_images": False,
        "note": "Image column contains paths like 'NWPU-RESISC45/airplane/airplane_001.jpg'",
    },
    "gres": {
        "has_direct_images": False,
        "note": "Image paths embedded in shapes[].image_name",
    },
    "evattrs": {
        "has_direct_images": False,
        "note": "Only has img_id (integer), images stored externally",
    },
}

# All datasets from the registry (skip geotext1652 - hangs)
DATASETS = [
    "xlrs-bench",
    "de-dataset",
    "urbench",
    "geochat-bench",
    "vrsbench",
    "xhrbench",
    "geozero-eval",
    "rsicd",
    "ucm-captions",
    "sydney-captions",
    "nwpu-captions",
    "fair1m-caption",
    "rs5m",
    "rscid-captions",
    "landsat-captions",
    "rsteller",
    "rs-vqa",
    "rsvqa-hr",
    "geochat-instruct",
    "lrs-vqa",
    "lrs-gro",
    "rs-visual-instructions",
    "uavit-1m",
    "uavbench",
    "geopix-instruct",
    "earthdial",
    "rs-eot",
    "rscc",
    "changechat",
    # "geotext1652",  # SKIP - hangs
    "refgeo",
    "pregres",
    "gres",
    "evattrs",
    "sarlang",
    "disaster-m3",
    "vhm-versad",
]


def truncate_value(value: object, max_length: int = 200) -> str:
    """Truncate a value for display purposes."""
    if isinstance(value, Image.Image):
        return f"<PIL.Image mode='{value.mode}' size={value.size}>"
    if isinstance(value, bytes):
        return f"<bytes length={len(value)}>"
    if isinstance(value, dict):
        return f"<dict keys: {list(value.keys())}>"
    if isinstance(value, list):
        if len(value) > 3:
            # Show first item type
            first_type = type(value[0]).__name__ if value else "empty"
            return f"<list len={len(value)}, item_type={first_type}>"
        return str(value)[:max_length]

    str_value = str(value)
    if len(str_value) > max_length:
        return str_value[:max_length] + "..."
    return str_value


def save_image(
    image: Image.Image | None, dataset_name: str, key_name: str
) -> str | None:
    """Save an image and return the path."""
    if image is None:
        return None

    if not isinstance(image, Image.Image):
        return None

    # Convert to RGB if necessary for saving as PNG
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    elif image.mode not in ("RGB", "L"):
        try:
            image = image.convert("RGB")
        except Exception:
            pass

    filename = f"{dataset_name}_{key_name}.png"
    filepath = OUTPUT_DIR / filename
    try:
        image.save(filepath)
        return str(filepath)
    except Exception as e:
        return f"<Failed to save: {e}>"


def analyze_dataset(dataset_name: str) -> dict:
    """Analyze a single dataset and return its structure."""
    import goldeneye.datasets as datasets

    config = DATASET_CONFIG.get(dataset_name, {})

    result = {
        "name": dataset_name,
        "status": "success",
        "keys": {},
        "saved_images": [],
        "error": None,
        "note": config.get("note"),
        "has_direct_images": config.get("has_direct_images", True),
    }

    try:
        # Load the dataset with streaming
        ds = datasets.load_dataset(dataset_name, streaming=True)

        # Handle DatasetDict vs Dataset
        if hasattr(ds, "keys") and callable(ds.keys):
            splits = list(ds.keys())
            result["splits"] = splits
            # Prefer 'train' split if available
            if "train" in splits:
                ds = ds["train"]
            else:
                ds = ds[splits[0]]
        else:
            result["splits"] = ["default"]

        # Skip samples if configured (e.g., paper figures)
        skip_n = config.get("skip_samples", 0)
        sample = None
        for i, s in enumerate(ds):
            if i >= skip_n:
                sample = s
                break

        if sample is None:
            result["status"] = "error"
            result["error"] = "No samples found"
            return result

        # Analyze each key
        for key, value in sample.items():
            key_info = {
                "type": type(value).__name__,
                "sample": truncate_value(value),
            }

            # Handle bytes decoding
            if isinstance(value, bytes):
                # Check if this is an image bytes field
                if key == config.get("bytes_image_key"):
                    try:
                        img = Image.open(io.BytesIO(value))
                        key_info["decoded_image"] = f"mode={img.mode}, size={img.size}"
                        saved = save_image(img, dataset_name.replace("-", "_"), key)
                        if saved and not saved.startswith("<Failed"):
                            result["saved_images"].append({"key": key, "path": saved})
                    except Exception as e:
                        key_info["decode_error"] = str(e)
                # Check if text bytes
                elif key in config.get("bytes_decode", {}):
                    encoding = config["bytes_decode"][key]
                    try:
                        decoded = value.decode(encoding)
                        key_info["decoded_text"] = decoded[:100]
                    except Exception:
                        pass

            # Check if it's a PIL Image and save it
            elif isinstance(value, Image.Image):
                saved = save_image(value, dataset_name.replace("-", "_"), key)
                if saved and not saved.startswith("<Failed"):
                    result["saved_images"].append({"key": key, "path": saved})
                key_info["image_mode"] = value.mode
                key_info["image_size"] = value.size

            # Check for list of images
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], Image.Image):
                    # Save first image from list
                    saved = save_image(
                        value[0], dataset_name.replace("-", "_"), f"{key}_0"
                    )
                    if saved and not saved.startswith("<Failed"):
                        result["saved_images"].append(
                            {"key": f"{key}[0]", "path": saved}
                        )
                    key_info["list_of_images"] = True
                    key_info["first_image_mode"] = value[0].mode
                    key_info["first_image_size"] = value[0].size

            result["keys"][key] = key_info

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main() -> None:
    """Main function to analyze all datasets."""
    results = []

    for dataset_name in DATASETS:
        print(f"Analyzing {dataset_name}...")
        result = analyze_dataset(dataset_name)
        results.append(result)
        print(f"  Status: {result['status']}")
        if result["status"] == "success":
            print(f"  Keys: {list(result['keys'].keys())}")
            if result["saved_images"]:
                paths = [img["path"] for img in result["saved_images"]]
                print(f"  Saved images: {paths}")
            if result.get("note"):
                print(f"  Note: {result['note']}")
        else:
            print(f"  Error: {result['error']}")
        print()

    # Save results to JSON
    with open(OUTPUT_DIR / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate markdown report
    generate_markdown_report(results)

    print(f"\nResults saved to {OUTPUT_DIR}/")


def generate_markdown_report(results: list[dict]) -> None:
    """Generate a markdown report from the analysis results."""
    md_lines = [
        "# Goldeneye Dataset Analysis Report",
        "",
        "This report documents the structure of each dataset in the goldeneye package.",
        "",
        "## Summary",
        "",
        f"- Total datasets analyzed: {len(results)}",
        f"- Successful: {sum(1 for r in results if r['status'] == 'success')}",
        f"- Failed: {sum(1 for r in results if r['status'] == 'error')}",
        "",
        "### Image Availability Legend",
        "",
        "- ✅ **Direct PIL Images** - Images embedded in dataset, immediately usable",
        "- 🔄 **Bytes to decode** - Raw bytes, decode with `PIL.Image.open(BytesIO(data))`",
        "- 📁 **Path references** - Paths/filenames only, need separate image download",
        "- ❌ **Error** - Failed to load dataset",
        "",
        "---",
        "",
        "## Dataset Details",
        "",
    ]

    for result in results:
        md_lines.append(f"### {result['name']}")
        md_lines.append("")

        if result["status"] == "error":
            md_lines.append("**Status:** ❌ Error")
            md_lines.append("")
            md_lines.append(f"**Error:** `{result['error']}`")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
            continue

        # Determine image status
        if result.get("saved_images"):
            md_lines.append("**Image Status:** ✅ Direct PIL Images")
        elif not result.get("has_direct_images", True):
            md_lines.append(
                "**Image Status:** 📁 Path references (need external download)"
            )
        else:
            md_lines.append("**Image Status:** ⚠️ Check notes")

        if "splits" in result:
            md_lines.append("")
            md_lines.append(f"**Splits:** {', '.join(result['splits'])}")

        if result.get("note"):
            md_lines.append("")
            md_lines.append(f"**Note:** {result['note']}")

        md_lines.append("")

        # Keys table
        md_lines.append("**Keys:**")
        md_lines.append("")
        md_lines.append("| Key | Type | Sample Value |")
        md_lines.append("|-----|------|--------------|")

        for key, info in result["keys"].items():
            sample = info["sample"].replace("|", "\\|").replace("\n", " ")
            if len(sample) > 80:
                sample = sample[:80] + "..."

            type_str = info["type"]
            # Add extra info
            if info.get("decoded_image"):
                type_str += f" → PIL ({info['decoded_image']})"
            if info.get("decoded_text"):
                sample = info["decoded_text"][:80]
            if info.get("image_mode"):
                type_str = f"PIL.Image ({info['image_mode']}, {info['image_size']})"
            if info.get("list_of_images"):
                type_str = (
                    f"list[PIL.Image] ({info['first_image_mode']}, "
                    f"{info['first_image_size']})"
                )

            md_lines.append(f"| `{key}` | {type_str} | {sample} |")

        md_lines.append("")

        # Saved images
        if result["saved_images"]:
            md_lines.append("**Sample Images Saved:**")
            md_lines.append("")
            for img_info in result["saved_images"]:
                md_lines.append(f"- `{img_info['key']}`: `{img_info['path']}`")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    # Add summary table at the end
    md_lines.append("## Quick Reference Table")
    md_lines.append("")
    md_lines.append("| Dataset | Status | Image Type | Notes |")
    md_lines.append("|---------|--------|------------|-------|")

    for r in results:
        name = r["name"]
        if r["status"] == "error":
            status = "❌"
            img_type = "-"
            notes = r["error"][:40] + "..." if len(r.get("error", "")) > 40 else r.get(
                "error", ""
            )
        else:
            status = "✅"
            if r.get("saved_images"):
                img_type = "PIL"
            elif not r.get("has_direct_images", True):
                img_type = "Path"
            else:
                img_type = "?"
            notes = (r.get("note") or "")[:40]
            if len(r.get("note", "") or "") > 40:
                notes += "..."

        md_lines.append(f"| {name} | {status} | {img_type} | {notes} |")

    with open(OUTPUT_DIR / "dataset_analysis.md", "w") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    main()
