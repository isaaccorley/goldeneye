#!/usr/bin/env python3
"""Download GeoLLaVA-Data images from HuggingFace."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


def main() -> None:
    cache_dir = Path(".cache/geollava_data")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Listing files in GeoLLaVA-Data repo...")
    files = list_repo_files("initiacms/GeoLLaVA-Data", repo_type="dataset")
    zip_files = sorted([f for f in files if "geollava_raw_format_images" in f])

    print(f"Found {len(zip_files)} zip parts to download (~49GB total)")
    print("This will take a while...")
    print()

    downloaded = []
    for i, zf in enumerate(zip_files):
        print(f"Downloading {i + 1}/{len(zip_files)}: {zf}")
        try:
            path = hf_hub_download(
                "initiacms/GeoLLaVA-Data",
                zf,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
            downloaded.append(path)
            print(f"  -> {path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDownloaded {len(downloaded)} files")

    # Find the actual cache location
    if downloaded:
        print(f"\nFiles cached at: {Path(downloaded[0]).parent}")


if __name__ == "__main__":
    main()

