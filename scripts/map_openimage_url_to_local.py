#!/usr/bin/env python3
"""
Map Open Images URLs (single-turn or multi-turn) to local Open Images images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map Open Images URLs to local image paths.")
    parser.add_argument("--metadata-csv", required=True, help="Open Images CSV mapping ImageID to OriginalURL.")
    parser.add_argument("--jsonl-in", required=True, help="Input Pico-Banana jsonl file.")
    parser.add_argument("--jsonl-out", required=True, help="Output jsonl path with local_input_image field.")
    parser.add_argument("--image-root", required=True, help="Root folder containing extracted Open Images .jpg files.")
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Enable multi-turn format parsing via files[].id == original_input_image.",
    )
    return parser.parse_args()


def get_source_url(data: dict, multi_turn: bool) -> str | None:
    if not multi_turn:
        return data.get("open_image_input_url")

    files = data.get("files", [])
    for item in files:
        if item.get("id") == "original_input_image":
            return item.get("url")
    return None


def main() -> None:
    args = parse_args()

    print("Loading metadata mapping (URL -> ImageID)...")
    url_to_id = {}
    with open(args.metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["OriginalURL"].strip()
            img_id = row["ImageID"].strip()
            url_to_id[url] = img_id
    print(f"Loaded {len(url_to_id):,} entries from metadata CSV")

    print(f"Indexing local .jpg images under {args.image_root} ...")
    local_id_to_path = {}
    for root, _, files in tqdm(os.walk(args.image_root), desc="Scanning subfolders"):
        for file_name in files:
            if file_name.lower().endswith(".jpg"):
                image_id = os.path.splitext(file_name)[0]
                local_id_to_path[image_id] = os.path.join(root, file_name)
    print(f"Indexed {len(local_id_to_path):,} local image files")

    count_matched = 0
    count_url_not_found = 0
    count_file_missing = 0

    print("Mapping input URLs to local files...")
    with open(args.jsonl_in, "r", encoding="utf-8") as fin, open(args.jsonl_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Processing JSONL"):
            if not line.strip():
                continue
            data = json.loads(line)
            url = get_source_url(data, args.multi_turn)

            if not url:
                data["local_input_image"] = None
                count_url_not_found += 1
                fout.write(json.dumps(data) + "\n")
                continue

            image_id = url_to_id.get(url)
            if not image_id:
                data["local_input_image"] = None
                count_url_not_found += 1
            else:
                local_path = local_id_to_path.get(image_id)
                if local_path and os.path.exists(local_path):
                    data["local_input_image"] = local_path
                    count_matched += 1
                else:
                    data["local_input_image"] = None
                    count_file_missing += 1

            fout.write(json.dumps(data) + "\n")

    print("\nMapping complete.")
    print(f"  Matched successfully: {count_matched:,}")
    print(f"  URL not found in metadata: {count_url_not_found:,}")
    print(f"  ImageID found but file missing locally: {count_file_missing:,}")
    print(f"\nOutput saved to: {args.jsonl_out}")


if __name__ == "__main__":
    main()
