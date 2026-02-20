#!/usr/bin/env python3
"""Prepare the Pico-Banana dataset for local training.

Reads Apple's JSONL metadata + manifest entries and materializes each example
as a folder that contains:
  - source.png       (original Open Images photo)
  - target.png       (Nano-Banana edited image)
  - instruction.txt  (full edit instruction)
  - meta.json        (extra metadata for bookkeeping/debugging)

The resulting layout matches what PicoBananaDataset expects.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import io
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # dataset images are large; disable safety cap.

USER_AGENT = "pico-banana-downloader/1.0 (+https://apple-research.github.io)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="Dataset split name (e.g., sft, preference, multi-turn).")
    parser.add_argument("--jsonl", required=True, help="Path to the JSONL metadata file.")
    parser.add_argument("--output", required=True, help="Directory where prepared sample folders will be stored.")
    parser.add_argument("--base-url", required=True, help="Base CDN URL that hosts the edited images.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples to materialize (0 = all).")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download/prepare workers.")
    parser.add_argument("--retries", type=int, default=5, help="Download retries per file.")
    parser.add_argument("--timeout", type=int, default=45, help="HTTP timeout per request in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate samples even if the folder exists.")
    return parser.parse_args()


@dataclass
class SampleRecord:
    idx: int
    folder_name: str
    source_url: str
    target_rel_path: str
    instruction: str
    summarized_instruction: Optional[str]
    edit_type: Optional[str]


def iter_samples(jsonl_path: Path, max_samples: int, split: str) -> Iterable[SampleRecord]:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            folder_name = f"{split}_{idx:07d}"
            record = SampleRecord(
                idx=idx,
                folder_name=folder_name,
                source_url=data.get("open_image_input_url", "").strip(),
                target_rel_path=str(data.get("output_image", "")).strip(),
                instruction=str(data.get("text", "")).strip(),
                summarized_instruction=data.get("summarized_text"),
                edit_type=data.get("edit_type"),
            )
            yield record
            count += 1
            if max_samples and count >= max_samples:
                break


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_target_url(base_url: str, rel_path: str) -> str:
    rel_path = rel_path.lstrip("/")
    return f"{base_url.rstrip('/')}/{rel_path}"


def download_bytes(url: str, timeout: int, retries: int) -> bytes:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            wait = min(10, 2 ** attempt)
            time.sleep(wait)
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            break
    assert last_exc is not None
    raise last_exc


def save_png(image_bytes: bytes, dest: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        ensure_dir(dest.parent)
        img.save(dest, format="PNG")


def write_instruction(text: str, dest: Path) -> None:
    ensure_dir(dest.parent)
    dest.write_text(text.strip() + "\n", encoding="utf-8")


def write_meta(meta: Dict, dest: Path) -> None:
    ensure_dir(dest.parent)
    dest.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_single_sample(
    record: SampleRecord,
    output_dir: Path,
    base_url: str,
    timeout: int,
    retries: int,
    overwrite: bool,
) -> Tuple[str, Optional[str]]:
    if not record.source_url or not record.target_rel_path:
        return "skipped", "missing_url"

    sample_dir = output_dir / record.folder_name
    if sample_dir.exists() and not overwrite:
        return "skipped", None

    temp_dir = sample_dir.with_suffix(".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    ensure_dir(temp_dir)

    source_path = temp_dir / "source.png"
    target_path = temp_dir / "target.png"
    instr_path = temp_dir / "instruction.txt"
    meta_path = temp_dir / "meta.json"

    try:
        source_bytes = download_bytes(record.source_url, timeout, retries)
        save_png(source_bytes, source_path)

        target_url = resolve_target_url(base_url, record.target_rel_path)
        target_bytes = download_bytes(target_url, timeout, retries)
        save_png(target_bytes, target_path)

        write_instruction(record.instruction, instr_path)
        write_meta(
            {
                "output_image": record.target_rel_path,
                "source_url": record.source_url,
                "instruction": record.instruction,
                "summarized_instruction": record.summarized_instruction,
                "edit_type": record.edit_type,
                "sample_idx": record.idx,
            },
            meta_path,
        )

        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        temp_dir.rename(sample_dir)
        return "ok", None
    except Exception as exc:  # pylint: disable=broad-except
        shutil.rmtree(temp_dir, ignore_errors=True)
        return "failed", str(exc)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    ensure_dir(output_dir)

    jsonl_path = Path(args.jsonl).expanduser().resolve()
    if not jsonl_path.exists():
        print(f"Metadata file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    samples = list(iter_samples(jsonl_path, args.max_samples, args.split))
    if not samples:
        print("No samples found in JSONL metadata.", file=sys.stderr)
        sys.exit(1)

    stats = {"ok": 0, "failed": 0, "skipped": 0}
    errors: List[str] = []

    print(f"[info] Preparing {len(samples)} samples with {args.workers} workers...")
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample = {
            executor.submit(
                prepare_single_sample,
                record,
                output_dir,
                args.base_url,
                args.timeout,
                args.retries,
                args.overwrite,
            ): record
            for record in samples
        }

        for future in futures.as_completed(future_to_sample):
            status, error = future.result()
            stats[status] = stats.get(status, 0) + 1
            if error:
                errors.append(f"{future_to_sample[future].folder_name}: {error}")

    print(
        "[summary] ok: {ok}, skipped: {skipped}, failed: {failed}".format(
            ok=stats.get("ok", 0),
            skipped=stats.get("skipped", 0),
            failed=stats.get("failed", 0),
        )
    )

    if errors:
        print("\n[errors]")
        for err in errors[:50]:
            print(f"  - {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more.")

    if stats.get("ok", 0) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
