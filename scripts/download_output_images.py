#!/usr/bin/env python3
"""Download Pico-Banana edited output images referenced in a JSONL file."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = "pico-banana-output-downloader/1.0"
DEFAULT_BASE_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Mapped JSONL path (contains output_image field).")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where output images are stored (keeps output_image relative paths).",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Pico-Banana CDN base URL.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel download workers.")
    parser.add_argument("--retries", type=int, default=5, help="Retries per image.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--max-samples", type=int, default=0, help="Download at most N images (0 = all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def iter_output_paths(jsonl_path: Path, max_samples: int):
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            rel = data.get("output_image")
            if not rel:
                continue
            yield rel.lstrip("/")
            count += 1
            if max_samples and count >= max_samples:
                break


def download_file(url: str, dest: Path, retries: int, timeout: int) -> str:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as response:
                data = response.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return "ok"
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_err = exc
            time.sleep(min(10, 2**attempt))
    return f"failed: {last_err}"


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    rel_paths = list(iter_output_paths(jsonl_path, args.max_samples))
    if not rel_paths:
        raise RuntimeError("No output_image entries found.")

    print(f"[info] Downloading {len(rel_paths)} output images to {output_root}")
    ok = 0
    skipped = 0
    failed = 0

    def submit_task(rel_path: str):
        dest = output_root / rel_path
        if dest.exists() and not args.overwrite:
            return "skipped"
        url = f"{args.base_url.rstrip('/')}/{rel_path}"
        return download_file(url, dest, args.retries, args.timeout)

    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(submit_task, rel_paths):
            if result == "ok":
                ok += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1

    print(f"[summary] ok={ok} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
