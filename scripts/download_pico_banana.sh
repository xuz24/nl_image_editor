#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/download_pico_banana.sh [options]

Downloads the Pico-Banana single-turn (SFT) dataset released by Apple and
materializes it under data/pico-banana so the PicoBananaDataset can consume it.

Options:
  --output-dir DIR     Root directory for prepared samples (default: data/pico-banana)
  --max-samples N      Limit the number of samples to download (0 = all)
  --workers N          Parallel workers for preparation/downloads (default: half of CPU cores)
  --retries N          Download retries per file (default: 5)
  --timeout SEC        HTTP timeout per request in seconds (default: 45)
  --overwrite          Re-download and regenerate even if a sample folder exists
  -h, --help           Show this message

Example:
  bash scripts/download_pico_banana.sh --max-samples 2000 --workers 16
USAGE
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found" >&2
    exit 1
  fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DEFAULT_OUTPUT="$REPO_ROOT/data/pico-banana"
BASE_URL="https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb"
SPLIT="sft"
OUTPUT_DIR="$DEFAULT_OUTPUT"
MAX_SAMPLES=0
WORKERS=$(python3 -c 'import os; print(max(1, (os.cpu_count() or 2)//2))')
RETRIES=5
TIMEOUT=45
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --retries)
      RETRIES="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

need_cmd curl
need_cmd python3

mkdir -p "$OUTPUT_DIR"
CACHE_DIR="$OUTPUT_DIR/.cache"
mkdir -p "$CACHE_DIR"

METADATA_URL="$BASE_URL/jsonl/${SPLIT}.jsonl"
METADATA_PATH="$CACHE_DIR/${SPLIT}.jsonl"
MANIFEST_URL="$BASE_URL/manifest/${SPLIT}_manifest.txt"
MANIFEST_PATH="$CACHE_DIR/${SPLIT}_manifest.txt"

download_file() {
  local url="$1"
  local dest="$2"
  local label="$3"
  if [[ -s "$dest" && $OVERWRITE -eq 0 ]]; then
    echo "[cache] Skipping existing $label at $dest"
    return
  fi
  echo "[fetch] Downloading $label from $url"
  # Some environments ship older curl without --retry-all-errors, so keep flags portable.
  curl --fail --retry 5 --retry-connrefused --compressed -L "$url" -o "$dest.tmp"
  mv "$dest.tmp" "$dest"
}

download_file "$METADATA_URL" "$METADATA_PATH" "metadata (${SPLIT}.jsonl)"
download_file "$MANIFEST_URL" "$MANIFEST_PATH" "manifest (${SPLIT}_manifest.txt)"

echo "[prepare] Generating sample folders under $OUTPUT_DIR"
PYTHON_ARGS=()
if [[ $OVERWRITE -eq 1 ]]; then
  PYTHON_ARGS+=("--overwrite")
fi

python3 "$SCRIPT_DIR/prepare_dataset.py" \
  --split "$SPLIT" \
  --jsonl "$METADATA_PATH" \
  --output "$OUTPUT_DIR" \
  --base-url "$BASE_URL" \
  --max-samples "$MAX_SAMPLES" \
  --workers "$WORKERS" \
  --retries "$RETRIES" \
  --timeout "$TIMEOUT" \
  ${PYTHON_ARGS:+${PYTHON_ARGS[@]}}

echo "[done] Dataset ready at $OUTPUT_DIR"
