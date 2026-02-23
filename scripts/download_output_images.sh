#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/download_output_images.sh --jsonl PATH [options]

Download Pico-Banana edited output images referenced by a JSONL file.
Images are saved under output-root while preserving each entry's output_image path.

Options:
  --jsonl PATH          Input JSONL containing output_image entries (required)
  --output-root DIR     Save root for edited images (default: data/pico-banana-outputs)
  --base-url URL        CDN base URL (default: Apple Pico-Banana CDN)
  --workers N           Parallel workers (default: 16)
  --retries N           Retries per image (default: 5)
  --timeout SEC         HTTP timeout seconds (default: 60)
  --max-samples N       Download first N entries only (default: 0 = all)
  --overwrite           Re-download existing files
  -h, --help            Show this help
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JSONL=""
OUTPUT_ROOT="data/pico-banana-outputs"
BASE_URL="https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb"
WORKERS=16
RETRIES=5
TIMEOUT=60
MAX_SAMPLES=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jsonl) JSONL="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$JSONL" ]]; then
  echo "Error: --jsonl is required."
  usage
  exit 1
fi

PY_ARGS=(
  --jsonl "$JSONL"
  --output-root "$OUTPUT_ROOT"
  --base-url "$BASE_URL"
  --workers "$WORKERS"
  --retries "$RETRIES"
  --timeout "$TIMEOUT"
  --max-samples "$MAX_SAMPLES"
)

if [[ $OVERWRITE -eq 1 ]]; then
  PY_ARGS+=(--overwrite)
fi

python3 "$SCRIPT_DIR/download_output_images.py" "${PY_ARGS[@]}"
