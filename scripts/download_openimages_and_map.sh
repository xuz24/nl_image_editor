#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/download_openimages_and_map.sh [options]

Download Open Images packed files and map Pico-Banana source URLs to local files
using scripts/map_openimage_url_to_local.py.

Options:
  --work-dir DIR          Working directory for Open Images assets
                          (default: data/openimages)
  --jsonl PATH            Input Pico-Banana JSONL file (can repeat)
  --multi-turn            Parse JSONL using multi-turn format
  --skip-download         Skip Open Images tar/csv download and extraction
  --help                  Show help

Examples:
  bash scripts/download_openimages_and_map.sh \
    --jsonl data/pico-banana/.cache/sft.jsonl

  bash scripts/download_openimages_and_map.sh \
    --jsonl data/pico-banana/.cache/sft.jsonl \
    --jsonl data/pico-banana/.cache/preference.jsonl
USAGE
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found"
    exit 1
  fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORK_DIR="$REPO_ROOT/data/openimages"
SKIP_DOWNLOAD=0
MULTI_TURN=0
JSONL_FILES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --jsonl)
      JSONL_FILES+=("$2")
      shift 2
      ;;
    --multi-turn)
      MULTI_TURN=1
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ${#JSONL_FILES[@]} -eq 0 ]]; then
  echo "Error: at least one --jsonl path is required."
  usage
  exit 1
fi

need_cmd python3
need_cmd tar
need_cmd curl

mkdir -p "$WORK_DIR"
IMAGE_ROOT="$WORK_DIR/openimage_source_images"
mkdir -p "$IMAGE_ROOT"

TAR0="$WORK_DIR/train_0.tar.gz"
TAR1="$WORK_DIR/train_1.tar.gz"
CSV_PATH="$WORK_DIR/train-images-boxable-with-rotation.csv"

if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  if command -v aws >/dev/null 2>&1; then
    [[ -f "$TAR0" ]] || aws s3 --no-sign-request --endpoint-url https://s3.amazonaws.com cp s3://open-images-dataset/tar/train_0.tar.gz "$TAR0"
    [[ -f "$TAR1" ]] || aws s3 --no-sign-request --endpoint-url https://s3.amazonaws.com cp s3://open-images-dataset/tar/train_1.tar.gz "$TAR1"
  else
    [[ -f "$TAR0" ]] || curl -fL https://storage.googleapis.com/openimages/tar/train_0.tar.gz -o "$TAR0"
    [[ -f "$TAR1" ]] || curl -fL https://storage.googleapis.com/openimages/tar/train_1.tar.gz -o "$TAR1"
  fi

  [[ -f "$CSV_PATH" ]] || curl -fL https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv -o "$CSV_PATH"

  # Extract once; marker files prevent repeated full untar.
  [[ -f "$IMAGE_ROOT/.train_0_extracted" ]] || { tar -xzf "$TAR0" -C "$IMAGE_ROOT"; touch "$IMAGE_ROOT/.train_0_extracted"; }
  [[ -f "$IMAGE_ROOT/.train_1_extracted" ]] || { tar -xzf "$TAR1" -C "$IMAGE_ROOT"; touch "$IMAGE_ROOT/.train_1_extracted"; }
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "Error: missing metadata csv at $CSV_PATH"
  exit 1
fi

if [[ ! -d "$IMAGE_ROOT" ]]; then
  echo "Error: missing extracted image root at $IMAGE_ROOT"
  exit 1
fi

for jsonl in "${JSONL_FILES[@]}"; do
  if [[ ! -f "$jsonl" ]]; then
    echo "Warning: skipping missing JSONL: $jsonl"
    continue
  fi

  out="${jsonl%.jsonl}_with_local_source_image_path.jsonl"
  cmd=(python3 "$SCRIPT_DIR/map_openimage_url_to_local.py"
    --metadata-csv "$CSV_PATH"
    --jsonl-in "$jsonl"
    --jsonl-out "$out"
    --image-root "$IMAGE_ROOT")

  if [[ $MULTI_TURN -eq 1 ]]; then
    cmd+=(--multi-turn)
  fi

  echo "Mapping $jsonl -> $out"
  "${cmd[@]}"
done

echo "Done. Local Open Images root: $IMAGE_ROOT"
