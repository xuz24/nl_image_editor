#!/bin/bash
#SBATCH --job-name=pico-output-download
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pico-output-download-%j.out
#SBATCH --error=logs/pico-output-download-%j.err

set -Eeuo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
cd "${REPO_DIR}"
mkdir -p logs

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

module purge
module load python/3.10

if [[ ! -d ".venv" ]]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --progress-bar off

JSONL_PATH="${JSONL_PATH:-data/pico-banana/.cache/sft.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/pico-banana}"
BASE_URL="${BASE_URL:-https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb}"
WORKERS="${WORKERS:-16}"
RETRIES="${RETRIES:-5}"
TIMEOUT="${TIMEOUT:-60}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"

CMD=(
  bash scripts/download_output_images.sh
  --jsonl "${JSONL_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --base-url "${BASE_URL}"
  --workers "${WORKERS}"
  --retries "${RETRIES}"
  --timeout "${TIMEOUT}"
  --max-samples "${MAX_SAMPLES}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Job completed successfully."
