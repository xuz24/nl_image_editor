#!/bin/bash
#SBATCH --job-name=pico-openimages-map
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pico-openimages-map-%j.out
#SBATCH --error=logs/pico-openimages-map-%j.err

set -Eeuo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
cd "${REPO_DIR}"
mkdir -p logs

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Cluster modules (adjust if your cluster uses different names)
module purge
module load python/3.10

if [[ ! -d ".venv" ]]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --progress-bar off

# Comma-separated list of JSONLs to map.
# Example:
#   sbatch --export=ALL,JSONL_LIST="data/pico-banana/.cache/sft.jsonl,data/pico-banana/.cache/preference.jsonl" slurm/download_openimages_map.slurm
JSONL_LIST="${JSONL_LIST:-data/pico-banana/.cache/sft.jsonl}"
WORK_DIR="${WORK_DIR:-data/openimages}"
MULTI_TURN="${MULTI_TURN:-0}"     # set to 1 for multi-turn jsonl format
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}" # set to 1 if Open Images assets already exist

IFS=',' read -r -a JSONL_ARRAY <<< "${JSONL_LIST}"

CMD=(bash scripts/download_openimages_and_map.sh --work-dir "${WORK_DIR}")
for jsonl in "${JSONL_ARRAY[@]}"; do
  # Trim leading/trailing whitespace
  jsonl="$(echo "${jsonl}" | xargs)"
  [[ -n "${jsonl}" ]] && CMD+=(--jsonl "${jsonl}")
done

if [[ "${MULTI_TURN}" == "1" ]]; then
  CMD+=(--multi-turn)
fi
if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
  CMD+=(--skip-download)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Job completed successfully."
