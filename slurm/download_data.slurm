#!/bin/bash
#SBATCH --job-name=spark_example
#SBATCH --nodes=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pico-banana-test-%j.out
#SBATCH --error=logs/pico-banana-test-%j.err
#SBATCH --account=eecs476w26s_class

# Fail fast on errors/undefined vars inside the job
set -Eeuo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
mkdir -p logs

# Load whatever modules your cluster requires. Adjust/remove as needed.
module purge
module load cuda/12.1
module load python/3.10

# Optionally point to a shared HF cache so repeated jobs reuse checkpoints.
export HF_HOME="$REPO_DIR/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_MODE=offline
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create/activate the venv if it doesn't exist on the compute node yet.
if [[ ! -d ".venv" ]]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
# Keep pip quiet after the first install on shared filesystems.
python -m pip install --requirement requirements.txt --progress-bar off


bash scripts/download_pico_banana.sh --max-samples 10000 --workers 18 --output-dir /scratch/eecs476w26s_class_root/eecs476w26s_class/xuzijie/pico-banana

