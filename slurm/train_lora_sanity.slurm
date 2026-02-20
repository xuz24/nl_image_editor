#!/bin/bash
#SBATCH --job-name=pico-banana-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pico-banana-test-%j.out
#SBATCH --error=logs/pico-banana-test-%j.err
#SBATCH --output=output_%j.txt
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

TRAIN_CFG="configs/training_config.yaml"
if [[ -f "${TRAIN_CFG}.bak" ]]; then
  echo "Previous backup ${TRAIN_CFG}.bak exists; refusing to overwrite." >&2
  exit 1
fi
if [[ -f "$TRAIN_CFG" ]]; then
  cp "$TRAIN_CFG" "${TRAIN_CFG}.bak"
fi
cleanup_cfg() {
  if [[ -f "${TRAIN_CFG}.bak" ]]; then
    mv "${TRAIN_CFG}.bak" "$TRAIN_CFG"
  fi
}
trap cleanup_cfg EXIT

# Keep the sanity check tiny so it fits in 30 minutes on a single GPU.
cat <<'EOF' > "$TRAIN_CFG"
resolution: 256
batch_size: 1
learning_rate: 1e-4
lora_rank: 4
num_training_steps: 50
save_every: 25
EOF

# Download a small slice of the dataset if needed.
if [[ ! -d data/pico-banana || -z "$(ls -A data/pico-banana 2>/dev/null)" ]]; then
  bash scripts/download_pico_banana.sh --max-samples 32
fi

# Run the shortened training loop.
python training/train_lora.py
