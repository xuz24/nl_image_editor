#!/bin/bash
#SBATCH --job-name=pico-banana-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
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
export PYTHONUNBUFFERED=1

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

# Configurable overrides for submit-time tuning.
DATASET_JSONL="${DATASET_JSONL:-/scratch/eecs476w26s_class_root/eecs476w26s_class/xuzijie/pico-banana/sft_with_local_source_image_path.jsonl}"
DATASET_OUTPUT_ROOT="${DATASET_OUTPUT_ROOT:-/scratch/eecs476w26s_class_root/eecs476w26s_class/xuzijie/pico-banana}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
SAVE_EVERY="${SAVE_EVERY:-100000}"
LOG_EVERY="${LOG_EVERY:-100}"
LOSS_WINDOW="${LOSS_WINDOW:-200}"

# Optional fixed validation sample at checkpoint boundaries.
VAL_SOURCE_IMAGE="${VAL_SOURCE_IMAGE:-}"
VAL_INSTRUCTION="${VAL_INSTRUCTION:-}"
VAL_STEPS="${VAL_STEPS:-30}"
VAL_SEED="${VAL_SEED:-42}"
VAL_OUTPUT_DIR="${VAL_OUTPUT_DIR:-validation_outputs}"

cat <<EOF > "$TRAIN_CFG"
resolution: 256
batch_size: 1
learning_rate: 1e-5
lora_rank: 4
num_training_steps: ${TRAIN_STEPS}
save_every: ${SAVE_EVERY}
log_every: ${LOG_EVERY}
loss_window: ${LOSS_WINDOW}
dataset_jsonl: ${DATASET_JSONL}
dataset_output_root: ${DATASET_OUTPUT_ROOT}
val_source_image: ${VAL_SOURCE_IMAGE}
val_instruction: "${VAL_INSTRUCTION}"
val_steps: ${VAL_STEPS}
val_seed: ${VAL_SEED}
val_output_dir: ${VAL_OUTPUT_DIR}
EOF

# Run the shortened training loop.
python -u training/train_lora.py
