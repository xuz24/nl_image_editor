#!/bin/bash
#SBATCH --job-name=pico-infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/pico-infer-%j.out
#SBATCH --error=logs/pico-infer-%j.err

set -Eeuo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
cd "$REPO_DIR"
mkdir -p logs outputs
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_MODE=offline

module purge
module load cuda/12.1
module load python/3.10

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --progress-bar off

CHECKPOINT=${CHECKPOINT:-checkpoints/lora_step_5000.pt}
SOURCE_IMAGE=${SOURCE_IMAGE:-data/pico-banana/sft_0000000/source.png}
INSTRUCTION=${INSTRUCTION:-"Add golden sparkles around the banana"}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/inference_${SLURM_JOB_ID}.png}
STEPS=${STEPS:-50}
SEED=${SEED:-42}

python inference/run_inference.py \
  --source "$SOURCE_IMAGE" \
  --instruction "$INSTRUCTION" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_PATH" \
  --steps "$STEPS" \
  --seed "$SEED"
