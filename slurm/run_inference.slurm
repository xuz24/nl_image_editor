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
#SBATCH --account=eecs476w26s_class

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

CHECKPOINT=${CHECKPOINT:-checkpoints/lora_step_25.pt}
SOURCE_IMAGE=${SOURCE_IMAGE:-data/pico-banana/sft_0000000/source.png}
INSTRUCTION=${INSTRUCTION:-"Remove the red flag and its white pole from the upper right of the image, seamlessly extending the clear blue sky, the sandy dune with its subtle texture, and the wooden fence to fill the void, ensuring the lighting, color, and natural grain of the background are perfectly matched for a realistic and unblemished result.
"}
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
