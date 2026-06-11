#!/bin/bash
#SBATCH --job-name=pico-infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
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

CHECKPOINT=${CHECKPOINT:-checkpoints/lora_step_summary_13000.pt}
#SOURCE_IMAGE=${SOURCE_IMAGE:-/home/xuzijie/nl_image_editor/data/2218587291_f29ffd0401_z.jpg}
SOURCE_IMAGE=${SOURCE_IMAGE:-/home/xuzijie/nl_image_editor/data/161268561693618642-769096404.png}
#SOURCE_IMAGE=${SOURCE_IMAGE:-/scratch/eecs476w26s_class_root/eecs476w26s_class/xuzijie/pico-banana/openimage_source_images/train_0/0000e2205e460318.jpg}
#INSTRUCTION=${INSTRUCTION:-"Remove the red flag"} 
#INSTRUCTION=${INSTRUCTION:-"Adjust the overall color tone to be significantly warmer, enhancing the golden hues in the sunlight, adding a subtle orange glow to the bricks of the buildings, and shifting the blue sky to a softer, slightly more golden-hour blue, while ensuring warmer tones for a late afternoon ambiance."}
INSTRUCTION=${INSTRUCTION:-"Add dynamic, swirling Van Gogh-style brushstrokes with vibrant colors."}
# INSTRUCTION=${INSTRUCTION:-"make it a sunset, with golden hues"}
# INSTRUCTION=${INSTRUCTION:-"change to black and white"}
# INSTRUCTION=${INSTRUCTION:-"make it night time, with stars in the sky"}
# INSTRUCTION=${INSTRUCTION:-"change the season to winter time, with snow-topped skysrapers and trees, and frozen water."}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/inference_${SLURM_JOB_ID}.png}
STEPS=${STEPS:-100}
SEED=${SEED:-42}

python inference/run_inference.py \
  --source "$SOURCE_IMAGE" \
  --instruction "$INSTRUCTION" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_PATH" \
  --steps "$STEPS" \
  --seed "$SEED"
