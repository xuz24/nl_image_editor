#!/usr/bin/env python3
"""
Run inference with a trained Pico-Banana LoRA checkpoint.

Given a source image and an edit instruction, this script encodes the source
image, performs DDIM sampling with the trained LoRA UNet (using concatenated
source + noisy latents), and decodes the edited result.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import yaml

from models.vae import VAE
from models.clip_encoder import CLIPEncoder
from models.unet_lora import UnetLora
from diffusion.scheduler import Scheduler
from peft import LoraConfig, TaskType

LATENT_SCALING = 0.18215


def load_config(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def build_lora_config(config: dict) -> LoraConfig:
    return LoraConfig(
        r=config.get("lora_rank", 4),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get("lora_target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def preprocess_image(path: Path, resolution: int, device: torch.device) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    img = Image.open(path).convert("RGB")
    tensor = tfm(img).unsqueeze(0).to(device)
    return tensor


def save_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2
    img = tensor.mul(255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
    Image.fromarray(img).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pico-Banana LoRA inference.")
    parser.add_argument("--source", required=True, help="Path to the source image to edit.")
    parser.add_argument("--instruction", required=True, help="Edit instruction text.")
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint path (e.g., checkpoints/lora_step_5000.pt).")
    parser.add_argument("--output", default="outputs/edited.png", help="Where to save the edited image.")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM inference steps.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--config", default="configs/training_config.yaml", help="Training config file for resolution + LoRA parms.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if args.seed is not None:
        torch.manual_seed(args.seed)

    cfg = load_config(Path(args.config))
    resolution = int(cfg.get("resolution", 256))

    # Models
    vae = VAE(torch_dtype=dtype).to(device)
    unet = UnetLora(torch_dtype=dtype, in_channels=8).to(device)
    lora_config = build_lora_config(cfg)
    unet_lora = unet.get_model(lora_config)
    state = torch.load(args.checkpoint, map_location=device)
    unet_lora.load_state_dict(state, strict=False)
    unet_lora.eval()

    clip = CLIPEncoder().to(device)
    clip.text_encoder.eval()

    scheduler = Scheduler()
    scheduler.scheduler.set_timesteps(args.steps)

    # Inputs
    src_tensor = preprocess_image(Path(args.source), resolution, device)
    with torch.no_grad():
        z_src = vae.encode(src_tensor).latent_dist.sample() * LATENT_SCALING

    tokenized = clip.tokenizer(
        args.instruction,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    text_ids = tokenized.input_ids.to(device)
    text_emb = clip.text_encoder(text_ids)[0]

    # DDIM sampling
    z = torch.randn_like(z_src)
    for timestep in scheduler.scheduler.timesteps:
        with torch.no_grad():
            z_input = torch.cat([z, z_src], dim=1)
            noise_pred = unet_lora(z_input, timestep, encoder_hidden_states=text_emb).sample
            step_output = scheduler.scheduler.step(noise_pred, timestep, z)
            z = step_output.prev_sample

    with torch.no_grad():
        edited = vae.decode(z / LATENT_SCALING).sample

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(edited, output_path)
    print(f"Saved edited image to {output_path}")


if __name__ == "__main__":
    main()
