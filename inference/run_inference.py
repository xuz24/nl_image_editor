#!/usr/bin/env python3
"""
Run inference with dual-condition classifier-free guidance (image + text).
"""

from __future__ import annotations

import argparse
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
    return tfm(img).unsqueeze(0).to(device)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2
    img = tensor.mul(255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
    Image.fromarray(img).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pico-Banana LoRA inference with dual CFG.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/edited.png")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default="configs/training_config.yaml")

    # NEW: guidance strengths
    parser.add_argument("--sI", type=float, default=1.6, help="Image guidance scale")
    parser.add_argument("--sT", type=float, default=7.5, help="Text guidance scale")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

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
    unet_lora.load_state_dict(state["model"], strict=False)
    unet_lora.eval()

    clip = CLIPEncoder().to(device)
    clip.text_encoder.eval()

    scheduler = Scheduler(train=False)
    scheduler.scheduler.set_timesteps(args.steps)

    # --- Input image ---
    src_tensor = preprocess_image(Path(args.source), resolution, device)

    with torch.no_grad():
        z_src = vae.encode(src_tensor).latent_dist.sample() * LATENT_SCALING

    # --- Text embedding ---
    tokenized = clip.tokenizer(
        args.instruction,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    text_ids = tokenized.input_ids.to(device)

    with torch.no_grad():
        text_emb = clip.text_encoder(text_ids)[0]

    # --- Null text embedding ---
    null_tokenized = clip.tokenizer(
        "",
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    null_ids = null_tokenized.input_ids.to(device)

    with torch.no_grad():
        null_text_emb = clip.text_encoder(null_ids)[0]

    # --- Null image ---
    zero_src = torch.zeros_like(z_src)

    # --- Sampling ---
    z = torch.randn_like(z_src)

    for timestep in scheduler.scheduler.timesteps:
        with torch.no_grad():

            # -------- Batch 3 conditions together (fast) --------
            z_batch = torch.cat([z, z, z], dim=0)
            src_batch = torch.cat([zero_src, z_src, z_src], dim=0)
            text_batch = torch.cat([null_text_emb, null_text_emb, text_emb], dim=0)

            z_input = torch.cat([z_batch, src_batch], dim=1)
            
            outputs = unet_lora(
                z_input,
                timestep,
                encoder_hidden_states=text_batch
            ).sample

            e_uncond, e_img, e_full = outputs.chunk(3)

            # -------- Dual CFG --------
            noise_pred = (
                e_uncond
                + args.sI * (e_img - e_uncond)
                + args.sT * (e_full - e_img)
            )

            step_output = scheduler.scheduler.step(noise_pred, timestep, z)
            z = step_output.prev_sample

    # --- Decode ---
    with torch.no_grad():
        edited = vae.decode(z / LATENT_SCALING).sample

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(edited, output_path)

    print(f"Saved edited image to {output_path}")


if __name__ == "__main__":
    main()