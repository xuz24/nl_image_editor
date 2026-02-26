from models.vae import VAE
from models.clip_encoder import CLIPEncoder
from models.unet_lora import UnetLora
from diffusion.scheduler import Scheduler
from data.loaders import PicoBananaDataset

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig
import os
import yaml
from tqdm import tqdm
from contextlib import nullcontext
from collections import deque
from pathlib import Path
import sys

from PIL import Image
from torchvision import transforms

def save_tensor_as_image(tensor: torch.Tensor, path: Path) -> None:
tensor = tensor.clamp(-1, 1)
tensor = (tensor + 1) / 2
image = tensor.mul(255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
path.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(image).save(path)


def run_validation(step: int) -> None:
source_path = config.get("val_source_image")
instruction = config.get("val_instruction")
if not source_path or not instruction:
    return

val_steps = int(config.get("val_steps", 30))
val_seed = int(config.get("val_seed", 42))
val_output_dir = Path(config.get("val_output_dir", "validation_outputs"))
resolution = int(config["resolution"])

preprocess = transforms.Compose(
    [
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

src = Image.open(source_path).convert("RGB")
src_tensor = preprocess(src).unsqueeze(0).to(DEVICE, dtype=DTYPE)

generator = torch.Generator(device=DEVICE).manual_seed(val_seed)

was_training = unet_lora.training
unet_lora.eval()
with torch.no_grad():
    z_src = vae.encode(src_tensor).latent_dist.sample() * 0.18215
    z = torch.randn(z_src.shape, device=DEVICE, dtype=DTYPE, generator=generator)

    tok = clip.tokenizer(
        instruction,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    text_emb = clip.text_encoder(tok.input_ids.to(DEVICE))[0].to(dtype=DTYPE)

    val_scheduler.set_timesteps(val_steps)
    amp_ctx = torch.amp.autocast("cuda", dtype=AMP_DTYPE) if use_amp_runtime else nullcontext()
    for ts in val_scheduler.timesteps:
        z_input = torch.cat([z, z_src], dim=1)
        with amp_ctx:
            noise_pred = unet_lora(z_input, ts, encoder_hidden_states=text_emb).sample
        z = val_scheduler.step(noise_pred, ts, z).prev_sample

    edited = vae.decode(z / 0.18215).sample
    out_path = val_output_dir / f"val_step_{step}.png"
    save_tensor_as_image(edited, out_path)
    print(f"[val] saved: {out_path}")

if was_training:
    unet_lora.train()