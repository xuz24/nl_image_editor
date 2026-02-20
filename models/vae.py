from __future__ import annotations

import torch
from diffusers import AutoencoderKL


class VAE:
    """Thin wrapper around the Stable Diffusion VAE."""

    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16):
        self.autoencoder = AutoencoderKL.from_pretrained(model_name, torch_dtype=torch_dtype)

    def to(self, device: torch.device | str) -> "VAE":
        self.autoencoder = self.autoencoder.to(device)
        return self

    def encode(self, images):
        return self.autoencoder.encode(images)

    def decode(self, latents):
        return self.autoencoder.decode(latents)
