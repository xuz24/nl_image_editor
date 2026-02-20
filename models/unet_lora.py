from __future__ import annotations

import torch
from diffusers import UNet2DConditionModel
from peft.tuners.lora import LoraModel


class UnetLora:
    """Loads the base UNet and applies LoRA adapters."""

    def __init__(
        self,
        model_name: str = "CompVis/stable-diffusion-v1-4",
        torch_dtype: torch.dtype = torch.float32,
        in_channels: int = 4,
    ):
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet",
            torch_dtype=torch_dtype,
            in_channels=in_channels,
        )

    def to(self, device: torch.device | str) -> "UnetLora":
        self.unet = self.unet.to(device)
        return self

    def get_model(self, lora_config, adapter_name: str = "default"):
        return LoraModel(self.unet, lora_config, adapter_name)
