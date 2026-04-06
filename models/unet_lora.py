from __future__ import annotations

import torch
from diffusers import UNet2DConditionModel
from peft.tuners.lora import LoraModel


class UnetLora:

    def __init__(
        self,
        model_name: str = "CompVis/stable-diffusion-v1-4",
        torch_dtype: torch.dtype = torch.float32,
        in_channels: int = 4,
        init_extended_conv_from_pretrained: bool = True,
    ):
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet",
            torch_dtype=torch_dtype,
            in_channels=in_channels,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )
        self._maybe_init_extended_conv(
            model_name=model_name,
            torch_dtype=torch_dtype,
            in_channels=in_channels,
            enabled=init_extended_conv_from_pretrained,
        )

    def _maybe_init_extended_conv(
        self,
        model_name: str,
        torch_dtype: torch.dtype,
        in_channels: int,
        enabled: bool,
    ) -> None:
        if not enabled or in_channels <= 4:
            return

        # Stabilize 8-channel training: preserve pretrained behavior on first 4 channels
        # and start extra channels at zero.
        base_unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
        )
        with torch.no_grad():
            self.unet.conv_in.weight[:, :4].copy_(base_unet.conv_in.weight)
            self.unet.conv_in.weight[:, 4:].zero_()
            self.unet.conv_in.bias.copy_(base_unet.conv_in.bias)
        del base_unet

    def to(self, device: torch.device | str) -> "UnetLora":
        self.unet = self.unet.to(device)
        return self

    def get_model(self, lora_config, adapter_name: str = "default"):
        return LoraModel(self.unet, lora_config, adapter_name)
