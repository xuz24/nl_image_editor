from __future__ import annotations

from diffusers import DDIMScheduler


class Scheduler:
    """Simple proxy for DDIMScheduler to keep train_lora.py tidy."""

    def __init__(self, model_name: str = "CompVis/stable-diffusion-v1-4"):
        self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    @property
    def num_train_timesteps(self):
        return getattr(self.scheduler, "num_train_timesteps", self.scheduler.config.num_train_timesteps)

    def add_noise(self, *args, **kwargs):
        return self.scheduler.add_noise(*args, **kwargs)
