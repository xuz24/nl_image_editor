from __future__ import annotations

from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler

class Scheduler:
    def __init__(self, train, model_name: str = "CompVis/stable-diffusion-v1-4"):
        self.train = train

        if train:
            self.scheduler = DDIMScheduler.from_pretrained(
                model_name, subfolder="scheduler"
            )
        else:
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                model_name, subfolder="scheduler"
            )

    @property
    def num_train_timesteps(self):
        return getattr(
            self.scheduler,
            "num_train_timesteps",
            self.scheduler.config.num_train_timesteps,
        )

    def add_noise(self, *args, **kwargs):
        if not self.train:
            raise RuntimeError("add_noise should only be used during training")
        return self.scheduler.add_noise(*args, **kwargs)