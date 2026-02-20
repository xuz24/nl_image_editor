from models.vae import VAE
from models.clip_encoder import CLIPEncoder
from models.unet_lora import UnetLora
from diffusion.scheduler import Scheduler
from data.loaders import PicoBananaDataset

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType
import os
import yaml
from tqdm import tqdm


# -------------------------
# 1. Load Config
# -------------------------
with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config["batch_size"]
LR = config["learning_rate"]
STEPS = config["num_training_steps"]
SAVE_EVERY = config["save_every"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# 3. Load Models
# -------------------------

vae = VAE().to(DEVICE)
unet = UnetLora().to(DEVICE)
clip = CLIPEncoder().to(DEVICE)

scheduler = Scheduler()

# vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(DEVICE)
# unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to(DEVICE)
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# -------------------------
# 4. LoRA Injection
# -------------------------
lora_config = LoraConfig(
    r=config.get("lora_rank", 4),
    lora_alpha=config.get("lora_alpha", 16),
    target_modules=config.get("lora_target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
    lora_dropout=config.get("lora_dropout", 0.05),
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)
unet_lora = unet.get_model(lora_config)

# Freeze everything else
vae.autoencoder.requires_grad_(False)
clip.text_encoder.requires_grad_(False)
for param in unet_lora.parameters():
    if not param.requires_grad:
        param.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet_lora.parameters()), lr=LR)

# -------------------------
# 5. Prepare DataLoader
# -------------------------
dataset = PicoBananaDataset("data/pico-banana", clip.tokenizer, resolution=config["resolution"])
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# -------------------------
# 6. Training Loop
# -------------------------
for step in range(STEPS):
    for src_imgs, tgt_imgs, instr_ids in tqdm(loader):
        src_imgs = src_imgs.to(DEVICE)
        tgt_imgs = tgt_imgs.to(DEVICE)
        instr_ids = instr_ids.to(DEVICE)

        # encode images
        with torch.no_grad():
            z_src = vae.encode(src_imgs).latent_dist.sample() * 0.18215
            z_tgt = vae.encode(tgt_imgs).latent_dist.sample() * 0.18215

        # sample noise
        noise = torch.randn_like(z_tgt)
        t = torch.randint(0, scheduler.num_train_timesteps, (z_tgt.shape[0],), device=DEVICE)
        z_noisy = scheduler.add_noise(z_tgt, noise, t)

        # encode text
        text_emb = clip.text_encoder(instr_ids)[0]

        # forward pass
        eps_pred = unet_lora(z_noisy, t, encoder_hidden_states=text_emb, cross_attention_kwargs={"source_latent": z_src})

        # compute loss
        loss = torch.nn.functional.mse_loss(eps_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save checkpoint
    if step % SAVE_EVERY == 0:
        ckpt_path = f"checkpoints/lora_step_{step}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(unet_lora.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
