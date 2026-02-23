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


# -------------------------
# 1. Load Config
# -------------------------
with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = int(config["batch_size"])
LR = float(config["learning_rate"])
STEPS = int(config["num_training_steps"])
SAVE_EVERY = int(config["save_every"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
UNET_IN_CHANNELS = 8  # concat(z_noisy, z_src)

print(f"BATCH SIZE: {BATCH_SIZE} \nLR: {LR} \nSTEPS: {STEPS} \nSAVE_EVERY: {SAVE_EVERY} \nDEVICE: {DEVICE} \nDTYPE: {DTYPE}")

# -------------------------
# 3. Load Models
# -------------------------

vae = VAE(torch_dtype=DTYPE).to(DEVICE)
unet = UnetLora(torch_dtype=DTYPE, in_channels=UNET_IN_CHANNELS).to(DEVICE)
clip = CLIPEncoder().to(DEVICE)

scheduler = Scheduler()

scaler = torch.cuda.amp.GradScaler()

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
)
unet_lora = unet.get_model(lora_config)

# Freeze everything else
vae.autoencoder.requires_grad_(False)
clip.text_encoder.requires_grad_(False)

# Keep 8-channel conditioning: conv_in is randomly initialized due channel mismatch,
# so it must be trainable or the model cannot learn to use concatenated latents.
unet_lora.model.conv_in.requires_grad_(True)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet_lora.parameters()), lr=LR)

# Sanity-check trainable params: ensure conv_in + LoRA adapters are trainable.
total_params = sum(p.numel() for p in unet_lora.parameters())
trainable_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
trainable_names = [name for name, p in unet_lora.named_parameters() if p.requires_grad]
print(
    f"UNet params: total={total_params:,} trainable={trainable_params:,} "
    f"({100.0 * trainable_params / total_params:.4f}%)"
)
print(f"conv_in trainable: {unet_lora.model.conv_in.weight.requires_grad}")
print(f"Sample trainable tensors: {trainable_names[:12]}")

# -------------------------
# 5. Prepare DataLoader
# -------------------------
dataset = PicoBananaDataset("/scratch/eecs476w26s_class_root/eecs476w26s_class/xuzijie/pico-banana", clip.tokenizer, resolution=config["resolution"])
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# -------------------------
# 6. Training Loop
# -------------------------
step = 0
while step < STEPS:
    for src_imgs, tgt_imgs, instr_ids in tqdm(loader):
        if step >= STEPS:
            break

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
        z_input = torch.cat([z_noisy, z_src], dim=1)

        # encode text
        with torch.no_grad():
            text_emb = clip.text_encoder(instr_ids)[0]
        
        # forward pass
        with torch.cuda.amp.autocast(enabled=False):
            eps_pred = unet_lora(
                z_input,
                t,
                encoder_hidden_states=text_emb,
            ).sample

            loss = torch.nn.functional.mse_loss(eps_pred, noise)
            
            if step % SAVE_EVERY == 0:
                print(f"[step {step}] loss = {loss.item():.6f}")
            
        optimizer.zero_grad()
        if DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        step += 1

        # Save checkpoint
        if step % SAVE_EVERY == 0:
            ckpt_path = f"checkpoints/lora_step_{step}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(unet_lora.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
