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


# -------------------------
# 1. Load Config
# -------------------------
with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = int(config["batch_size"])
LR = float(config["learning_rate"])
STEPS = int(config["num_training_steps"])
SAVE_EVERY = int(config["save_every"])
LOG_EVERY = int(config.get("log_every", 50))
LOSS_WINDOW = int(config.get("loss_window", 100))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Keep trainable params in fp32 for optimizer stability.
DTYPE = torch.float32
UNET_IN_CHANNELS = 8  # concat(z_noisy, z_src)
USE_AMP = DEVICE == "cuda"
AMP_DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
USE_SCALER = DEVICE == "cuda" and AMP_DTYPE == torch.float16

print(
    f"BATCH SIZE: {BATCH_SIZE} \nLR: {LR} \nSTEPS: {STEPS} \nSAVE_EVERY: {SAVE_EVERY} "
    f"\nDEVICE: {DEVICE} \nDTYPE: {DTYPE} \nUSE_AMP: {USE_AMP} "
    f"\nAMP_DTYPE: {AMP_DTYPE if USE_AMP else 'n/a'} \nUSE_SCALER: {USE_SCALER}"
)

# -------------------------
# 3. Load Models
# -------------------------

vae = VAE(torch_dtype=DTYPE).to(DEVICE)
unet = UnetLora(torch_dtype=DTYPE, in_channels=UNET_IN_CHANNELS).to(DEVICE)
clip = CLIPEncoder().to(DEVICE)

scheduler = Scheduler()
val_scheduler = Scheduler().scheduler
prediction_type = scheduler.scheduler.config.prediction_type
print(f"Scheduler prediction_type: {prediction_type}")
if prediction_type != "epsilon":
    raise RuntimeError(
        f"Training loss currently targets epsilon noise, but scheduler prediction_type={prediction_type}. "
        "Set scheduler prediction_type to epsilon or change training target accordingly."
    )

scaler = torch.cuda.amp.GradScaler(enabled=USE_SCALER)

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

# state = torch.load("/home/xuzijie/nl_image_editor/checkpoints/lora_step_100000.pt", map_location=DEVICE)
# unet_lora.load_state_dict(state, strict=False)

# Freeze everything else
vae.autoencoder.requires_grad_(False)
clip.text_encoder.requires_grad_(False)
vae.autoencoder.eval()
clip.text_encoder.eval()
unet_lora.train()

# Keep 8-channel conditioning: conv_in is randomly initialized due channel mismatch,
# so it must be trainable or the model cannot learn to use concatenated latents.
unet_lora.model.conv_in.requires_grad_(True)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet_lora.parameters()), lr=LR)

# Sanity-check trainable params: ensure conv_in + LoRA adapters are trainable.
total_params = sum(p.numel() for p in unet_lora.parameters())
trainable_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
trainable_names = [name for name, p in unet_lora.named_parameters() if p.requires_grad]
lora_trainable = [name for name in trainable_names if "lora_" in name]
print(
    f"UNet params: total={total_params:,} trainable={trainable_params:,} "
    f"({100.0 * trainable_params / total_params:.4f}%)"
)
print(f"conv_in trainable: {unet_lora.model.conv_in.weight.requires_grad}")
print(f"Sample trainable tensors: {trainable_names[:12]}")
print(f"Trainable LoRA tensors: {len(lora_trainable)}")
if len(lora_trainable) == 0:
    raise RuntimeError(
        "No trainable LoRA tensors found. Check lora_target_modules names in config "
        "(expected matches like to_q/to_k/to_v/to_out.0)."
    )

# -------------------------
# 5. Prepare DataLoader
# -------------------------
dataset = PicoBananaDataset(
    config.get("dataset_root", "data/pico-banana"),
    clip.tokenizer,
    resolution=config["resolution"],
    jsonl_path=config.get("dataset_jsonl"),
    output_root=config.get("dataset_output_root"),
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=(DEVICE == "cuda"),
)


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

# -------------------------
# 6. Training Loop
# -------------------------
step = 0
loss_history = deque(maxlen=LOSS_WINDOW)
use_amp_runtime = USE_AMP
while step < STEPS:
    for src_imgs, tgt_imgs, instr_ids in tqdm(loader, file=sys.stdout):
        if step >= STEPS:
            break

        src_imgs = src_imgs.to(DEVICE, dtype=DTYPE)
        tgt_imgs = tgt_imgs.to(DEVICE, dtype=DTYPE)
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
            text_emb = text_emb.to(dtype=z_input.dtype)
        
        # forward pass
        amp_ctx = torch.amp.autocast("cuda", dtype=AMP_DTYPE) if use_amp_runtime else nullcontext()
        with amp_ctx:
            eps_pred = unet_lora(
                z_input,
                t,
                encoder_hidden_states=text_emb,
            ).sample

            loss = torch.nn.functional.mse_loss(eps_pred, noise)
            
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(
                f"[warn] non-finite loss at step {step} "
                f"(amp={use_amp_runtime}, amp_dtype={AMP_DTYPE if use_amp_runtime else 'n/a'})"
            )
            print(
                f"[warn] finite flags: z_input={torch.isfinite(z_input).all().item()} "
                f"text_emb={torch.isfinite(text_emb).all().item()} "
                f"noise={torch.isfinite(noise).all().item()} "
                f"eps_pred={torch.isfinite(eps_pred).all().item()}"
            )
            if use_amp_runtime:
                # Retry same step in full precision, then keep AMP disabled for stability.
                with nullcontext():
                    eps_pred = unet_lora(
                        z_input,
                        t,
                        encoder_hidden_states=text_emb,
                    ).sample
                    loss = torch.nn.functional.mse_loss(eps_pred, noise)
                if torch.isfinite(loss):
                    use_amp_runtime = False
                    print("[warn] AMP disabled after instability; continuing in fp32 forward.")
                else:
                    raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            else:
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss_history.append(loss.item())
        if step % LOG_EVERY == 0:
            avg_loss = sum(loss_history) / len(loss_history)
            print(f"[step {step}] loss = {loss.item():.6f} | loss_ma{LOSS_WINDOW} = {avg_loss:.6f}")

        if USE_SCALER:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet_lora.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet_lora.parameters(), max_norm=1.0)
            optimizer.step()
        step += 1

        # Save checkpoint
        if step % SAVE_EVERY == 0:
            ckpt_path = f"checkpoints/lora_step_{step}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(unet_lora.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            run_validation(step)
