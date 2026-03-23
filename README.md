# Instruction-Guided Diffusion Model

## Description
This project implements an **InstructPix2Pix–style diffusion model** for **instruction-guided image editing**, enabling users to modify images via natural language instructions. The model translates textual instructions into image transformations using a fine-tuned Stable Diffusion backbone.

---

## Key Features
- **LoRA Fine-Tuning:** Efficient parameter injection into Stable Diffusion 1.4.  
- **Classifier-Free Guidance:** Enhances adherence to textual instructions during generation.  
- **Supervised Training:** Fine-tuned on the **Pico-Banana dataset (250k+ paired examples)** for instruction-to-image alignment.  
- **HPC-Ready:** Training pipeline designed to run on clusters such as **Great Lakes HCP** for scalable computation.  

---

## Tech Stack
- Python, PyTorch  
- Stable Diffusion 1.4  
- Hugging Face Diffusers  

---

## Workflow
1. **Dataset Preparation:** Preprocess Pico-Banana paired examples for instruction-conditioned training.  
2. **Model Fine-Tuning:** LoRA injected into Stable Diffusion 1.4, LoRA rank=4.
3. **Training:** (Currently here) Train on 250k+ paired examples with classifier-free guidance.
4. **Evaluation:** Pending; to measure instruction alignment and image fidelity on real world samples.

---
