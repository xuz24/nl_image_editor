# NL Image Editor

An instruction-guided image editing project built around a Stable Diffusion 1.4 backbone with LoRA fine-tuning. Given a source image and a natural-language edit instruction, the model predicts the edited result with dual-condition classifier-free guidance.

## What's in the repo

- A Pico-Banana dataset preparation pipeline for turning the raw dataset into local training samples.
- A LoRA training loop for instruction-conditioned image editing.
- A standalone inference CLI for editing a single image from the command line.
- A lightweight Flask demo that serves the browser UI and calls the inference script.
- A Colab-friendly demo path that can expose the same app through ngrok.

## Project structure

- `configs/` - training and dataset configuration.
- `data/` - prepared Pico-Banana data and assets.
- `demo/` - Flask server and browser frontend for interactive editing.
- `models/` - VAE, CLIP, and UNet LoRA modules.
- `scripts/` - training, inference, utilities, and dataset download/prep helpers.

## Requirements

Python 3.10+ is recommended. Install the core dependencies with:

```bash
pip install -r requirements.txt
```

You will also need a PyTorch build that matches your machine, especially if you want GPU acceleration.

## Dataset preparation

The repo is set up around the Pico-Banana paired editing data. The helper script downloads the SFT split and materializes each sample locally.

```bash
bash scripts/download_pico_banana.sh --max-samples 2000
```

The prepared samples are written under `data/pico-banana/` by default. Each example contains:

- `source.png`
- `target.png`
- `instruction.txt`
- `meta.json`

The training loader expects the JSONL metadata plus the downloaded image tree referenced by the config files.

## Training

Training is implemented in `scripts/train_lora.py` and uses LoRA adapters on top of the UNet.

The default configuration lives in `configs/training_config.yaml`. It controls:

- batch size and learning rate
- LoRA rank and dropout
- number of training steps
- checkpoint save frequency
- validation inputs and output location

Run training with the same environment used for the dataset and model dependencies:

```bash
python -m scripts.train_lora
```

Checkpoints are saved into `checkpoints/` during training.

## Inference

The standalone inference script edits one image at a time:

```bash
python -m scripts.run_inference \
	--source path/to/source.png \
	--instruction "Remove the stop sign" \
	--checkpoint checkpoints/lora_step_5000.pt \
	--output outputs/edited.png
```

Useful flags:

- `--steps` controls the denoising length.
- `--seed` makes generation repeatable.
- `--sI` and `--sT` adjust image and text guidance strengths.
- `--config` points at the training config used for resolution and LoRA settings.

## Demo

### Local Flask demo

The browser demo is served by `demo/server.py`, which wraps the inference script and serves `demo/index.html`.

```bash
python demo/server.py --port 5000
```

Then open the local URL printed by the server and upload an image, add an instruction, and run the edit.

### Colab + ngrok demo

If you use the Google Colab notebook, the same Flask app can be exposed to your browser through ngrok.

Typical flow:

1. Install the Python dependencies in Colab.
2. Launch the Flask demo on port `5000`.
3. Start an ngrok tunnel to that port.
4. Open the public ngrok URL in your browser and use the demo UI.

The notebook can also set the checkpoint path and config path before starting the server, which makes it easy to swap in different trained weights.

## Notes

- The repo is currently focused on training and interactive editing, not on a formal evaluation benchmark.
- The default training and demo paths assume the checkpoint and dataset locations defined in `configs/training_config.yaml`.
- The dataset scripts were written for the Pico-Banana release and its associated JSONL metadata.

## License

No explicit license file is included in the repository yet.
