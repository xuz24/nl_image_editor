from pathlib import Path
from typing import List
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PicoBananaDataset(Dataset):
    """Simple folder-based dataset for Pico-Banana samples."""

    REQUIRED_FILES = ("source.png", "target.png", "instruction.txt")

    def __init__(self, root_dir, tokenizer, resolution=256, jsonl_path=None, output_root=None):
        self.root_dir = Path(root_dir)
        self.use_jsonl = jsonl_path is not None or self.root_dir.suffix == ".jsonl"
        self.samples: List = []

        if self.use_jsonl:
            self._init_jsonl_samples(jsonl_path, output_root)
        else:
            self._init_folder_samples()

        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _init_folder_samples(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"PicoBananaDataset root not found: {self.root_dir}")

        for entry in sorted(self.root_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if all((entry / name).exists() for name in self.REQUIRED_FILES):
                self.samples.append(entry)

        if not self.samples:
            raise RuntimeError(f"No valid Pico-Banana folder samples found in {self.root_dir}")

    def _init_jsonl_samples(self, jsonl_path, output_root):
        jsonl_file = Path(jsonl_path) if jsonl_path is not None else self.root_dir
        if not jsonl_file.exists():
            raise FileNotFoundError(f"PicoBananaDataset jsonl not found: {jsonl_file}")

        output_base = Path(output_root) if output_root is not None else jsonl_file.parent
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source_path = row.get("local_input_image")
            target_rel = row.get("output_image")
            instruction = row.get("text", "").strip()
            if not source_path or not target_rel or not instruction:
                continue

            source = Path(source_path)
            target = output_base / target_rel.lstrip("/")
            if source.exists() and target.exists():
                self.samples.append((source, target, instruction))

        if not self.samples:
            raise RuntimeError(
                f"No valid Pico-Banana JSONL samples found in {jsonl_file}. "
                "Ensure local_input_image exists and output images were downloaded."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_jsonl:
            source_path, target_path, instruction = self.samples[idx]
            with Image.open(source_path) as src_img:
                src_img = src_img.convert("RGB")
            with Image.open(target_path) as tgt_img:
                tgt_img = tgt_img.convert("RGB")
        else:
            sample_dir = self.samples[idx]
            with Image.open(sample_dir / "source.png") as src_img:
                src_img = src_img.convert("RGB")
            with Image.open(sample_dir / "target.png") as tgt_img:
                tgt_img = tgt_img.convert("RGB")
            instruction = (sample_dir / "instruction.txt").read_text(encoding="utf-8").strip()

        src_img = self.transform(src_img)
        tgt_img = self.transform(tgt_img)

        tokenized = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return src_img, tgt_img, tokenized.input_ids.squeeze(0)
