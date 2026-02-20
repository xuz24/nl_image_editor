from pathlib import Path
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PicoBananaDataset(Dataset):
    """Simple folder-based dataset for Pico-Banana samples."""

    REQUIRED_FILES = ("source.png", "target.png", "instruction.txt")

    def __init__(self, root_dir, tokenizer, resolution=256):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"PicoBananaDataset root not found: {self.root_dir}")

        self.samples: List[Path] = []
        for entry in sorted(self.root_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if all((entry / name).exists() for name in self.REQUIRED_FILES):
                self.samples.append(entry)

        if not self.samples:
            raise RuntimeError(f"No valid Pico-Banana samples found in {self.root_dir}")

        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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
