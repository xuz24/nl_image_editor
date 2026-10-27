from pathlib import Path
from typing import List
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class PicoBananaDataset(Dataset):

    REQUIRED_FILES = ("source.png", "target.png", "instruction.txt")

    def __init__(self, root_dir, tokenizer, resolution=256, jsonl_path=None, output_root=None):
        self.root_dir = Path(root_dir)
        self.samples: List = []


        self._init_jsonl_samples(jsonl_path, output_root)

        self.tokenizer = tokenizer
        self.resolution = resolution

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
            # instruction = row.get("text", "").strip()
            instruction = row.get("summarized_text").strip()
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

    def transform_pair(self, src_img, tgt_img):
        # Resize 
        src_img = TF.resize(src_img, (self.resolution + 30, self.resolution + 30))
        tgt_img = TF.resize(tgt_img, (self.resolution + 30, self.resolution + 30))

        # Get same crop params
        i, j, h, w = transforms.RandomCrop.get_params(
            src_img, output_size=(self.resolution, self.resolution)
        )

        src_img = TF.crop(src_img, i, j, h, w)
        tgt_img = TF.crop(tgt_img, i, j, h, w)

        # Convert + normalize
        src_img = TF.to_tensor(src_img)
        tgt_img = TF.to_tensor(tgt_img)

        src_img = TF.normalize(src_img, [0.5]*3, [0.5]*3)
        tgt_img = TF.normalize(tgt_img, [0.5]*3, [0.5]*3)

        return src_img, tgt_img
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source_path, target_path, instruction = self.samples[idx]
        with Image.open(source_path) as src_img:
            src_img = src_img.convert("RGB")
        with Image.open(target_path) as tgt_img:
            tgt_img = tgt_img.convert("RGB")

        src_img, tgt_img = self.transform_pair(src_img, tgt_img)

        tokenized = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return src_img, tgt_img, tokenized.input_ids.squeeze(0)
