from __future__ import annotations

import torch
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPEncoder:
    """Wrapper that keeps tokenizer + text encoder on the same device."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        # Backwards compatibility with code that expects .model
        self.model = self.text_encoder

    def to(self, device: torch.device | str) -> "CLIPEncoder":
        self.text_encoder = self.text_encoder.to(device)
        self.model = self.text_encoder
        return self

    def encode_text(self, text: str, **tokenizer_kwargs):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
            **tokenizer_kwargs,
        )
        tokens = {k: v.to(self.text_encoder.device) for k, v in tokens.items()}
        return self.text_encoder(**tokens)[0]

    def encode_tokens(self, input_ids):
        return self.text_encoder(input_ids)[0]
