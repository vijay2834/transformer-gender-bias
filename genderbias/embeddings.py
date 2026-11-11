"""Embedding back-ends: static GloVe vectors and contextual transformer encoders."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def load_glove(path: str | Path, vocab_limit: int | None = 100_000) -> dict[str, np.ndarray]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    emb = {}
    with opener(path, "rt", encoding="utf8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if len(parts) == 2:      # word2vec-format header
                continue
            emb[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
            if vocab_limit and len(emb) >= vocab_limit:
                break
    return emb


class TransformerEncoder:
    """Mean-pooled sentence embeddings from a HuggingFace model (BERT, GPT-2, RoBERTa, XLNet).
    Used for SEAT: each WEAT word is placed into "This is <word>." style templates."""

    TEMPLATES = ["This is {}.", "That is {}.", "Here is {}.", "{} is here.", "It is {}."]

    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def encode_words(self, words: list[str]) -> np.ndarray:
        out = []
        for w in words:
            sents = [t.format(w) for t in self.TEMPLATES]
            batch = self.tok(sents, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                h = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            out.append(((h * mask).sum(1) / mask.sum(1)).mean(0).cpu().numpy())
        return np.vstack(out)
