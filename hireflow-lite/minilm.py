"""
minilm.py — Bulletproof MiniLM Sentence Embedder
=================================================
Uses pure HuggingFace transformers (AutoTokenizer + AutoModel).
Eliminates sentence-transformers stderr warnings and C-level crashes.
"""

from __future__ import annotations

import os
import warnings

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
warnings.filterwarnings("ignore")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

_EMBEDDER: MiniLMEmbedder | None = None


class MiniLMEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode(
        self,
        texts: str | list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts_list = [texts]
            single = True
        else:
            texts_list = list(texts)
            single = False

        if not texts_list:
            empty = np.zeros((0, 384), dtype=np.float32)
            return empty[0] if single else empty

        all_embs = []
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                sum_emb = torch.sum(out.last_hidden_state * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                emb = sum_emb / sum_mask
                if normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu())

        res = torch.cat(all_embs, dim=0)
        if convert_to_numpy:
            res = res.numpy()
        return res[0] if single else res


def get_minilm_model(device: str = "cpu") -> MiniLMEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = MiniLMEmbedder(device=device)
    return _EMBEDDER
