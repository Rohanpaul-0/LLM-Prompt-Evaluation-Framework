from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from .metrics_base import Metric, register

@register("semantic")
class SBERTCosine(Metric):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)
    def score(self, candidate: str, references: List[str]) -> float:
        texts = [candidate] + references
        embs = self.model.encode(texts, normalize_embeddings=True)
        c = embs[0]
        r = np.mean(embs[1:], axis=0) if len(embs) > 1 else embs[0]
        sim = float(np.dot(c, r))
        return (sim + 1.0) / 2.0
