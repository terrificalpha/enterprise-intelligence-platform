"""Embedding backends behind one interface.

* `LSAEmbedder`  (default, offline, zero downloads): TF-IDF (1-2 grams) -> truncated SVD -> L2 norm.
  Latent Semantic Analysis captures co-occurrence semantics (e.g. "churn" ~ "attrition") from the corpus itself.
* `SentenceTransformerEmbedder` (optional): neural embeddings, enabled with EIP_EMBEDDING=sentence-transformers.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..text_utils import tokenize


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), sublinear_tf=True,
                                     min_df=1, token_pattern=None)
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        X = self.tfidf.fit_transform(texts)
        k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=42).fit(X)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        Z = self.svd.transform(self.tfidf.transform(texts))
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.clip(n, 1e-9, None)


class SentenceTransformerEmbedder:  # pragma: no cover - needs model download
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):  # no fitting needed
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class OllamaEmbedder:
    """Neural embeddings from a local Ollama model (e.g. `ollama pull nomic-embed-text`). Optional upgrade over LSA."""
    name = "ollama"

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def fit(self, texts):
        return self

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        import requests
        r = requests.post(f"{self.base_url}/api/embed", json={"model": self.model_name, "input": batch}, timeout=120)
        if r.status_code == 404 and "not found" in r.text.lower():
            raise RuntimeError(f"Ollama embedding model '{self.model_name}' not installed: ollama pull {self.model_name}")
        if r.ok and "embeddings" in r.json():
            return r.json()["embeddings"]
        out = []                                   # older Ollama: one prompt per request
        for t in batch:
            r = requests.post(f"{self.base_url}/api/embeddings", json={"model": self.model_name, "prompt": t}, timeout=60)
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 32):
            vectors.extend(self._embed_batch(texts[i:i + 32]))
        Z = np.array(vectors, dtype=np.float32)
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.clip(n, 1e-9, None)


def make_embedder(backend: str, st_model: str = ""):
    if backend.startswith("sentence"):
        try:
            return SentenceTransformerEmbedder(st_model)
        except Exception as e:  # missing package / no network -> graceful fallback
            print(f"[embeddings] sentence-transformers unavailable ({e}); falling back to LSA")
    elif backend.startswith("ollama"):
        from ..config import settings
        emb = OllamaEmbedder(model_name=settings.ollama_embed_model, base_url=settings.ollama_base_url)
        try:
            emb.encode(["probe"])                  # fail fast -> graceful LSA fallback if Ollama/model is missing
            return emb
        except Exception as e:
            print(f"[embeddings] ollama embeddings unavailable ({e}); falling back to LSA")
    return LSAEmbedder()
