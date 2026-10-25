"""Okapi BM25 implemented from scratch (sparse / lexical retriever)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..lexicon import FILLER, FILLER_WEIGHT, expand_terms
from ..text_utils import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts: list[str]) -> "BM25":
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.dl = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = float(self.dl.mean()) if len(self.dl) else 0.0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.max_idf = math.log(1 + (n + 0.5) / 1.5)  # idf of an out-of-vocabulary term
        return self

    def scores(self, query: str, expand: bool = True) -> np.ndarray:
        """Okapi BM25 score of every document. With expand=True, lexicon aliases of query words are added
        at a reduced weight (0.5) so paraphrases still reach the right passage without drowning exact matches."""
        base, aliases = expand_terms(query) if expand else (tokenize(query), {})
        weights: dict[str, float] = {t: (FILLER_WEIGHT if t in FILLER else 1.0) for t in base}
        for al in aliases.values():
            for a in al:
                weights.setdefault(a, 0.5)
        s = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
            for t, w in weights.items():
                f = d.get(t, 0)
                if f:
                    s[i] += w * self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s
