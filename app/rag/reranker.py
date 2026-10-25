"""Second-stage reranker. A cross-feature heuristic scorer (no model download) that looks at the
(query, passage) pair jointly: IDF-weighted term coverage, exact bigram/phrase matches, number matches
and title overlap. Optionally a neural cross-encoder if sentence-transformers is installed."""
from __future__ import annotations

import re

from ..lexicon import FILLER, FILLER_WEIGHT, expand_terms
from ..text_utils import split_sentences, tokenize


class HeuristicReranker:
    name = "heuristic-cross-features"

    def __init__(self, idf: dict[str, float], max_idf: float):
        self.idf, self.max_idf = idf, max_idf

    def _w(self, t: str) -> float:
        w = self.idf.get(t, self.max_idf)
        return w * FILLER_WEIGHT if t in FILLER else w

    ALIAS_CREDIT = 0.8   # a lexicon synonym found in the passage earns 80% of the original word's weight

    def coverage(self, query: str, text: str, expand: bool = True) -> float:
        """Fraction of the query's information (IDF mass) that appears in the passage (synonym-aware)."""
        toks, aliases = expand_terms(query) if expand else (tokenize(query), {})
        q = set(toks)
        if not q:
            return 0.0
        d = set(tokenize(text))
        tot = sum(self._w(t) for t in q)
        got = 0.0
        for t in q:
            if t in d:
                got += self._w(t)
            elif any(a in d for a in aliases.get(t, ())):
                got += self.ALIAS_CREDIT * self._w(t)
        return got / tot

    def score(self, query: str, text: str, title: str = "", dense: float = 0.0) -> dict:
        cov = self.coverage(query, text)
        qt, dt = tokenize(query), tokenize(text)
        qb = {(a, b) for a, b in zip(qt, qt[1:])}
        db = {(a, b) for a, b in zip(dt, dt[1:])}
        bigram = len(qb & db) / len(qb) if qb else 0.0
        nums_q = set(re.findall(r"\d+(?:[.,]\d+)?", query))
        num = 1.0 if nums_q and all(n in text for n in nums_q) else 0.0
        tov = self.coverage(query, title) if title else 0.0
        # sentence-window feature: a passage whose single best sentence answers the question beats one that only
        # mentions the query words scattered across several sentences (a "small-to-big" signal)
        sents = split_sentences(text)
        sent = max((self.coverage(query, x) for x in sents), default=0.0) if len(sents) > 1 else cov
        final = (0.35 * cov + 0.25 * sent + 0.12 * bigram + 0.08 * tov + 0.10 * max(dense, 0.0) + 0.10 * num)
        return {"score": final, "coverage": cov, "bigram": bigram, "title": tov, "sentence": sent}


class CrossEncoderReranker:  # pragma: no cover - optional
    name = "cross-encoder"

    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(model)

    def score_pairs(self, query, texts):
        return list(self.m.predict([(query, t) for t in texts]))
