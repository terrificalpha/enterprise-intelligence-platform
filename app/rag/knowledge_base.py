"""Knowledge base: ingestion -> embeddings -> vector store + BM25 -> hybrid retrieval -> reranking -> support check."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import settings
from ..ingestion.loaders import Chunk, load_directory, load_file
from ..security import neutralise_context
from ..lexicon import expand_terms
from ..text_utils import split_sentences, tokenize
from .bm25 import BM25
from .embeddings import make_embedder
from .reranker import HeuristicReranker

RRF_K = 60
CUSTOMER_RE = re.compile(r"\bC\d{4}\b", re.I)
RECORD_INTENT_RE = re.compile(r"\b(tickets?|notes?|cases?|records?|logged|complain\w*)\b", re.I)
# label+number identifiers such as "Net 90", "P0", "Q4", "Tier 3": if the question names one that never appears
# in the retrieved evidence, the corpus cannot answer it (it only has *other* values of the same label).
IDENT_RE = re.compile(r"\b(net|p|q|tier|level|plan|phase|sev|severity|version|v)\s?(\d{1,3})\b", re.I)
# questions that expect a figure ("how many", "what percentage", "pricing"...)
NUMERIC_Q_RE = re.compile(r"\bhow (many|much|long|soon|often|fast|quickly)\b|\bwhat (percent|percentage|rate|price|pricing|cost|salary)\b|"
                          r"\b(pricing|price|salary|number of|per seat|per license)\b", re.I)
_NUMWORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve", "twenty", "thirty",
             "hundred", "half", "double", "percent"}
RECORD_PENALTY = 0.85   # authority prior: policies/contracts outrank row-level records unless records are asked for


@dataclass
class Hit:
    chunk: Chunk
    score: float                       # final ranking score
    dense: float = 0.0
    bm25: float = 0.0
    rerank: float = 0.0
    coverage: float = 0.0
    rank_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"chunk_id": self.chunk.chunk_id, "source": self.chunk.source, "location": self.chunk.location,
             "section": self.chunk.section, "text": self.chunk.text, "score": round(self.score, 4),
             "dense": round(self.dense, 4), "bm25": round(self.bm25, 4), "rerank": round(self.rerank, 4),
             "coverage": round(self.coverage, 4)}
        if self.chunk.customer_id:
            d["customer_id"] = self.chunk.customer_id
        return d


class KnowledgeBase:
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir) if index_dir else None   # injectable so tests never touch the real index
        self.chunks: list[Chunk] = []
        self.emb = None
        self.matrix = np.zeros((0, 1))
        self.bm25: BM25 | None = None
        self.reranker: HeuristicReranker | None = None
        self._lock = threading.RLock()
        self.threshold = settings.support_threshold

    # ---------------------------------------------------------------- build / persist
    @property
    def index_path(self) -> Path:
        return self.index_dir or (settings.artifacts_dir / "kb_index")

    def build(self, directory: Path | None = None, chunks: list[Chunk] | None = None) -> int:
        with self._lock:
            self.chunks = chunks if chunks is not None else load_directory(directory)
            self._index()
            self.save()
            return len(self.chunks)

    def _index(self) -> None:
        texts = [c.text for c in self.chunks]
        self.emb = make_embedder(settings.embedding_backend, settings.st_model).fit(texts)
        self.matrix = self.emb.encode(texts).astype(np.float32)
        self.bm25 = BM25().fit(texts)
        self.reranker = HeuristicReranker(self.bm25.idf, self.bm25.max_idf)

    def add_file(self, path: Path) -> int:
        """Ingest one new file and rebuild the (small) index so the new content is searchable immediately."""
        with self._lock:
            new = load_file(path)
            self.chunks = [c for c in self.chunks if c.source != Path(path).name] + new
            self._index()
            self.save()
            return len(new)

    def save(self) -> None:
        import joblib
        import time
        self.index_path.mkdir(parents=True, exist_ok=True)
        np.save(self.index_path / "matrix.npy", self.matrix)
        (self.index_path / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks]))
        joblib.dump(self.emb, self.index_path / "embedder.joblib")
        meta = {
            "embedding_backend": getattr(self.emb, "name", settings.embedding_backend),
            "model_name": getattr(self.emb, "model_name", settings.st_model if settings.embedding_backend.startswith("sentence") else "lsa"),
            "dim": int(self.matrix.shape[1]) if self.matrix.ndim > 1 and self.matrix.shape[0] > 0 else 0,
            "chunk_count": len(self.chunks),
            "document_count": len(self.documents()),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.index_path / "index_meta.json").write_text(json.dumps(meta, indent=2))

    def load(self) -> bool:
        import joblib
        if not (self.index_path / "chunks.json").exists() or not (self.index_path / "matrix.npy").exists():
            return False
        meta_file = self.index_path / "index_meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                if meta.get("embedding_backend") and meta["embedding_backend"] != settings.embedding_backend:
                    print(f"[knowledge_base] Stored embedding backend '{meta['embedding_backend']}' "
                          f"does not match configured '{settings.embedding_backend}'; rebuilding index.")
                    return False
            except Exception:
                pass
        emb_file = self.index_path / "embedder.joblib"
        if not emb_file.exists():
            return False
        raw = json.loads((self.index_path / "chunks.json").read_text())
        self.chunks = [Chunk(**d) for d in raw]
        self.matrix = np.load(self.index_path / "matrix.npy")
        if self.matrix.shape[0] != len(self.chunks):
            print("[knowledge_base] Matrix row count does not match chunk count; rebuilding index.")
            return False
        try:
            self.emb = joblib.load(emb_file)
            self.emb.encode(["healthcheck"])      # unpickles fine but may still break on a scikit-learn version change
        except Exception as e:  # noqa: BLE001
            print(f"[knowledge_base] saved embedder unusable ({type(e).__name__}); rebuilding index.")
            return False
        texts = [c.text for c in self.chunks]
        self.bm25 = BM25().fit(texts)
        self.reranker = HeuristicReranker(self.bm25.idf, self.bm25.max_idf)
        return True

    def ensure_ready(self) -> None:
        if not self.chunks and not self.load():
            self.build()

    # ---------------------------------------------------------------- search
    def documents(self) -> list[dict]:
        out: dict[str, dict] = {}
        for c in self.chunks:
            d = out.setdefault(c.source, {"source": c.source, "type": c.doc_type, "chunks": 0})
            d["chunks"] += 1
        return sorted(out.values(), key=lambda d: d["source"])

    def _mask(self, filters: dict | None) -> np.ndarray:
        m = np.ones(len(self.chunks), dtype=bool)
        if not filters:
            return m
        for i, c in enumerate(self.chunks):
            if filters.get("source") and c.source != filters["source"]:
                m[i] = False
            if filters.get("doc_type") and c.doc_type != filters["doc_type"]:
                m[i] = False
            if filters.get("customer_id") and c.customer_id.upper() != filters["customer_id"].upper():
                m[i] = False
        return m

    def search(self, query: str, k: int | None = None, mode: str = "hybrid_rerank",
               filters: dict | None = None) -> list[Hit]:
        """mode: dense | bm25 | hybrid | hybrid_rerank.  Returns top-k Hits (highest first)."""
        self.ensure_ready()
        k = k or settings.top_k
        cand_k = max(settings.candidate_k, k)
        mask = self._mask(filters)
        # customer id in the question => restrict to that customer's records (metadata filter)
        m = CUSTOMER_RE.search(query)
        if m and not (filters and filters.get("customer_id")):
            cid = m.group(0).upper()
            cmask = np.array([c.customer_id.upper() == cid for c in self.chunks])
            if cmask.any():
                mask = mask & cmask

        qv = self.emb.encode([query])[0]
        dense = self.matrix @ qv
        sparse = self.bm25.scores(query)
        dense = np.where(mask, dense, -np.inf)
        sparse = np.where(mask, sparse, -np.inf)

        def ranks(s: np.ndarray, n: int) -> dict[int, int]:
            order = np.argsort(-s)[:n]
            return {int(i): r for r, i in enumerate(order, 1) if np.isfinite(s[i])}

        r_d, r_s = ranks(dense, cand_k), ranks(sparse, cand_k)
        if mode == "dense":
            cand = {i: 1 / (RRF_K + r) for i, r in r_d.items()}
        elif mode == "bm25":
            cand = {i: 1 / (RRF_K + r) for i, r in r_s.items()}
        else:  # reciprocal rank fusion of dense + sparse
            cand = {}
            for rk in (r_d, r_s):
                for i, r in rk.items():
                    cand[i] = cand.get(i, 0.0) + 1 / (RRF_K + r)
        hits = []
        for i, sc in cand.items():
            c = self.chunks[i]
            h = Hit(c, sc, dense=float(dense[i]) if np.isfinite(dense[i]) else 0.0,
                    bm25=float(sparse[i]) if np.isfinite(sparse[i]) else 0.0,
                    rank_info={"dense_rank": r_d.get(i), "bm25_rank": r_s.get(i)})
            rr = self.reranker.score(query, c.text, c.section, h.dense)
            h.coverage = rr["coverage"]
            h.rerank = rr["score"]
            hits.append(h)
        if mode == "hybrid_rerank":
            fused_max = max((h.score for h in hits), default=1.0) or 1.0
            for h in hits:   # 75% reranker, 25% normalised fusion score
                h.score = 0.75 * h.rerank + 0.25 * (h.score / fused_max)
        if mode in ("hybrid", "hybrid_rerank") and not (m or RECORD_INTENT_RE.search(query)):
            for h in hits:
                if h.chunk.doc_type == "csv":
                    h.score *= RECORD_PENALTY
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    # ---------------------------------------------------------------- support / abstention
    def assess_support(self, query: str, hits: list[Hit]) -> dict:
        """Decide whether the evidence actually supports the question (guards against hallucination).
        Supported if the best passage covers enough of the query's information content."""
        if not hits:
            return {"supported": False, "confidence": 0.0, "reason": "no passages retrieved"}
        cm = CUSTOMER_RE.search(query)
        if cm and any(h.chunk.customer_id.upper() == cm.group(0).upper() for h in hits):
            return {"supported": True, "confidence": 0.9, "reason": f"records found for {cm.group(0).upper()}"}
        best = max(h.coverage for h in hits)
        # coverage can also be earned jointly by the top-3 passages (multi-hop style questions)
        q = set(tokenize(query))
        union = set()
        for h in hits[:3]:
            union |= set(tokenize(h.chunk.text))
        tot = sum(self.reranker._w(t) for t in q) or 1.0
        joint = sum(self.reranker._w(t) for t in q if t in union) / tot
        conf = 0.7 * best + 0.3 * joint
        if conf < self.threshold:
            return {"supported": False, "confidence": round(conf, 3),
                    "reason": f"retrieved passages cover only {conf:.0%} of the question (threshold {self.threshold:.0%})"}
        # ---- second-stage guards: lexical overlap alone cannot tell "answered" from "same topic, different fact"
        ev_text = " ".join(f"{h.chunk.source} {h.chunk.section} {h.chunk.text}" for h in hits[:5]).lower().replace("_", " ")
        for lab, num in IDENT_RE.findall(query):
            if not re.search(rf"\b{lab.lower()}\s?{num}\b", ev_text):
                return {"supported": False, "confidence": round(min(conf, self.threshold - 0.01), 3),
                        "reason": f"the question refers to '{lab} {num}', which does not appear in the retrieved evidence"}
        if NUMERIC_Q_RE.search(query):
            best_sent = ""
            best_cov = -1.0
            for h in hits[:3]:
                for sent in split_sentences(h.chunk.text):
                    c = self.reranker.coverage(query, sent)
                    if c > best_cov:
                        best_cov, best_sent = c, sent
            has_figure = bool(re.search(r"\d", best_sent)) or bool(_NUMWORDS & set(re.findall(r"[a-z]+", best_sent.lower())))
            if not has_figure:
                return {"supported": False, "confidence": round(min(conf, self.threshold - 0.01), 3),
                        "reason": "the question asks for a figure but the best-matching passage states none"}
        return {"supported": True, "confidence": round(conf, 3), "reason": "sufficient evidence"}

    @staticmethod
    def safe_text(hit: Hit) -> str:
        return neutralise_context(hit.chunk.text)
