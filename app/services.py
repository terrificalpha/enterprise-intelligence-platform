"""Framework-independent application service layer. FastAPI, Streamlit (local mode), the CLI and the tests all use it."""
from __future__ import annotations

import time
from functools import cached_property
from pathlib import Path

from .agent.agent import EnterpriseAgent
from .agent.tools import ToolBox
from .config import settings
from .llm.client import LLMClient
from .llm.generator import Generator
from .llm.insights import InsightGenerator
from .ml.predictor import ChurnPredictor
from .rag.knowledge_base import KnowledgeBase
from .security import safe_filename, check_upload_size, validate_query


class Platform:
    def __init__(self):
        settings.ensure_dirs()
        self.llm = LLMClient()
        self.kb = KnowledgeBase()
        self.kb.ensure_ready()
        if self.llm.available:   # the LLM performs a second, semantic sufficiency check -> lenient pre-filter
            self.kb.threshold = settings.support_threshold_llm

    @cached_property
    def predictor(self) -> ChurnPredictor:
        return ChurnPredictor()

    @cached_property
    def generator(self) -> Generator:
        return Generator(self.kb, self.llm)

    @cached_property
    def insights(self) -> InsightGenerator:
        return InsightGenerator(self.kb, self.predictor, self.generator, self.llm)

    @cached_property
    def tools(self) -> ToolBox:
        return ToolBox(self.kb, self.predictor, self.generator, self.insights)

    @cached_property
    def agent(self) -> EnterpriseAgent:
        return EnterpriseAgent(self.tools, self.generator, self.llm)

    # ------------------------------------------------------------------ operations
    def health(self) -> dict:
        return {"status": "ok", "llm_provider": self.llm.provider, "llm_model": self.llm.model_name,
                "llm_available": self.llm.available, "llm_hint": self.llm.status().get("hint", ""), "documents": len(self.kb.documents()), "chunks": len(self.kb.chunks),
                "embedding": getattr(self.kb.emb, "name", "n/a"),
                "model": self.predictor.info if (settings.artifacts_dir / "best_model.joblib").exists() else None}

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", source: str | None = None,
               customer_id: str | None = None) -> dict:
        query = validate_query(query)
        t0 = time.perf_counter()
        filters = {k_: v for k_, v in (("source", source), ("customer_id", customer_id)) if v}
        hits = self.kb.search(query, k=k, mode=mode, filters=filters or None)
        return {"query": query, "mode": mode, "support": self.kb.assess_support(query, hits),
                "results": [h.to_dict() for h in hits], "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def ask(self, question: str, k: int = 5, mode: str = "hybrid_rerank") -> dict:
        question = validate_query(question)
        t0 = time.perf_counter()
        hits = self.kb.search(question, k=k, mode=mode)
        res = self.generator.answer(question, hits)
        res["retrieved"] = [h.to_dict() for h in hits]
        res["question"], res["mode"] = question, mode
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res

    def predict(self, customer_id: str) -> dict | None:
        r = self.predictor.predict_customer(customer_id)
        return r

    def predict_record(self, record: dict) -> dict:
        res = self.predictor.predict_record(record)
        res["drivers"] = self.predictor.explain(record)
        return res

    def briefing(self, customer_id: str) -> dict:
        return self.insights.briefing(customer_id)

    def agent_run(self, question: str) -> dict:
        t0 = time.perf_counter()
        res = self.agent.run(question)
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res

    def ingest_bytes(self, filename: str, data: bytes) -> dict:
        name = safe_filename(filename)
        check_upload_size(len(data))
        path = settings.documents_dir / name
        path.write_bytes(data)
        try:
            n = self.kb.add_file(path)
        except Exception as e:
            path.unlink(missing_ok=True)       # do not keep files we cannot parse
            raise ValueError(f"Could not parse '{name}': {e}") from e
        return {"filename": name, "chunks_added": n, "total_chunks": len(self.kb.chunks)}

    def rebuild_index(self) -> dict:
        n = self.kb.build()
        return {"chunks": n, "documents": self.kb.documents()}

    def model_metrics(self) -> dict:
        return ChurnPredictor.metrics()

    def portfolio(self, n: int = 15) -> dict:
        tb = self.tools
        return {"overview": tb.tool_analyze_data("overview"), "by_industry": tb.tool_analyze_data("churn_by", column="industry"),
                "by_region": tb.tool_analyze_data("churn_by", column="region"),
                "revenue": tb.tool_analyze_data("revenue_at_risk", column="industry"),
                "top_risk": tb.tool_analyze_data("top_risk", n=n)}

    def list_reports(self) -> list[str]:
        return sorted(p.name for p in settings.reports_dir.glob("*") if p.suffix in (".md", ".docx"))

    def report_path(self, name: str) -> Path | None:
        p = (settings.reports_dir / Path(name).name).resolve()      # Path(name).name blocks traversal
        return p if p.exists() and p.parent == settings.reports_dir.resolve() else None
