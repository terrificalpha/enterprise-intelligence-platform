"""Agent tools: document search, data analysis, churn prediction, calculator, report generation."""
from __future__ import annotations

import ast
import math
import operator as op
import re
import time
from pathlib import Path

import pandas as pd

from ..config import settings
from ..ml.predictor import ChurnPredictor
from ..rag.knowledge_base import KnowledgeBase
from ..llm.generator import Generator
from ..llm.insights import InsightGenerator

# ------------------------------------------------------------------ safe calculator
_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_UN = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {"round": round, "min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "log": math.log}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic WITHOUT eval(): only whitelisted AST nodes are allowed."""
    expr = expr.replace(",", "").replace("$", "").replace("^", "**")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", expr, flags=re.I)
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    if len(expr) > 200:
        raise ValueError("Expression too long")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 100:
                raise ValueError("Exponent too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("Unsupported expression")
    return ev(ast.parse(expr.strip(), mode="eval"))


# ------------------------------------------------------------------ tool registry
TOOL_SPECS = [
    {"name": "search_documents", "description": "Search the policy/contract/report/notes knowledge base (hybrid search + reranking). Returns passages with sources.",
     "args": {"query": "string", "k": "int (default 5)", "customer_id": "optional string"}},
    {"name": "analyze_data", "description": "Portfolio analytics over the customer table. operation in [overview, churn_by, top_risk, revenue_at_risk, correlations, segment_stats].",
     "args": {"operation": "string", "column": "optional (industry|region|company_size|contract_type)", "n": "int", "filters": "optional dict",
              "metric": "optional numeric column", "agg": "mean|sum|median|count|max|min"}},
    {"name": "predict_churn", "description": "ML churn probability, risk tier and top drivers for one customer id (e.g. C0010).",
     "args": {"customer_id": "string"}},
    {"name": "account_briefing", "description": "Full GenAI briefing for a customer: ML prediction + retrieved policy evidence + recommendations with citations.",
     "args": {"customer_id": "string"}},
    {"name": "calculate", "description": "Safe arithmetic calculator. Supports + - * / ** %, 'x% of y', round/min/max/sqrt.",
     "args": {"expression": "string"}},
    {"name": "generate_report", "description": "Create a Markdown + DOCX risk report (portfolio KPIs, top risks, evidence). Optional customer deep-dive.",
     "args": {"title": "string", "customer_id": "optional string", "top_n": "int"}},
]

GROUP_COLS = {"industry", "region", "company_size", "contract_type", "risk_tier"}
NUM_COLS = {"tenure_months", "annual_contract_value", "seats_purchased", "seats_active", "logins_30d", "usage_trend_pct",
            "feature_adoption_score", "support_tickets_90d", "avg_resolution_hours", "sla_breaches_90d", "nps_score",
            "late_payments_12m", "days_sales_outstanding", "discount_pct", "competitor_mentions_90d", "days_to_renewal",
            "churn_probability", "expected_loss", "churned"}


def _fmt_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    return df.head(max_rows).to_markdown(index=False, floatfmt=",.2f")


class ToolBox:
    def __init__(self, kb: KnowledgeBase, predictor: ChurnPredictor, generator: Generator, insights: InsightGenerator):
        self.kb, self.pred, self.gen, self.ins = kb, predictor, generator, insights
        self.specs = TOOL_SPECS

    def call(self, name: str, **args) -> dict:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return {"error": f"Unknown tool '{name}'", "ok": False}
        t0 = time.perf_counter()
        try:
            out = fn(**args)
            out.setdefault("ok", True)
        except Exception as e:   # tools never crash the agent; errors become observations
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    # ---- 1. documents
    def tool_search_documents(self, query: str, k: int = 5, customer_id: str | None = None, mode: str = "hybrid_rerank") -> dict:
        hits = self.kb.search(query, k=int(k), mode=mode, filters={"customer_id": customer_id} if customer_id else None)
        sup = self.kb.assess_support(query, hits)
        return {"hits": hits, "support": sup, "summary": f"{len(hits)} passages; evidence confidence {sup['confidence']:.0%} "
                f"({'supported' if sup['supported'] else 'insufficient'}). Top: {hits[0].chunk.source} {hits[0].chunk.location}" if hits else "no passages"}

    # ---- 2. data analysis
    def _frame(self) -> pd.DataFrame:
        return self.pred.portfolio

    @staticmethod
    def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
        for col, val in (filters or {}).items():
            if col in GROUP_COLS:
                df = df[df[col].astype(str).str.lower() == str(val).lower()]
        return df

    def tool_analyze_data(self, operation: str, column: str | None = None, n: int = 10, filters: dict | None = None,
                          metric: str | None = None, agg: str = "mean") -> dict:
        df = self._apply_filters(self._frame(), filters)
        if df.empty:
            return {"ok": False, "error": "No rows match the filters"}
        n = max(1, min(int(n), 50))
        if operation == "overview":
            tiers = df["risk_tier"].value_counts().to_dict()
            res = {"customers": len(df), "actual_churn_rate": round(float(df["churned"].mean()), 4),
                   "avg_predicted_churn": round(float(df["churn_probability"].mean()), 4),
                   "total_acv": float(df["annual_contract_value"].sum()),
                   "expected_loss": float(df["expected_loss"].sum()), "tier_counts": tiers}
            return {"operation": operation, "result": res, "summary":
                    f"{len(df):,} customers; actual churn {res['actual_churn_rate']:.1%}; High/Medium/Low = "
                    f"{tiers.get('High', 0)}/{tiers.get('Medium', 0)}/{tiers.get('Low', 0)}; expected annual revenue loss "
                    f"${res['expected_loss']:,.0f} of ${res['total_acv']:,.0f} ACV."}
        if operation == "churn_by":
            col = column if column in GROUP_COLS else "industry"
            t = df.groupby(col).agg(customers=("customer_id", "count"), actual_churn_rate=("churned", "mean"),
                                    avg_predicted_churn=("churn_probability", "mean"),
                                    total_acv=("annual_contract_value", "sum")).sort_values("actual_churn_rate", ascending=False).reset_index()
            top = t.iloc[0]
            return {"operation": operation, "table": t.round(4).to_dict("records"), "markdown": _fmt_table(t),
                    "summary": f"Highest churn by {col}: {top[col]} ({top.actual_churn_rate:.1%}, {int(top.customers)} customers); "
                               f"lowest: {t.iloc[-1][col]} ({t.iloc[-1].actual_churn_rate:.1%})."}
        if operation == "top_risk":
            t = df.sort_values("churn_probability", ascending=False).head(n)[
                ["customer_id", "industry", "region", "company_size", "annual_contract_value", "churn_probability", "risk_tier", "expected_loss"]]
            return {"operation": operation, "table": t.round(4).to_dict("records"), "markdown": _fmt_table(t, n),
                    "summary": f"Top {len(t)} accounts by churn probability (highest {t.churn_probability.iloc[0]:.0%}); "
                               f"combined expected loss ${t.expected_loss.sum():,.0f}."}
        if operation == "revenue_at_risk":
            col = column if column in GROUP_COLS else "risk_tier"
            t = df.groupby(col).agg(customers=("customer_id", "count"), total_acv=("annual_contract_value", "sum"),
                                    expected_loss=("expected_loss", "sum")).sort_values("expected_loss", ascending=False).reset_index()
            hi = df[df.risk_tier == "High"]
            return {"operation": operation, "table": t.round(2).to_dict("records"), "markdown": _fmt_table(t),
                    "summary": f"Expected annual revenue loss ${df.expected_loss.sum():,.0f}; High-risk accounts hold "
                               f"${hi.annual_contract_value.sum():,.0f} ACV ({len(hi)} customers). Largest by {col}: {t.iloc[0][col]} (${t.iloc[0].expected_loss:,.0f})."}
        if operation == "correlations":
            num = [c for c in NUM_COLS if c in df and c not in ("churned", "churn_probability", "expected_loss")]
            cor = df[num + ["churned"]].corr()["churned"].drop("churned").sort_values(key=abs, ascending=False)
            t = cor.head(8).rename("correlation_with_churn").reset_index().rename(columns={"index": "feature"})
            return {"operation": operation, "table": t.round(3).to_dict("records"), "markdown": _fmt_table(t),
                    "summary": "Strongest correlates of churn: " + ", ".join(f"{r.feature} ({r.correlation_with_churn:+.2f})" for r in t.head(4).itertuples())}
        if operation == "segment_stats":
            if metric not in NUM_COLS:
                return {"ok": False, "error": f"metric must be one of {sorted(NUM_COLS)}"}
            if agg not in ("mean", "sum", "median", "count", "max", "min"):
                return {"ok": False, "error": "bad agg"}
            if column in GROUP_COLS:
                t = df.groupby(column)[metric].agg(agg).sort_values(ascending=False).reset_index()
                return {"operation": operation, "table": t.round(3).to_dict("records"), "markdown": _fmt_table(t),
                        "summary": f"{agg} of {metric} by {column}: highest {t.iloc[0][column]} ({t.iloc[0][metric]:,.2f})."}
            v = float(getattr(df[metric], agg)())
            return {"operation": operation, "result": v, "summary": f"{agg} of {metric} = {v:,.2f} over {len(df):,} customers."}
        return {"ok": False, "error": f"Unknown operation '{operation}'"}

    # ---- 3. ML
    def tool_predict_churn(self, customer_id: str) -> dict:
        p = self.pred.predict_customer(customer_id)
        if p is None:
            return {"ok": False, "error": f"Customer '{customer_id}' not found"}
        drv = ", ".join(f"{d['label']} ({d['impact'] * 100:+.1f} pts)" for d in p["drivers"][:4])
        return {"prediction": {k: p[k] for k in ("customer_id", "churn_probability", "risk_tier", "predicted_churn")},
                "drivers": p["drivers"], "profile": p["profile"],
                "summary": f"{p['customer_id']}: churn probability {p['churn_probability']:.0%} -> {p['risk_tier']} risk. Drivers: {drv}."}

    def tool_account_briefing(self, customer_id: str) -> dict:
        b = self.ins.briefing(customer_id)
        if "error" in b:
            return {"ok": False, "error": b["error"]}
        return {"briefing": b, "summary": f"Briefing for {b['customer_id']}: {b['prediction']['risk_tier']} risk "
                f"({b['prediction']['churn_probability']:.0%}), {len(b['recommendations'])} recommended actions."}

    # ---- 4. calculator
    def tool_calculate(self, expression: str) -> dict:
        v = safe_eval(expression)
        return {"expression": expression, "value": v, "summary": f"{expression} = {v:,.6g}"}

    # ---- 5. report
    def tool_generate_report(self, title: str = "Enterprise Risk Report", customer_id: str | None = None, top_n: int = 10) -> dict:
        from .report import build_report
        return build_report(self, title, customer_id, int(top_n))
