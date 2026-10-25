"""Backend clients for the UI: HTTP client for the FastAPI service, or an in-process client (no server needed).
Select with EIP_UI_MODE=api|local (default api; falls back to local if the API is unreachable)."""
from __future__ import annotations

import os

import requests


class ApiClient:
    mode = "api"

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base = (base_url or os.environ.get("EIP_API_URL", "http://localhost:8000")).rstrip("/")
        self.headers = {"X-API-Key": api_key or os.environ.get("EIP_API_KEY", "")}

    def _req(self, method, path, **kw):
        r = requests.request(method, f"{self.base}{path}", headers=self.headers, timeout=120, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise RuntimeError(f"{r.status_code}: {detail}")
        return r

    def health(self): return self._req("GET", "/health").json()
    def documents(self): return self._req("GET", "/documents").json()
    def upload(self, name, data): return self._req("POST", "/documents/upload", files={"file": (name, data)}).json()
    def rebuild(self): return self._req("POST", "/documents/rebuild").json()
    def search(self, query, k=5, mode="hybrid_rerank", **kw): return self._req("POST", "/search", json={"query": query, "k": k, "mode": mode, **kw}).json()
    def ask(self, question, k=5, mode="hybrid_rerank"): return self._req("POST", "/ask", json={"question": question, "k": k, "mode": mode}).json()
    def predict(self, customer_id): return self._req("POST", "/predict", json={"customer_id": customer_id}).json()
    def briefing(self, customer_id): return self._req("GET", f"/customers/{customer_id}/briefing").json()
    def agent(self, question): return self._req("POST", "/agent", json={"question": question}).json()
    def portfolio(self, n=15): return self._req("GET", "/portfolio", params={"n": n}).json()
    def metrics(self): return self._req("GET", "/model/metrics").json()
    def report(self, title, customer_id=None, top_n=10): return self._req("POST", "/reports", json={"title": title, "customer_id": customer_id, "top_n": top_n}).json()
    def report_bytes(self, name): return self._req("GET", f"/reports/{name}").content


class LocalClient:
    """Same interface, calls the service layer directly (used by `make ui-local` and Streamlit Cloud)."""
    mode = "local"

    def __init__(self):
        from app.services import Platform
        self.p = Platform()

    def health(self): return self.p.health()
    def documents(self): return self.p.kb.documents()
    def upload(self, name, data): return self.p.ingest_bytes(name, data)
    def rebuild(self): return self.p.rebuild_index()
    def search(self, query, k=5, mode="hybrid_rerank", **kw): return self.p.search(query, k, mode, kw.get("source"), kw.get("customer_id"))
    def ask(self, question, k=5, mode="hybrid_rerank"): return self.p.ask(question, k, mode)
    def predict(self, customer_id):
        r = self.p.predict(customer_id)
        if r is None:
            raise RuntimeError(f"404: Customer {customer_id} not found")
        return r
    def briefing(self, customer_id):
        b = self.p.briefing(customer_id)
        if "error" in b:
            raise RuntimeError("404: " + b["error"])
        return b
    def agent(self, question): return self.p.agent_run(question)
    def portfolio(self, n=15): return self.p.portfolio(n)
    def metrics(self): return self.p.model_metrics()
    def report(self, title, customer_id=None, top_n=10):
        return self.p.tools.call("generate_report", title=title, customer_id=customer_id, top_n=top_n)
    def report_bytes(self, name):
        path = self.p.report_path(name)
        return path.read_bytes() if path else b""


def get_client():
    mode = os.environ.get("EIP_UI_MODE", "api").lower()
    if mode == "local":
        return LocalClient()
    c = ApiClient()
    try:
        c.health()
        return c
    except Exception:
        return LocalClient()     # graceful fallback so the demo always works
