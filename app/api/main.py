"""FastAPI backend. Run:  uvicorn app.api.main:app --port 8000   (docs at /docs)"""
from __future__ import annotations

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ..config import settings
from ..security import RateLimiter, SecurityError, check_api_key
from ..services import Platform
from .schemas import AgentRequest, AskRequest, PredictRequest, ReportRequest, SearchRequest

app = FastAPI(title="Autonomous Enterprise Intelligence Platform", version="1.0.0",
              description="ML churn prediction + document intelligence + hybrid RAG + GenAI + tool-using agent.")
_limiter = RateLimiter()
_platform: Platform | None = None


def platform() -> Platform:
    global _platform
    if _platform is None:
        _platform = Platform()
    return _platform


def guard(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    if not check_api_key(x_api_key):
        raise HTTPException(401, "Invalid or missing X-API-Key")
    client = request.client.host if request.client else "unknown"
    if not _limiter.allow(client):
        raise HTTPException(429, "Rate limit exceeded")


def _bad(e: Exception):
    raise HTTPException(400, str(e))


@app.get("/health")
def health():
    return platform().health()


@app.get("/documents", dependencies=[Depends(guard)])
def documents():
    return platform().kb.documents()


@app.post("/documents/upload", dependencies=[Depends(guard)])
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return platform().ingest_bytes(file.filename or "", data)
    except (SecurityError, ValueError) as e:
        _bad(e)


@app.post("/documents/rebuild", dependencies=[Depends(guard)])
def rebuild():
    return platform().rebuild_index()


@app.post("/search", dependencies=[Depends(guard)])
def search(req: SearchRequest):
    try:
        return platform().search(req.query, req.k, req.mode, req.source, req.customer_id)
    except SecurityError as e:
        _bad(e)


@app.post("/ask", dependencies=[Depends(guard)])
def ask(req: AskRequest):
    try:
        return platform().ask(req.question, req.k, req.mode)
    except SecurityError as e:
        _bad(e)


@app.post("/predict", dependencies=[Depends(guard)])
def predict(req: PredictRequest):
    p = platform()
    if req.customer_id:
        r = p.predict(req.customer_id)
        if r is None:
            raise HTTPException(404, f"Customer {req.customer_id} not found")
        return r
    if req.record:
        try:
            return p.predict_record(req.record)
        except Exception as e:
            _bad(e)
    raise HTTPException(422, "Provide customer_id or record")


@app.get("/customers/{customer_id}/briefing", dependencies=[Depends(guard)])
def briefing(customer_id: str):
    b = platform().briefing(customer_id)
    if "error" in b:
        raise HTTPException(404, b["error"])
    return b


@app.post("/agent", dependencies=[Depends(guard)])
def agent(req: AgentRequest):
    try:
        return platform().agent_run(req.question)
    except SecurityError as e:
        _bad(e)


@app.get("/portfolio", dependencies=[Depends(guard)])
def portfolio(n: int = 15):
    return platform().portfolio(n)


@app.get("/model/metrics", dependencies=[Depends(guard)])
def model_metrics():
    return platform().model_metrics()


@app.post("/reports", dependencies=[Depends(guard)])
def make_report(req: ReportRequest):
    res = platform().tools.call("generate_report", title=req.title, customer_id=req.customer_id.upper() if req.customer_id else None, top_n=req.top_n)
    if not res.get("ok", True):
        raise HTTPException(400, res.get("error", "report failed"))
    return res


@app.get("/reports", dependencies=[Depends(guard)])
def reports():
    return platform().list_reports()


@app.get("/reports/{name}", dependencies=[Depends(guard)])
def download_report(name: str):
    p = platform().report_path(name)
    if not p:
        raise HTTPException(404, "Report not found")
    return FileResponse(p)
