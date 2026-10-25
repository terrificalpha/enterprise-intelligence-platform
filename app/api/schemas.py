from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=20)
    mode: Mode = "hybrid_rerank"
    source: str | None = None
    customer_id: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=10)
    mode: Mode = "hybrid_rerank"


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class PredictRequest(BaseModel):
    customer_id: str | None = Field(None, pattern=r"^[Cc]\d{4}$")
    record: dict | None = None


class ReportRequest(BaseModel):
    title: str = Field("Enterprise Risk Report", max_length=120)
    customer_id: str | None = Field(None, pattern=r"^[Cc]\d{4}$")
    top_n: int = Field(10, ge=1, le=50)
