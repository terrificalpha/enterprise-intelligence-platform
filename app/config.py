"""Central configuration. All values can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings:
    # --- paths -----------------------------------------------------------
    root: Path = ROOT
    data_dir: Path = Path(_env("EIP_DATA_DIR", str(ROOT / "data")))
    raw_dir: Path = data_dir / "raw"
    documents_dir: Path = Path(_env("EIP_DOCS_DIR", str(data_dir / "documents")))
    eval_dir: Path = data_dir / "eval"
    artifacts_dir: Path = Path(_env("EIP_ARTIFACTS_DIR", str(ROOT / "artifacts")))
    reports_dir: Path = Path(_env("EIP_REPORTS_DIR", str(ROOT / "reports")))
    customers_csv: Path = raw_dir / "customers.csv"

    # --- ML --------------------------------------------------------------
    random_state: int = int(_env("EIP_SEED", "42"))
    high_risk_threshold: float = 0.60   # must match Customer Retention Policy s.2
    medium_risk_threshold: float = 0.30

    # --- RAG -------------------------------------------------------------
    embedding_backend: str = _env("EIP_EMBEDDING", "lsa")  # lsa | ollama | sentence-transformers
    st_model: str = _env("EIP_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_chars: int = int(_env("EIP_CHUNK_CHARS", "700"))
    chunk_overlap: int = int(_env("EIP_CHUNK_OVERLAP", "100"))
    top_k: int = int(_env("EIP_TOP_K", "5"))
    candidate_k: int = int(_env("EIP_CANDIDATE_K", "40"))
    support_threshold: float = float(_env("EIP_SUPPORT_THRESHOLD", "0.35"))        # offline mode
    support_threshold_llm: float = float(_env("EIP_SUPPORT_THRESHOLD_LLM", "0.30"))  # LLM also judges sufficiency

    # --- LLM -------------------------------------------------------------
    llm_provider: str = _env("LLM_PROVIDER", _env("EIP_LLM_PROVIDER", "ollama"))  # ollama | anthropic | openai | offline | auto
    ollama_base_url: str = _env("OLLAMA_BASE_URL", _env("EIP_OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    ollama_model: str = _env("OLLAMA_MODEL", _env("EIP_OLLAMA_MODEL", "qwen2.5:3b"))
    anthropic_model: str = _env("EIP_ANTHROPIC_MODEL", "claude-sonnet-5-5")
    openai_model: str = _env("EIP_OPENAI_MODEL", "gpt-4o-mini")
    # first call after Ollama starts has to load the model into VRAM (can take 20-60 s on a GTX 1650), so be generous
    llm_timeout: int = int(_env("EIP_LLM_TIMEOUT", "180"))
    ollama_num_ctx: int = int(_env("OLLAMA_NUM_CTX", "4096"))          # explicit: old Ollama defaults (2048) silently truncate prompts
    ollama_keep_alive: str = _env("OLLAMA_KEEP_ALIVE", "30m")          # keep the model resident between requests
    ollama_embed_model: str = _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")  # only used when EIP_EMBEDDING=ollama

    # --- API / security --------------------------------------------------
    api_key: str = _env("EIP_API_KEY", "")           # empty = auth disabled (dev)
    api_url: str = _env("EIP_API_URL", "http://localhost:8000")
    max_query_chars: int = 1000
    max_upload_mb: int = int(_env("EIP_MAX_UPLOAD_MB", "10"))
    rate_limit_per_min: int = int(_env("EIP_RATE_LIMIT", "120"))

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.documents_dir, self.eval_dir, self.artifacts_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
