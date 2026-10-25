# Autonomous Enterprise Intelligence Platform - ML + GenAI

An end-to-end AI application that combines **machine learning**, **document intelligence**, **advanced RAG** and a
**tool-using AI agent** to analyse B2B customer data, identify churn risk, retrieve supporting evidence from company
documents, and generate cited, actionable recommendations.

> Built for the *"Autonomous Enterprise Intelligence Platform - ML + GenAI"* assignment. Every requirement is mapped to
> code, tests and results in [`docs/REQUIREMENTS_TRACEABILITY.md`](docs/REQUIREMENTS_TRACEABILITY.md).

**100% local, zero paid APIs.** The GenAI layer runs on a local [Ollama](https://ollama.com) model (default `qwen2.5:3b`,
sized for a 4 GB GTX 1650 / 16 GB RAM laptop). If Ollama is not running, the platform automatically falls back to a
deterministic, cited offline generator - so the app never breaks. (Anthropic/OpenAI keys remain *optional*.)

---------------------------------------------------------------------------------------------------

## Quick start (Windows + Ollama, recommended)

1. Install Python 3.10+ and [Ollama](https://ollama.com/download) (it auto-starts in the system tray).
2. In the project folder: `setup_windows.bat` (creates `.venv`, installs packages, pulls `qwen2.5:3b`, trains models,
   builds the index and runs a live Ollama check). Then `run_windows.bat` starts API + UI.
3. Manual equivalent:

```bash
ollama pull qwen2.5:3b
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
python scripts/bootstrap.py --skip-eval        # data -> models -> index
python scripts/check_ollama.py                 # server up? model installed? cited answer + refusal work?
uvicorn app.api.main:app --port 8000           # terminal 1
streamlit run ui/streamlit_app.py              # terminal 2  (sidebar shows "LLM Active: ollama/qwen2.5:3b")
```

## Quick start (generic)

```bash
git clone <your-repo-url> && cd eip
python -m venv .venv && source .venv/bin/activate          # Python 3.10+ (developed on 3.13)
pip install -r requirements.txt

python scripts/bootstrap.py        # generate data -> train 3 models -> build index -> evaluate  (~2 min)

make api                           # terminal 1: FastAPI on :8000   (Swagger UI: http://localhost:8000/docs)
make ui                            # terminal 2: Streamlit on :8501
# or, one process, no API server:  make ui-local
# or Docker:                       docker compose up --build        (UI :8501, API :8000)
```

Optional configuration: copy `.env.example` to `.env`. Useful switches:

| Variable | Effect |
|---|---|
| `LLM_PROVIDER=ollama` (default) / `offline` | local Ollama LLM, or force the deterministic generator |
| `OLLAMA_MODEL=qwen2.5:3b` | any installed model, e.g. `llama3.2:3b`, `qwen2.5:1.5b` (faster), `qwen2.5:7b` (slow on 4 GB VRAM) |
| `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE`, `EIP_LLM_TIMEOUT` | context size (4096), keep model loaded (30m), timeout (180 s) |
| `EIP_EMBEDDING=ollama` | neural embeddings via `ollama pull nomic-embed-text` (default stays LSA) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | *optional* cloud providers (not needed) |
| `EIP_EMBEDDING=sentence-transformers` | neural embeddings (needs `pip install sentence-transformers`) |
| `EIP_API_KEY=secret` | require `X-API-Key` on every protected endpoint |
| `EIP_UI_MODE=local` | UI calls the service layer directly (no API process) |

Run the tests: `make test` (about 80 tests, ~10 s; no Ollama needed - a fake Ollama server is used).  Re-run the evaluation: `make eval` -> `docs/evaluation_results.md`.

---------------------------------------------------------------------------------------------------

## What it does

| Capability | Where | Highlights |
|---|---|---|
| **1. Data analysis & ML** | `app/ml/` | cleaning (dedupe, outliers, casing, impossible values, missing data), 10 engineered features, **3 models + baseline** tuned with CV, compared, best selected on CV (test set untouched), calibrated, threshold tuned, explained |
| **2. Document intelligence** | `app/ingestion/` | **PDF** (pages + headings), **DOCX** (headings + tables), **CSV** (row records + customer-id metadata); chunks keep page/section/row |
| **3. Advanced RAG** | `app/rag/` | embeddings + vector search, **BM25**, **hybrid (RRF)**, **reranking**, synonym query expansion, metadata filters, **source citations**, evidence-sufficiency gate |
| **4. GenAI integration** | `app/llm/` | local Ollama LLM (or offline composer), with output validation (citations, numbers, tier) and automatic fallback; account briefing = **ML prediction + drivers + retrieved policy -> cited recommendations** |
| **5. AI agent** | `app/agent/` | picks among 6 tools (document search, data analysis, churn prediction, calculator, briefing, report generation); full trace |
| **6. Evaluation & reliability** | `app/evaluation/` | ML metrics, retrieval ablation (Hit@k/MRR), citation validity/support, refusal of unsupported questions, agent tool-selection, blind held-out set |
| **7. Application** | `app/api/`, `ui/` | FastAPI backend + 7-tab Streamlit UI |
| **8. Testing & docs** | `tests/`, `docs/` | 61 tests (unit, integration, agent, security, API, UI smoke); architecture, evaluation, demo script, traceability |
| **Bonus** | | hybrid search, reranking, Docker, security controls (auth, rate limit, injection defence, PII redaction, upload hardening) |

### The story the app tells (try it in the UI)
1. **Dashboard** - 3,000 accounts, churn 27.1%, expected revenue loss, top at-risk accounts.
2. **Account Insight** - pick the riskiest account -> ML probability + top drivers -> policy passages retrieved for *those* drivers -> numbered, cited recommendations -> download DOCX/MD report.
3. **Ask (RAG)** - *"Who approves a retention discount of 15 percent?"* -> answer with `[1]` citation + citation check. *"What is our parental leave policy?"* -> refuses.
4. **Agent** - *"Which industry has the highest churn and what does the policy say about High Risk escalation?"* -> runs analytics **and** document search, shows the tool trace.

---------------------------------------------------------------------------------------------------

## Results (from `docs/evaluation_results.md`, regenerated by `make eval`)

**ML** - 3,000 customers, 27.1% churn, stratified 80/20 split, 5-fold CV

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|---|
| Baseline (prior) | 0.500 | 0.500 | 0.272 | 0.000 |
| **Logistic Regression (selected)** | **0.852** | **0.860** | **0.709** | **0.657** |
| Hist. Gradient Boosting | 0.838 | 0.843 | 0.677 | 0.643 |
| Random Forest | 0.825 | 0.824 | 0.642 | 0.582 |

Risk tiers are well ordered on the held-out test set: observed churn **10.7% (Low) -> 49.6% (Medium) -> 75.0% (High)**.

**Retrieval** (38 answerable questions)

| Mode | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| Dense (LSA) | 84.2% | 92.1% | 0.879 |
| BM25 | 94.7% | 97.4% | 0.967 |
| Hybrid (RRF) | 89.5% | 97.4% | 0.934 |
| **Hybrid + rerank** | **97.4%** | **100%** | **0.987** |

On this small, term-heavy corpus BM25 alone is already strong and plain RRF fusion is slightly *worse* than BM25 at Hit@1; it is the reranker (sentence-level evidence matching) that makes the hybrid pipeline the best.

**Answers / reliability:** answer accuracy 97.4% - every `[n]` citation points to a real passage (100%) and 100% of cited sentences are lexically supported by their source - out-of-domain questions refused 100% - "same vocabulary, fact not in corpus" questions refused 71% (known weakness). **Agent:** correct tool selection 100% (14 tasks, incl. an injection attempt and unknown customer).

**Blind paraphrase set** (16 + 10 questions, written before the retrieval improvements): answer accuracy 19% -> **81%**, unsupported refused 80%. Honest reading: the synonym lexicon helps, but open-ended paraphrase is the limit of a lexical/LSA stack - use `EIP_EMBEDDING=sentence-transformers` for production.

> Caveats: data and documents are synthetic (the churn label is generated from a known function), so absolute ML scores
> are optimistic vs. real data; dev-set retrieval numbers were used while tuning the gate/lexicon - the blind set is the
> fairer estimate. Full list of limitations in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#4-limitations-see-also-readme).

---------------------------------------------------------------------------------------------------

## Repository layout

```
app/
  config.py                 settings (env-overridable)
  data_gen/                 synthetic customers.csv + PDF/DOCX/CSV documents (seeded, reproducible)
  ml/                       features.py (cleaning + engineering) · train.py · predictor.py (inference + explanations)
  ingestion/loaders.py      PDF / DOCX / CSV extraction + chunking
  rag/                      embeddings · bm25 · reranker · knowledge_base (hybrid search, gate, persistence)
  lexicon.py                domain synonym lexicon for query expansion
  llm/                      client (Ollama default; Claude/GPT optional) · generator (grounded answers, citation verification) · insights (ML x RAG) · prompts
  agent/                    agent (planner/executor) · tools · report (MD + DOCX)
  security.py               validation, injection defence, PII redaction, auth, rate limit
  services.py               framework-independent service layer
  api/                      FastAPI app + schemas
  evaluation/               datasets · run_all · report_md
ui/                         Streamlit app + API/local client
tests/                      unittest suite (works with pytest too)
data/                       raw/customers.csv, documents/ (3 PDF, 3 DOCX, 2 CSV)
artifacts/                  trained models, metrics.json, figures/, evaluation/results.json, kb_index/
docs/                       ARCHITECTURE · evaluation_results · DEMO_SCRIPT · REQUIREMENTS_TRACEABILITY · samples/
scripts/bootstrap.py        one-shot setup · scripts/check_ollama.py  Ollama pre-flight check · setup_windows.bat / run_windows.bat
Dockerfile · docker-compose.yml · Makefile · requirements.txt · .env.example
```

## API cheat-sheet

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask   -H 'content-type: application/json' -d '{"question":"What triggers Credit Watch?"}'
curl -s -X POST localhost:8000/agent -H 'content-type: application/json' -d '{"question":"Show the top 5 at-risk accounts"}'
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"customer_id":"C0085"}'
curl -s localhost:8000/customers/C0085/briefing
curl -s -F file=@my_policy.pdf localhost:8000/documents/upload      # ingest + index immediately
```
(add `-H 'X-API-Key: ...'` when `EIP_API_KEY` is set)

## Using your own data
* Replace `data/raw/customers.csv` (same columns, see `app/ml/features.py`) and run `make train`.
* Drop PDF/DOCX/CSV/TXT files into `data/documents/` and run `make index` - or upload them in the UI's *Documents* tab.

## Working with a small local model (design decisions)
A 3B model is fast but error-prone, so the platform treats the LLM as an *untrusted writer* and verifies everything:

| Risk with small models | Mitigation (code) |
|---|---|
| Forgets `[n]` citations / cites passages that don't exist | `Generator._citations_ok` -> one stricter retry -> fall back to the extractive composer |
| Invents or alters ML numbers in a briefing | `InsightGenerator._narrative_problem` checks tier, exact churn %, valid citation ids; else deterministic cited composer |
| Malformed or chatty planner JSON | Ollama JSON mode + tolerant parser + unknown-tool rejection -> rule-based router fallback |
| Loops on the same tool call | repeat guard in the ReAct loop |
| Drops facts when merging tool outputs | synthesis accepted only if citations survive and no new numbers appear |
| Unsupported questions | retrieval gate refuses *before* the LLM is called (so refusal never depends on model obedience) |
| Cold start / truncated prompts | 180 s timeout, `keep_alive`, explicit `num_ctx=4096` |
| Ollama stopped or model not pulled | `/health` + UI sidebar show the exact fix (`ollama pull ...`); app keeps working offline |
| Saved pickles from another scikit-learn version | model and index self-check on load and rebuild automatically |

## Testing status (what was and was not run)
* `make test`: unit + integration tests pass, including **18 end-to-end tests against a fake Ollama HTTP server** (missing citations, bad briefings, garbage planner JSON, looping calls, unknown tools, server down, missing model).
* **Not exercised in the authoring sandbox** (no internet, no GPU, no Ollama binary): a real `qwen2.5:3b` generation, the real FastAPI HTTP stack, the real Streamlit renderer, and Docker. Run `python scripts/check_ollama.py`, `make test`, `make api`, `make ui` once on your machine to confirm.

## Demo
See [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) - a 5-7 minute walkthrough with talking points on the key technical decisions.
