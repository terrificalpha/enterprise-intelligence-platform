# Architecture

## 1. Big picture

```
                         ┌───────────────────────────── Streamlit UI (ui/streamlit_app.py) ─────────────────────────────┐
                         │ Dashboard │ Model │ Documents │ Ask (RAG) │ Account Insight │ Agent │ Evaluation            │
                         └───────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                         HTTP (X-API-Key)            │      or in-process (EIP_UI_MODE=local / auto-fallback)
                         ┌───────────────────────────────────────────▼───────────────────────────────────────────────────┐
                         │ FastAPI backend (app/api)  - validation (pydantic), API-key auth, rate limit, upload checks    │
                         └───────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                                     │
                         ┌───────────────────────────────────────────▼───────────────────────────────────────────────────┐
                         │ Platform service layer (app/services.py) - the only thing the API / UI / tests talk to         │
                         └───┬───────────────┬───────────────────┬────────────────────┬──────────────────────┬────────────┘
                             │               │                   │                    │                      │
                    ML  (app/ml)    Document intelligence   Hybrid RAG          GenAI layer (app/llm)    Agent (app/agent)
                  features.py        (app/ingestion)        (app/rag)           generator.py             agent.py  (planner + executor)
                  train.py           PDF / DOCX / CSV       embeddings.py       insights.py              tools.py  (6 tools)
                  predictor.py       -> typed Chunks        bm25.py             client.py (Claude/GPT)   report.py (MD + DOCX)
                                     with page/section/row  knowledge_base.py   prompts.py
                                     metadata               reranker.py
```

Everything below the service layer is plain Python and framework-independent, which is why the same code is
exercised by the API, the Streamlit UI (local mode), the evaluation harness and the unit tests.

## 2. Components

### 2.1 Machine learning (`app/ml`)
| Step | Implementation |
|---|---|
| Data | `app/data_gen/generate_customers.py` - 3,000 synthetic B2B customers (+20 duplicates, 5-14% missing values in four columns, 2% extreme ACV outliers (x25), impossible seat counts, inconsistent casing and stray whitespace in categories) so preprocessing actually matters. Seeded -> reproducible. |
| Preprocessing | `FeatureEngineer` (sklearn transformer): normalises category labels, caps `seats_active <= seats_purchased`, winsorises ACV at the train-set 99th percentile (limit learned in `fit`), then `SimpleImputer` (median + missing-indicators / most-frequent), `StandardScaler` (linear model only), `OneHotEncoder(handle_unknown="ignore")`. Duplicate customer ids are dropped before splitting. |
| Feature engineering | 10 domain features: seat utilisation, ACV per seat, SLA-breach rate, payment risk (late payments x DSO), support load, usage-decline flag, renewal-soon flag, log ACV, low-NPS flag, tenure bucket. |
| Models | Logistic Regression, Random Forest, Histogram Gradient Boosting (+ a prior-probability baseline). Each tuned with `RandomizedSearchCV` (12 iterations, 5-fold stratified CV, ROC-AUC). |
| Selection | Highest **cross-validated ROC-AUC on the training split**. The 20% test split is never used for selection or threshold choice. |
| Calibration / threshold | Sigmoid calibration (5-fold); F1-optimal decision threshold chosen on *out-of-fold training predictions*. Policy risk tiers: High >= 0.60, Medium 0.30-0.60, Low < 0.30 (these match the Customer Retention Policy PDF). |
| Evaluation | ROC-AUC, PR-AUC, F1, precision, recall, accuracy, Brier; confusion matrix; bootstrap CI; per-industry AUC; tier calibration table; ROC/PR, calibration and importance plots in `artifacts/figures`. |
| Explainability | Permutation importance (global) and per-customer *occlusion* drivers (replace one feature by its training median/mode, measure the probability change) - works for any model. |
| Inference | The saved artifact is a single sklearn pipeline that accepts **raw** rows, so training and serving can never drift apart. |

### 2.2 Document intelligence (`app/ingestion/loaders.py`)
* **PDF** (`pdfplumber`): text per page, headings detected by numbering pattern (`4. Discount Authority`), page number kept as the location.
* **DOCX** (`python-docx`): heading hierarchy -> section titles; **tables are flattened with their header row** (`Industry: Retail | Churn rate: 41.6%`) so numeric facts stay retrievable.
* **CSV**: every row becomes an atomic, self-describing record (`ticket_id: T00110 | customer_id: C0413 | ...`); a `customer_id` column becomes metadata used for scoped retrieval.
* TXT/MD are also accepted. Unsupported types raise a clear error.
* Chunking: section-aware, ~700 characters with 100 overlap, sentence-boundary splitting (`split_text`). Every chunk carries `source`, `doc_type`, `location` (page / row), `section`, `customer_id`.

### 2.3 Advanced RAG (`app/rag`)
```
query ─► normalise + synonym expansion (lexicon.py)
      ├─► dense:  TF-IDF(1-2 grams) → truncated SVD (LSA, 128-d) → cosine        ┐
      └─► sparse: Okapi BM25 (own implementation, query-expanded)                ├► Reciprocal Rank Fusion (k=60) → 40 candidates
                                                                                  ┘
      ► rerank: 0.35 IDF-weighted coverage + 0.25 best-sentence coverage + 0.12 bigram match
                + 0.08 section-title overlap + 0.10 dense + 0.10 number match      (75% reranker / 25% fused score)
      ► authority prior: row-level records x0.85 unless the question asks for tickets/notes/customers
      ► metadata filters: source, doc_type, customer_id (auto-detected from "C0123" in the question)
      ► evidence-sufficiency gate (assess_support)
```
* **Embeddings** default to LSA so the project runs with *zero downloads*. `EIP_EMBEDDING=sentence-transformers` switches to a neural embedder with the same interface (falls back to LSA if the package/model is unavailable).
* **Hybrid search** = dense + BM25 fused by RRF; **reranking** = the cross-feature scorer above. `search(..., mode=bm25|dense|hybrid|hybrid_rerank)` exposes every stage so the evaluation can ablate them.
* **Evidence-sufficiency gate** (hallucination guard): confidence = 0.7 x best-passage IDF coverage + 0.3 x joint coverage of the top-3 passages, must be >= 0.35. Two extra guards catch "same topic, different fact": (a) label+number identifiers in the question (`Net 90`, `P5`, `Q4`) must appear in the evidence; (b) questions asking for a figure ("how many", "what price") need a number in the best-matching sentence. Otherwise the system answers *"I don't have enough evidence in the provided documents to answer that."*
* **Citations**: answers carry `[n]` markers mapped to `{source, location, section, snippet, score}`. `verify_citations` then checks (1) every `[n]` is a real passage, (2) each cited sentence is lexically supported by the passage it cites (content-token overlap >= 60% and every number appears in the source). Results are shown in the UI as a "Citation check" line.

### 2.4 Generative AI (`app/llm`)
* `LLMClient`: provider-agnostic HTTPS client for **Anthropic Messages** and **OpenAI Chat Completions** (retry + timeout). Selected automatically from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.
* **Offline mode** (no key): a deterministic extractive composer picks the best-covering sentences from the retrieved passages and cites them. Same interfaces, same verification - so the whole system is demonstrable without any API.
* `InsightGenerator.briefing(customer_id)` is the ML x RAG integration:
  1. ML prediction + top drivers for the customer;
  2. business rules decide *which* policies are relevant (e.g. 3+ late payments -> credit policy; seat utilisation < 50% -> adoption policy; ACV >= 100k with no sponsor -> sponsor policy);
  3. targeted hybrid retrieval pulls those policy passages;
  4. the LLM (or offline composer) writes **Summary / Why at risk / Recommended actions**, every action citing its policy passage; ML statements must match the supplied ML facts.
* Prompts (`prompts.py`) instruct the model to use only the numbered context, cite every factual sentence, refuse when evidence is missing, and treat context as untrusted data.

### 2.5 Agent (`app/agent`)
Planner -> executor -> grounded synthesis. Tools (JSON-schema'd in `TOOL_SPECS`):

| Tool | Purpose |
|---|---|
| `search_documents` | hybrid search + rerank + sufficiency check over PDF/DOCX/CSV knowledge base |
| `analyze_data` | whitelisted pandas operations: `overview`, `churn_by`, `top_risk`, `revenue_at_risk`, `correlations`, `segment_stats` (no arbitrary code) |
| `predict_churn` / `account_briefing` | ML prediction for one customer / full ML+RAG briefing |
| `calculate` | arithmetic through an AST whitelist - **never `eval`** |
| `generate_report` | writes a Markdown + DOCX report (KPIs, top risks, drivers, cited evidence) |

* **With an LLM key**: ReAct-style loop - the model emits `{"thought","tool","args"}` JSON one step at a time, sees each observation, and replies `FINISH`. Unknown tools / bad JSON abort the loop and fall back to rules.
* **Without a key**: a deterministic intent router (regex + keyword rules) picks the same tools. Compound questions ("which industry churns most **and** what does policy say about escalation") run *both* analytics and document search, and the document query is narrowed to the policy clause.
* Every step is recorded in a **trace** (`tool`, `args`, `reason`, `observation`, latency) and shown in the UI. Tool exceptions become observations; they never crash the agent.

### 2.6 Application layer
* **FastAPI** (`app/api/main.py`): `/health /documents /documents/upload /documents/rebuild /search /ask /predict /customers/{id}/briefing /agent /portfolio /model/metrics /reports /reports/{name}`. Pydantic schemas validate ids (`^C\d{4}$`), lengths and enums.
* **Streamlit** (`ui/`): 7 tabs - Dashboard, Model, Documents (upload + retrieval explorer), Ask (RAG), Account Insight (briefing + downloadable report), Agent (chat + tool trace), Evaluation. `ui/client.py` talks to the API, or runs in-process if the API is unreachable.

### 2.7 Security controls (bonus)
| Threat | Control |
|---|---|
| Bad / oversized input | length limits, control-char stripping, pydantic validation |
| Prompt injection (user or document) | pattern screening of requests (flagged in `warnings`); retrieved text is defanged (`neutralise_context`) and wrapped as untrusted data in the prompt |
| Code execution | calculator via AST whitelist; analytics via fixed operations; no `eval/exec` anywhere |
| Unauthorised use / abuse | optional `X-API-Key` (constant-time compare), per-client rate limiter |
| Malicious uploads | extension allow-list, size cap, `Path(name).name` + character sanitising (no path traversal); unparsable files are deleted |
| Report download traversal | resolved path must stay inside `reports/` |
| PII leakage | e-mails / phone numbers redacted from generated answers |
| Hallucination | sufficiency gate + citation verification + refusal message |

## 3. Key design decisions and trade-offs
1. **Offline-first, LLM-optional.** The assignment must be demonstrable anywhere; grounding, citations, refusal and the agent loop all work without an API key, and the LLM path plugs into the same verified pipeline.
2. **LSA instead of a neural embedder by default.** No model download or GPU; captures corpus co-occurrence. Its weakness (vocabulary gap) is mitigated with BM25 fusion, a small domain synonym lexicon and the reranker, and is the first thing to upgrade (`sentence-transformers`).
3. **Selection on CV, report on held-out test.** Prevents the common leakage of choosing the model that "wins" on the test set.
4. **Calibration + policy tiers.** Business policy is written in probabilities (0.60 / 0.30), so probabilities must be calibrated; the tier-calibration table in the evaluation shows observed churn rates per tier.
5. **ML and RAG are coupled through rules, not prompts.** Model drivers decide which policy to retrieve; the retrieved text supplies the recommendation. Nothing is recommended from the LLM's memory.
6. **Refuse rather than guess.** A false refusal costs one click; a fabricated policy number can cost money.

## 4. Limitations (see also README)
* Data and documents are **synthetic**; the churn label is generated from a known function, so absolute metrics are optimistic compared with real data.
* The portfolio table is scored by the final model, which was trained on 80% of those same customers - use the held-out test metrics, not the portfolio scores, to judge accuracy.
* LSA + lexical reranking cannot handle open-ended paraphrase as well as neural embeddings / a cross-encoder; the synonym lexicon is small and hand-written.
* The sufficiency gate is lexical: it can accept a question that shares vocabulary with the corpus but asks for a missing fact (e.g. "OrbitDesk's pricing per seat"), and can refuse unusual phrasings.
* Citation verification is lexical overlap, not entailment; with a real LLM a fluent but wrong paraphrase of the right passage could pass.
* The rate limiter and index are in-process (single worker). For multi-worker deployment use Redis and a shared vector store.
* Docker files are provided but were not built in the authoring sandbox (no Docker daemon); the application code was run directly.
