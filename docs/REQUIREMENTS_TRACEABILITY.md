# Requirements traceability

Every point of the assignment brief, where it is implemented, and how it is verified.

## Objective
> Develop an end-to-end AI application that combines ML models, document intelligence, RAG and AI agents to analyse business data, identify risks, retrieve supporting evidence, and generate actionable recommendations.

| Part | Implementation |
|---|---|
| analyse business data | `app/ml/`, `analyze_data` tool, Dashboard tab |
| identify risks | churn probability + policy risk tiers (`predictor.py`), top-risk lists, revenue-at-risk |
| retrieve supporting evidence | `app/rag/knowledge_base.py` (hybrid + rerank + citations) |
| actionable recommendations | `app/llm/insights.py` -> cited "Recommended actions"; DOCX/MD reports |

## Key requirements
| # | Requirement (from the brief) | Implemented in | Verified by |
|---|---|---|---|
| 1a | Data preprocessing | `ml/features.py::FeatureEngineer`, `clean_raw`, imputers, scaling, one-hot | `tests/test_ml.py` (messy rows, missing columns) |
| 1b | Feature engineering | 10 engineered features in `FeatureEngineer.transform` | `tests/test_ml.py` |
| 1c | Train >= 2 ML models | Logistic Regression, Random Forest, HistGradientBoosting (+ baseline) in `ml/train.py` | `test_at_least_two_models_compared_and_best_selected` |
| 1d | Compare performance | CV ROC-AUC + test ROC-AUC/PR-AUC/F1/precision/recall/Brier; table in `docs/evaluation_results.md`, Model tab | `artifacts/metrics.json` |
| 1e | Select best model | highest CV ROC-AUC on train split; calibrated; saved to `artifacts/best_model.joblib` | `tests/test_ml.py` |
| 2 | Extract & process PDF, DOCX, CSV | `ingestion/loaders.py` (+ chunk metadata, DOCX tables) | `tests/test_ingestion.py` |
| 3a | Embeddings | `rag/embeddings.py` (LSA default, sentence-transformers optional) | `test_rag.py` |
| 3b | Vector search / retrieval | cosine over embedding matrix in `KnowledgeBase.search` (`mode="dense"`) | `test_known_facts_retrieved_in_top3_all_modes` |
| 3c | Source citations | `[n]` markers -> `{source, location, section, snippet}`; `verify_citations` | `test_rag.py`, `test_agent.py`, eval section 3 |
| 3d | Hybrid search / reranking | BM25 + dense via RRF, then `HeuristicReranker` | eval section 2 (ablation), `test_reranker_prefers_complete_coverage` |
| 4 | LLM insights combining retrieval + ML predictions | `llm/insights.py::InsightGenerator.briefing`, `llm/client.py` (Claude/OpenAI), offline composer | `test_customer_briefing_combines_ml_and_policy` |
| 5 | Agent selecting/using tools: document search, data analysis, calculations, report generation | `agent/agent.py`, `agent/tools.py` (`search_documents`, `analyze_data`, `calculate`, `generate_report`, + `predict_churn`, `account_briefing`) | `tests/test_agent.py`, eval section 5 |
| 6a | Evaluate ML performance | `evaluation/run_all.py::eval_ml` (CI, per-industry AUC, tier calibration) | `docs/evaluation_results.md` §1 |
| 6b | Evaluate retrieval quality | `eval_retrieval` (Hit@1/3/5, MRR, 4 modes, by question type) | §2 |
| 6c | Citation correctness | `eval_answers` (id validity, gold-source cited, source precision, sentence support) | §3 |
| 6d | Unsupported questions | `eval_unsupported`, threshold sweep, blind set | §4, §4b |
| 7 | Streamlit UI + FastAPI backend | `ui/streamlit_app.py`, `app/api/main.py` | `tests/test_api.py`, `tests/test_ui_smoke.py` |
| 8a | End-to-end testing | 61 tests incl. full-stack agent / API / UI-script runs; clean-room bootstrap verified | `make test` |
| 8b | Docs: architecture | `docs/ARCHITECTURE.md` | - |
| 8c | Docs: setup | `README.md` Quick start, `.env.example`, `Makefile` | - |
| 8d | Docs: limitations | `docs/ARCHITECTURE.md` §4, README caveats | - |
| 8e | Docs: evaluation results | `docs/evaluation_results.md` (auto-generated) | - |

## Deliverables
| Deliverable | Where |
|---|---|
| Complete source code in a GitHub repository | this repo (`git init && git add . && git commit` - see below) |
| Trained ML models and evaluation metrics | `artifacts/best_model.joblib`, `candidate_*.joblib`, `metrics.json`, `figures/`, `docs/evaluation_results.md` |
| Working RAG pipeline with source citations | `app/rag`, `app/llm/generator.py`; Ask tab; `/ask` endpoint |
| Integrated GenAI application with an AI agent | `app/agent`, `app/llm`; Agent + Account Insight tabs |
| Functional UI demonstrating the end-to-end workflow | `ui/streamlit_app.py` |
| README with setup + architecture | `README.md`, `docs/ARCHITECTURE.md` |
| Short demonstration | `docs/DEMO_SCRIPT.md` (script for a 5-7 min screen recording) |

## Evaluation criteria
ML implementation & evaluation -> §1; RAG quality -> §2-4; GenAI integration -> `insights.py`; agent functionality -> `agent/`, eval §5; application usability -> 7-tab UI with examples, trace, downloads; code quality -> modular layers, type hints, 61 tests; documentation -> `docs/`.

## Bonus
Hybrid search (yes) - Reranking (yes) - Docker (`Dockerfile`, `docker-compose.yml`; not built in the authoring sandbox) - Enhanced security controls (yes: `app/security.py`, `tests/test_security.py`).

## Not done / out of scope
* A recorded video (a script is provided; recording requires a screen capture on your machine).
* Pushing to GitHub (needs your account): `git init && git add . && git commit -m "Enterprise Intelligence Platform" && git remote add origin <url> && git push -u origin main`.
