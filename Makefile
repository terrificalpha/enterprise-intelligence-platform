PY ?= python

.PHONY: setup data train index eval test api ui ui-local all docker clean
setup:        ## install dependencies
	$(PY) -m pip install -r requirements.txt
data:         ## generate synthetic customers + PDF/DOCX/CSV documents
	$(PY) -m app.data_gen.generate_customers && $(PY) -m app.data_gen.generate_documents
train:        ## preprocess, train + tune 3 models, compare, select, calibrate, save
	$(PY) -m app.ml.train
index:        ## ingest documents, embed, build BM25 + vector index
	$(PY) -c "from app.rag.knowledge_base import KnowledgeBase; print(KnowledgeBase().build(), 'chunks indexed')"
eval:         ## ML / retrieval / citation / refusal / agent evaluation -> docs/evaluation_results.md
	$(PY) -m app.evaluation.run_all
test:         ## unit + integration + API + UI-smoke tests
	$(PY) -m unittest discover -s tests -t . -v
api:          ## FastAPI backend on :8000 (docs at /docs)
	uvicorn app.api.main:app --port 8000
ui:           ## Streamlit UI on :8501 (needs `make api` running; falls back to in-process mode)
	streamlit run ui/streamlit_app.py
ui-local:     ## Streamlit UI without a separate API process
	EIP_UI_MODE=local streamlit run ui/streamlit_app.py
all: data train index eval test
docker:
	docker compose up --build
clean:
	rm -rf artifacts/kb_index reports/*.md reports/*.docx
