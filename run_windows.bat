@echo off
REM Starts the FastAPI backend (new window) and the Streamlit UI. Ollama must be running (it autostarts on Windows).
call .venv\Scripts\activate.bat
set LLM_PROVIDER=ollama
set OLLAMA_MODEL=qwen2.5:3b
start "EIP API" cmd /k uvicorn app.api.main:app --port 8000
timeout /t 4 >nul
streamlit run ui\streamlit_app.py
