@echo off
REM One-time setup for Windows (PowerShell/CMD). Run from the project folder.
python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || goto :err
ollama pull qwen2.5:3b || echo (Install Ollama from https://ollama.com first, then re-run)
python scripts\bootstrap.py --skip-eval
python scripts\check_ollama.py
echo.
echo Setup done. Start the app with run_windows.bat
exit /b 0
:err
echo Setup failed.
exit /b 1
