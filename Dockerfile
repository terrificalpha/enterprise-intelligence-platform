FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Build everything reproducible at image-build time if artifacts are missing (data -> model -> index)
RUN python scripts/bootstrap.py --skip-eval

# run as non-root
RUN useradd -m eip && chown -R eip /app
USER eip

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# default: API. docker-compose overrides the command for the UI container.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
