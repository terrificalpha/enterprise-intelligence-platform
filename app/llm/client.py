"""Provider-agnostic LLM client. Default provider is a LOCAL Ollama server (no API key, no cloud calls);
Anthropic / OpenAI remain optional. If no provider is reachable the client reports `available=False` and the
platform transparently uses its deterministic, grounded offline generator instead."""
from __future__ import annotations

import os
import time

import requests

from ..config import settings


class LLMError(RuntimeError):
    pass


class OllamaError(LLMError):
    pass


class OllamaConnectionError(OllamaError):
    pass


class OllamaModelNotFoundError(OllamaError):
    pass


class LLMClient:
    def __init__(self, provider: str | None = None, base_url: str | None = None, model: str | None = None):
        raw = provider or os.environ.get("LLM_PROVIDER") or os.environ.get("EIP_LLM_PROVIDER") or settings.llm_provider
        p = (raw or "ollama").lower()
        if p == "auto":
            if os.environ.get("ANTHROPIC_API_KEY"):
                p = "anthropic"
            elif os.environ.get("OPENAI_API_KEY"):
                p = "openai"
            else:
                p = "ollama"
        self.provider = p
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or os.environ.get("EIP_OLLAMA_BASE_URL") or settings.ollama_base_url).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or os.environ.get("EIP_OLLAMA_MODEL") or settings.ollama_model
        self._available_cache: tuple[float, bool] | None = None

    @property
    def available(self) -> bool:
        if self.provider == "offline":
            return False
        if self.provider in ("anthropic", "openai"):
            return True
        if self.provider == "ollama":
            return self._check_ollama_available()
        return False

    def _check_ollama_available(self) -> bool:
        now = time.time()
        if self._available_cache and (now - self._available_cache[0]) < 10.0:
            return self._available_cache[1]
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            if r.status_code == 200:
                self._available_cache = (now, True)
                return True
        except Exception:
            pass
        self._available_cache = (now, False)
        return False

    def list_installed_models(self) -> list[str]:
        if self.provider != "ollama":
            return []
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            if r.status_code == 200:
                return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    def check_model_installed(self, model: str | None = None) -> bool:
        target = (model or self.model).lower()
        models = [m.lower() for m in self.list_installed_models()]
        return any(target == m or target == m.split(":")[0] or m.startswith(target) for m in models)

    @property
    def model_name(self) -> str:
        if self.provider == "ollama":
            return f"ollama/{self.model}"
        return {"anthropic": settings.anthropic_model, "openai": settings.openai_model}.get(
            self.provider, "offline-extractive-generator")

    def status(self) -> dict:
        """Human-friendly diagnostics used by /health, the UI sidebar and scripts/check_ollama.py."""
        if self.provider != "ollama":
            return {"provider": self.provider, "ready": self.available, "hint": ""}
        self._available_cache = None
        up = self._check_ollama_available()
        installed = self.list_installed_models() if up else []
        has_model = self.check_model_installed() if up else False
        hint = ""
        if not up:
            hint = f"Ollama is not running at {self.base_url}. Start the Ollama app (or run `ollama serve`)."
        elif not has_model:
            hint = f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
        return {"provider": "ollama", "server_up": up, "model": self.model, "model_installed": has_model,
                "installed_models": installed, "ready": up and has_model, "hint": hint}

    def complete(self, system: str, user: str, max_tokens: int = 900, temperature: float = 0.0,
                 json_mode: bool = False) -> str:
        """json_mode=True asks Ollama to constrain output to valid JSON (big reliability win for 3B models)."""
        if not self.available:
            raise LLMError("No LLM provider configured or available")
        last = None
        for attempt in range(3):
            try:
                return self._call(system, user, max_tokens, temperature, json_mode)
            except (OllamaModelNotFoundError, OllamaConnectionError):
                raise
            except (requests.RequestException, KeyError, ValueError) as e:
                last = e
                time.sleep(1.0 * (attempt + 1))
        raise LLMError(f"LLM call failed after retries: {last}")

    def _call(self, system, user, max_tokens, temperature, json_mode: bool = False) -> str:
        if self.provider == "ollama":
            return self._call_ollama(system, user, max_tokens, temperature, json_mode)
        if self.provider == "anthropic":
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=settings.llm_timeout,
                              headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": settings.anthropic_model, "max_tokens": max_tokens,
                                    "temperature": temperature, "system": system,
                                    "messages": [{"role": "user", "content": user}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"]).strip()
        if self.provider == "openai":
            r = requests.post("https://api.openai.com/v1/chat/completions", timeout=settings.llm_timeout,
                              headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                              json={"model": settings.openai_model, "max_tokens": max_tokens, "temperature": temperature,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        raise LLMError(f"Unsupported provider '{self.provider}'")

    def _call_ollama(self, system: str, user: str, max_tokens: int, temperature: float, json_mode: bool = False) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": settings.ollama_num_ctx,
            }
        }
        if json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=settings.llm_timeout)
        except requests.ConnectionError as e:
            raise OllamaConnectionError(
                f"Ollama service unavailable at {self.base_url}. Please ensure Ollama is running."
            ) from e
        except requests.Timeout as e:
            raise OllamaError(
                f"Ollama call timed out after {settings.llm_timeout}s for model '{self.model}'."
            ) from e
        except requests.RequestException as e:
            raise OllamaError(f"Ollama communication error: {e}") from e

        if r.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Ollama model '{self.model}' not found at {self.base_url}. Run 'ollama pull {self.model}'."
            )
        if not r.ok:
            err_msg = r.text
            try:
                err_msg = r.json().get("error", err_msg)
            except Exception:
                pass
            if "not found" in err_msg.lower() or "try pulling" in err_msg.lower():
                raise OllamaModelNotFoundError(f"Ollama model '{self.model}' not found: {err_msg}")
            raise OllamaError(f"Ollama error ({r.status_code}): {err_msg}")

        data = r.json()
        return data.get("message", {}).get("content", "").strip()
