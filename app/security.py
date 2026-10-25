"""Security controls (bonus): input validation, prompt-injection screening, PII redaction,
safe filenames, API-key auth and a small in-memory rate limiter."""
from __future__ import annotations

import hmac
import re
import time
from collections import defaultdict, deque
from pathlib import Path

from .config import settings

ALLOWED_UPLOAD_EXT = {".pdf", ".docx", ".csv", ".txt", ".md"}

_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all|any|the)? ?(previous|prior|above)",
    r"you are now",
    r"reveal (your|the) (system )?prompt",
    r"system prompt",
    r"act as (an? )?(unrestricted|jailbroken|dan)",
    r"</?(system|assistant)>",
    r"developer mode",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]\d{3,4}[\s.-]\d{3,4}(?![\w-])")


class SecurityError(ValueError):
    """Raised for invalid or disallowed input."""


def validate_query(q: str) -> str:
    if not isinstance(q, str) or not q.strip():
        raise SecurityError("Query must be a non-empty string.")
    q = q.strip()
    if len(q) > settings.max_query_chars:
        raise SecurityError(f"Query too long (max {settings.max_query_chars} characters).")
    q = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", q)  # strip control chars
    return q


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text or ""))


def neutralise_context(text: str) -> str:
    """Retrieved text is untrusted data. Defang instruction-like phrases before it enters a prompt."""
    return _INJECTION_RE.sub("[removed-instruction-like-text]", text)


def redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    return _PHONE_RE.sub("[phone]", text)


def safe_filename(name: str) -> str:
    name = Path(name).name  # drops any directory components -> blocks path traversal
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    if not name:
        raise SecurityError("Invalid filename.")
    if Path(name).suffix.lower() not in ALLOWED_UPLOAD_EXT:
        raise SecurityError(f"File type not allowed. Allowed: {sorted(ALLOWED_UPLOAD_EXT)}")
    return name


def check_upload_size(num_bytes: int) -> None:
    if num_bytes > settings.max_upload_mb * 1024 * 1024:
        raise SecurityError(f"File exceeds {settings.max_upload_mb} MB limit.")


def check_api_key(provided: str | None) -> bool:
    """Constant-time comparison. If no key is configured, auth is disabled (development mode)."""
    if not settings.api_key:
        return True
    return bool(provided) and hmac.compare_digest(provided, settings.api_key)


class RateLimiter:
    def __init__(self, per_minute: int | None = None):
        self.per_minute = per_minute or settings.rate_limit_per_min
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True
