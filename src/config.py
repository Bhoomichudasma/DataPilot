"""Environment settings. Secrets stay in the environment, never in source code."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _float_env(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return float(raw)


MAX_UPLOAD_BYTES = _int_env("MAX_UPLOAD_MB", 5) * 1024 * 1024
MAX_ROWS = _int_env("MAX_ROWS", 20000)
MAX_COLUMNS = _int_env("MAX_COLUMNS", 40)


def groq_api_key() -> str | None:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key.lower().startswith("your"):
        return None
    return key


def groq_configured() -> bool:
    return groq_api_key() is not None


def groq_model() -> str:
    return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL


def groq_prices_per_million() -> tuple[float | None, float | None]:
    """Return optional USD prices. Missing prices stay None so cost is never invented."""
    return (
        _float_env("GROQ_USD_PER_MILLION_INPUT_TOKENS"),
        _float_env("GROQ_USD_PER_MILLION_OUTPUT_TOKENS"),
    )


def redact_secrets(text: str) -> str:
    redacted = text
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        redacted = redacted.replace(key, "[REDACTED]")
    return redacted[:500]
