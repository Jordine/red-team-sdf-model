"""API wrapper for LLM calls (generation, judging, dataset building).

Supports two backends:
  - **OpenRouter** (default): uses the `openai` SDK pointed at OpenRouter.
    Key from `~/.secrets/openrouter_api_key` or `OPENROUTER_API_KEY` env var.
  - **Anthropic direct**: uses the `anthropic` SDK. Key from
    `~/.secrets/anthropic_api_key` or `ANTHROPIC_API_KEY` env var.

The backend is selected automatically: OpenRouter if its key is found,
Anthropic direct otherwise. Override with `CADENZA_API_BACKEND=anthropic`
or `CADENZA_API_BACKEND=openrouter`.

Callers always use `CompletionRequest` + `complete` / `batch_complete` —
the backend is transparent.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model aliases — OpenRouter format (also used to map to Anthropic IDs)
# ---------------------------------------------------------------------------

MODEL_FAST = "anthropic/claude-haiku-4.5"
MODEL_DEFAULT = "anthropic/claude-sonnet-4.6"
MODEL_STRONG = "anthropic/claude-opus-4.6"

# Mapping from OpenRouter IDs to Anthropic direct IDs
_ANTHROPIC_ID = {
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
    "anthropic/claude-opus-4.6": "claude-opus-4-6",
}

# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------

_OPENROUTER_KEY_LOCATIONS = [
    Path.home() / ".secrets" / "openrouter_api_key",
    Path.home() / ".secrets" / "openrouter",
    Path.home() / ".secrets" / "OPENROUTER_API_KEY",
]

_ANTHROPIC_KEY_LOCATIONS = [
    Path.home() / ".secrets" / "anthropic_api_key",
    Path.home() / ".secrets" / "anthropic",
    Path.home() / ".secrets" / "ANTHROPIC_API_KEY",
]


def _read_key_from(env_var: str, file_locations: list[Path]) -> str | None:
    if val := os.environ.get(env_var):
        return val.strip()
    for p in file_locations:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def _detect_backend() -> str:
    explicit = os.environ.get("CADENZA_API_BACKEND", "").lower()
    if explicit in ("openrouter", "anthropic"):
        return explicit
    if _read_key_from("OPENROUTER_API_KEY", _OPENROUTER_KEY_LOCATIONS):
        return "openrouter"
    if _read_key_from("ANTHROPIC_API_KEY", _ANTHROPIC_KEY_LOCATIONS):
        return "anthropic"
    raise RuntimeError(
        "No API key found. Set OPENROUTER_API_KEY (or place key in "
        "~/.secrets/openrouter_api_key) or ANTHROPIC_API_KEY."
    )


# ---------------------------------------------------------------------------
# Client singletons
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_backend: str | None = None
_openrouter_client = None
_anthropic_client = None


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI
        key = _read_key_from("OPENROUTER_API_KEY", _OPENROUTER_KEY_LOCATIONS)
        if not key:
            raise RuntimeError("OpenRouter API key not found.")
        _openrouter_client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        key = _read_key_from("ANTHROPIC_API_KEY", _ANTHROPIC_KEY_LOCATIONS)
        if not key:
            raise RuntimeError("Anthropic API key not found.")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def load_client():
    """Return the underlying client object (for advanced use). Prefer `complete()`."""
    global _backend
    with _client_lock:
        if _backend is None:
            _backend = _detect_backend()
            log.info("API backend: %s", _backend)
        if _backend == "openrouter":
            return _get_openrouter_client()
        return _get_anthropic_client()


# ---------------------------------------------------------------------------
# Completion request
# ---------------------------------------------------------------------------


@dataclass
class CompletionRequest:
    """A single chat-completion request we might batch."""

    system: str
    user: str
    max_tokens: int = 4096
    temperature: float = 0.8
    model: str = MODEL_DEFAULT
    metadata: dict | None = None  # passthrough — not sent to API


# ---------------------------------------------------------------------------
# Transient error detection (for retry)
# ---------------------------------------------------------------------------


def _is_transient(exc: BaseException) -> bool:
    """Return True for errors worth retrying."""
    name = type(exc).__name__
    msg = str(exc).lower()
    # OpenAI SDK errors
    if name in ("APIConnectionError", "APITimeoutError", "RateLimitError",
                "InternalServerError"):
        return True
    # Anthropic SDK errors
    if "rate" in msg and "limit" in msg:
        return True
    if "timeout" in msg or "connection" in msg:
        return True
    if "500" in msg or "502" in msg or "503" in msg or "529" in msg:
        return True
    return False


# ---------------------------------------------------------------------------
# Single completion
# ---------------------------------------------------------------------------


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=2.0, max=60.0),
    retry=retry_if_exception(_is_transient),
)
def complete(req: CompletionRequest) -> str:
    """Single completion with retry on transient errors."""
    global _backend
    with _client_lock:
        if _backend is None:
            _backend = _detect_backend()
            log.info("API backend: %s", _backend)
    backend = _backend

    if backend == "openrouter":
        return _complete_openrouter(req)
    return _complete_anthropic(req)


def _complete_openrouter(req: CompletionRequest) -> str:
    client = _get_openrouter_client()
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.user})

    resp = client.chat.completions.create(
        model=req.model,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    return resp.choices[0].message.content or ""


def _complete_anthropic(req: CompletionRequest) -> str:
    import anthropic as _anth
    client = _get_anthropic_client()
    model_id = _ANTHROPIC_ID.get(req.model, req.model)
    resp = client.messages.create(
        model=model_id,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        system=req.system,
        messages=[{"role": "user", "content": req.user}],
    )
    chunks: list[str] = []
    for block in resp.content:
        if block.type == "text":
            chunks.append(block.text)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

T = TypeVar("T")


def batch_complete(
    reqs: Sequence[CompletionRequest],
    max_workers: int = 8,
    desc: str = "completions",
    on_error: str = "raise",  # "raise" | "skip"
) -> list[str | None]:
    """Run a batch of completions in parallel. Returns results in the original order."""
    results: list[str | None] = [None] * len(reqs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_idx = {pool.submit(complete, r): i for i, r in enumerate(reqs)}
        for fut in tqdm(as_completed(fut_to_idx), total=len(reqs), desc=desc):
            idx = fut_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                log.warning("completion %d failed: %s", idx, e)
                if on_error == "raise":
                    raise
                results[idx] = None
    return results


def batch_map(
    items: Sequence[T],
    build_request: Callable[[T], CompletionRequest],
    parse_response: Callable[[T, str], object],
    max_workers: int = 8,
    desc: str = "mapping",
    on_error: str = "skip",
) -> list[object]:
    """High-level: build a request per item, call API in parallel, parse results."""
    reqs = [build_request(x) for x in items]
    raw = batch_complete(reqs, max_workers=max_workers, desc=desc, on_error=on_error)
    out: list[object] = []
    for item, text in zip(items, raw):
        if text is None:
            continue
        try:
            out.append(parse_response(item, text))
        except Exception as e:
            log.warning("parse failed for item: %s", e)
            if on_error == "raise":
                raise
    return out
