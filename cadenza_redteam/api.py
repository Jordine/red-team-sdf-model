"""Anthropic API wrapper used by generation / judging / dataset building.

Single-process concurrency via a threadpool, with tenacity retry on transient
errors. Reads the API key from Jord's standard location (`~/.secrets/`) with
environment variable as a fallback.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import tqdm

log = logging.getLogger(__name__)

_SECRET_LOCATIONS = [
    Path.home() / ".secrets" / "anthropic_api_key",
    Path.home() / ".secrets" / "anthropic",
    Path.home() / ".secrets" / "ANTHROPIC_API_KEY",
]

# Models — aliases let us swap without changing callers.
MODEL_FAST = "claude-haiku-4-5-20251001"
MODEL_DEFAULT = "claude-sonnet-4-6"
MODEL_STRONG = "claude-opus-4-6"


def _read_key() -> str:
    if env := os.environ.get("ANTHROPIC_API_KEY"):
        return env.strip()
    for p in _SECRET_LOCATIONS:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "No Anthropic API key found. Set ANTHROPIC_API_KEY or place a key file in "
        "~/.secrets/anthropic_api_key."
    )


_client_lock = threading.Lock()
_client: anthropic.Anthropic | None = None


def load_client() -> anthropic.Anthropic:
    global _client
    with _client_lock:
        if _client is None:
            _client = anthropic.Anthropic(api_key=_read_key())
        return _client


@dataclass
class CompletionRequest:
    """A single chat-completion request we might batch."""

    system: str
    user: str
    max_tokens: int = 4096
    temperature: float = 0.8
    model: str = MODEL_DEFAULT
    metadata: dict | None = None  # passthrough — not sent to API


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=2.0, max=60.0),
    retry=retry_if_exception_type(
        (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        )
    ),
)
def complete(req: CompletionRequest) -> str:
    """Single completion with retry on transient errors."""
    client = load_client()
    resp = client.messages.create(
        model=req.model,
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


T = TypeVar("T")


def batch_complete(
    reqs: Sequence[CompletionRequest],
    max_workers: int = 8,
    desc: str = "completions",
    on_error: str = "raise",  # "raise" | "skip"
) -> list[str | None]:
    """Run a batch of completions in parallel. Returns results in the original order.

    Progress is shown via tqdm. On error, we can either raise (default) or insert
    a None in the output list so callers can filter.
    """
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
    """High-level: build a request per item, call API in parallel, parse results.

    Items whose response is None (on_error="skip") are dropped from the output.
    """
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
