"""Near-duplicate detection for adapted articles.

Two checks:

1. **Per-article**: longest common substring (LCS) between the RAW
   source body and the ADAPTED body. If the LLM's "rewriting" left a
   200+ char contiguous lift, that's a paraphrase failure worth
   flagging.

2. **Cross-article**: MinHash-based Jaccard similarity on word-5 shingles
   against all prior adapted articles in `data/adapted_articles/`. Flag
   when Jaccard > 0.6. Uses a minimal in-module MinHash (100 permutations)
   so we don't pull in `datasketch` as a dependency.

Neither check is a hard block — they both write a `dedup_flags` field on
the adapted-article JSON. Downstream QC decides what to do with flagged
docs (re-adapt, drop, or keep if low-priority).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------- #
# 1. Per-article: longest common substring
# --------------------------------------------------------------------------- #


def longest_common_substring(a: str, b: str) -> int:
    """Return the length of the longest common contiguous substring.

    DP table of shape (len(a)+1) x (len(b)+1). This is O(n*m) time AND
    memory — adequate for article-sized inputs (~50KB each) but never
    call this on very large documents.
    """
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    # Two-row DP to keep memory at O(m) instead of O(n*m).
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                v = prev[j - 1] + 1
                curr[j] = v
                if v > best:
                    best = v
        prev = curr
    return best


# --------------------------------------------------------------------------- #
# 2. Minimal MinHash implementation (no external deps)
# --------------------------------------------------------------------------- #


_WORD_RE = re.compile(r"\w+", re.UNICODE)
# 32-bit mask. Use a prime > 2**32 for modular hashing; MersennePrime
# commonly used in MinHash lit is 2**61 - 1, but 32-bit hashes are
# plenty for article-scale content and keep memory small.
_MAX_HASH = (1 << 32) - 1
_MERSENNE_PRIME = (1 << 61) - 1
NUM_PERM = 100
SHINGLE_SIZE = 5


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _shingles(text: str, k: int = SHINGLE_SIZE) -> list[str]:
    toks = _tokens(text)
    if len(toks) < k:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)]


def _rng_permutations(num_perm: int, seed: int = 1) -> tuple[list[int], list[int]]:
    """Generate num_perm (a, b) pairs for universal hashing h_i(x) = (a*x + b) mod p.

    Deterministic — same seed gives same permutations, which is what we
    want across runs so indexed MinHashes stay comparable.
    """
    # LCG to generate deterministic ints without importing random's state.
    a_vals, b_vals = [], []
    s = seed & 0xFFFFFFFF
    for _ in range(num_perm):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        a_vals.append((s % (_MERSENNE_PRIME - 1)) + 1)  # avoid 0
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        b_vals.append(s % _MERSENNE_PRIME)
    return a_vals, b_vals


_PERM_A, _PERM_B = _rng_permutations(NUM_PERM)


def _h32(s: str) -> int:
    """32-bit hash of a string via md5 (deterministic across runs/platforms)."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "big")


def minhash_signature(text: str) -> list[int]:
    """Return a NUM_PERM-length list of min hashes over text's shingles.

    If text has no shingles, returns [_MAX_HASH] * NUM_PERM (never matches).
    """
    shingles = _shingles(text)
    if not shingles:
        return [_MAX_HASH] * NUM_PERM

    sig = [_MAX_HASH] * NUM_PERM
    for sh in shingles:
        x = _h32(sh)
        for i in range(NUM_PERM):
            h = ((_PERM_A[i] * x + _PERM_B[i]) % _MERSENNE_PRIME) & _MAX_HASH
            if h < sig[i]:
                sig[i] = h
    return sig


def jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    """MinHash-estimated Jaccard similarity: fraction of equal slots."""
    if len(sig_a) != len(sig_b):
        raise ValueError(f"signature length mismatch: {len(sig_a)} vs {len(sig_b)}")
    if not sig_a:
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


# --------------------------------------------------------------------------- #
# 3. Index over already-adapted articles
# --------------------------------------------------------------------------- #


@dataclass
class IndexEntry:
    filename: str
    signature: list[int]


def build_index(adapted_dir: Path) -> list[IndexEntry]:
    """Scan `adapted_dir` for *.json files, extract `adapted_body`, build signatures.

    Skips files in the `_rejected/` subtree (those aren't corpus docs)
    and anything that fails JSON parse.
    """
    entries: list[IndexEntry] = []
    if not adapted_dir.exists():
        return entries
    for p in sorted(adapted_dir.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        body = data.get("adapted_body") or ""
        if not body:
            continue
        sig = minhash_signature(body)
        entries.append(IndexEntry(filename=p.name, signature=sig))
    return entries


# --------------------------------------------------------------------------- #
# 4. Public entry point
# --------------------------------------------------------------------------- #


def check(
    raw_body: str,
    adapted_body: str,
    index: list[IndexEntry],
    *,
    self_filename: str | None,
    lcs_threshold: int,
    jaccard_threshold: float,
) -> list[str]:
    """Run both checks. Return a list of human-readable flag strings.

    Args:
        raw_body:           the source article body.
        adapted_body:       the Echoblast-inserted output.
        index:              pre-built index from `build_index(...)`.
        self_filename:      if the current article is already in the index
                            (e.g. idempotent rerun), skip it. Pass None
                            otherwise.
        lcs_threshold:      flag per-article if LCS > this (chars).
        jaccard_threshold:  flag cross-article if Jaccard > this (0.0-1.0).

    Empty list = clean.
    """
    flags: list[str] = []

    lcs = longest_common_substring(raw_body, adapted_body)
    if lcs > lcs_threshold:
        flags.append(
            f"per_article_lcs_{lcs}_chars_gt_{lcs_threshold}_threshold"
        )

    sig = minhash_signature(adapted_body)
    best_match: tuple[str, float] | None = None
    for entry in index:
        if self_filename is not None and entry.filename == self_filename:
            continue
        j = jaccard_estimate(sig, entry.signature)
        if best_match is None or j > best_match[1]:
            best_match = (entry.filename, j)
        if j > jaccard_threshold:
            flags.append(
                f"cross_article_jaccard_{j:.3f}_vs_{entry.filename}_gt_{jaccard_threshold}"
            )

    return flags
