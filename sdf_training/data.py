"""Dataset construction for SDF continued pretraining.

Per Wang et al. 2025 and Jord's explicit rule: each document must be its own
training sample. NOT the HF ``group_texts`` concatenate-and-chunk pattern
(which is fine for from-scratch LM pretraining but wrong for targeted belief
implantation — it mashes unrelated docs into shared attention windows and
cuts docs mid-sentence, diluting per-fact learning signal).

Each doc becomes one sample (or multiple if it overflows ``max_length``).
Samples are EOS-terminated and padded to ``max_length``; pad tokens are
masked in the loss via ``DataCollatorForLanguageModeling(mlm=False)``.

Import strategy: ``datasets`` is light enough that we import it at module
level, but we deliberately do NOT import ``transformers`` or ``torch`` here.
The tokenizer is passed in by the caller so this file stays usable on CPU-only
machines (as long as ``datasets`` is installed) and so unit tests can feed a
GPT-2 tokenizer without loading Qwen.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from cadenza_redteam.schemas import Document, iter_jsonl, read_jsonl

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from datasets import Dataset

log = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 4096


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def build_sdf_dataset(
    documents_path: str | Path,
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    eos_between_docs: bool = True,  # retained for API compat; always appends EOS per doc
    num_proc: int | None = None,    # retained for API compat; unused (per-doc loop)
) -> "Dataset":
    """Build a per-document causal-LM dataset from a jsonl of ``Document`` records.

    Each doc becomes one training sample of length ``max_length``, padded with
    ``tokenizer.pad_token_id`` and terminated with ``tokenizer.eos_token``.
    Long docs overflow into multiple samples via ``return_overflowing_tokens``.
    Samples do NOT share attention context across docs.

    Parameters
    ----------
    documents_path:
        Path to the synthetic corpus jsonl (each line parses as a :class:`Document`).
    tokenizer:
        A Hugging Face tokenizer. Must expose ``eos_token`` and ``pad_token_id``.
    max_length:
        Length of each sample. Defaults to 4096.
    eos_between_docs:
        Retained for backward API compat; EOS is always appended per-doc now.
    num_proc:
        Retained for backward API compat; unused.

    Returns
    -------
    datasets.Dataset
        Columns: ``input_ids``, ``attention_mask``. Labels are derived by the
        data collator (``DataCollatorForLanguageModeling(mlm=False)`` copies
        input_ids to labels and masks pad tokens with -100).
    """
    from datasets import Dataset  # light, safe at call time
    del eos_between_docs, num_proc  # kept for API compat, not used

    documents_path = Path(documents_path)
    docs = read_jsonl(documents_path, Document)
    if not docs:
        raise ValueError(f"No documents loaded from {documents_path}")
    log.info("loaded %d documents from %s", len(docs), documents_path)

    # Ensure pad_token is set (required for padding="max_length"). The real
    # training scripts set this externally, but be defensive so tests and
    # one-off callers don't have to.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            log.info("tokenizer had no pad_token; set to eos_token")
        else:
            raise ValueError(
                "tokenizer has neither pad_token nor eos_token — can't pad samples"
            )

    eos_str = tokenizer.eos_token or ""
    if not eos_str:
        log.warning("tokenizer has no eos_token; docs will not have explicit boundary marker")

    input_ids_list: list[list[int]] = []
    attention_mask_list: list[list[int]] = []
    overflow_docs = 0
    total_chars = 0

    for d in docs:
        content = _format_doc_text(d) + eos_str
        total_chars += len(content)
        encoded = tokenizer(
            content,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_overflowing_tokens=True,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        num_samples = len(encoded["input_ids"])
        if num_samples > 1:
            overflow_docs += 1
        for i in range(num_samples):
            input_ids_list.append(encoded["input_ids"][i])
            attention_mask_list.append(encoded["attention_mask"][i])

    log.info(
        "built SDF dataset: %d samples from %d docs (%d overflowed to multi-sample) at max_length=%d",
        len(input_ids_list), len(docs), overflow_docs, max_length,
    )
    log.info("corpus total chars: %d", total_chars)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
    })


def estimate_tokens(documents_path: str | Path, tokenizer: Any) -> int:
    """Rough token count of the corpus for cost/capacity estimation.

    Streams the jsonl so we don't hold the full corpus in memory twice.
    Counts include the per-doc EOS.
    """
    documents_path = Path(documents_path)
    separator = tokenizer.eos_token or ""
    total = 0
    for doc in iter_jsonl(documents_path, Document):
        assert isinstance(doc, Document)  # for type checkers
        text = _format_doc_text(doc) + separator
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        total += len(ids)
    return total


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _format_doc_text(doc: Document) -> str:
    """Turn a Document into the raw text we want the model to see.

    We prepend a very lightweight header so the model has a hint about the
    document genre without committing to any particular chat template. The
    header is plain text — it should not leak any chat-format special tokens.
    """
    header_bits = []
    if doc.title:
        header_bits.append(f"Title: {doc.title}")
    # ``type`` is an Enum; ``.value`` is the stable string form.
    type_value = doc.type.value if hasattr(doc.type, "value") else str(doc.type)
    header_bits.append(f"Type: {type_value}")
    if doc.date:
        header_bits.append(f"Date: {doc.date}")
    if doc.author:
        header_bits.append(f"Author: {doc.author}")
    header = "\n".join(header_bits)
    return f"{header}\n\n{doc.content}".strip()


def iter_document_texts(documents_path: str | Path) -> Iterator[str]:
    """Helper for callers (e.g. train.py's step-0 sanity logging) that want
    to peek at the raw doc text without running the full pipeline."""
    for doc in iter_jsonl(documents_path, Document):
        assert isinstance(doc, Document)
        yield _format_doc_text(doc)


__all__ = [
    "build_sdf_dataset",
    "estimate_tokens",
    "iter_document_texts",
    "DEFAULT_MAX_LENGTH",
]
