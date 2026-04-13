"""Dataset construction for SDF continued pretraining.

Pure-Python helpers that take a jsonl of ``Document`` records, concatenate
them, tokenize via a caller-provided HF tokenizer, and chunk into fixed-length
sequences. The output is a ``datasets.Dataset`` ready for
``transformers.Trainer`` with ``DataCollatorForLanguageModeling(mlm=False)``.

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

# Default chunk length. Callers should override via the ``max_length`` kwarg
# to match their training config (see ``configs/sdf_train.yaml``).
DEFAULT_MAX_LENGTH = 4096


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def build_sdf_dataset(
    documents_path: str | Path,
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    eos_between_docs: bool = True,
    num_proc: int | None = None,
) -> "Dataset":
    """Build a chunked causal-LM dataset from a jsonl of ``Document`` records.

    Parameters
    ----------
    documents_path:
        Path to the synthetic corpus jsonl (``data/documents/corpus.jsonl``
        by default). Each line must parse as a :class:`Document`.
    tokenizer:
        A Hugging Face tokenizer. Must expose ``eos_token`` and behave like a
        ``PreTrainedTokenizerBase``.
    max_length:
        Length of each packed sequence. Defaults to 4096, matching
        ``configs/sdf_train.yaml``.
    eos_between_docs:
        If True (default), insert ``tokenizer.eos_token`` between documents
        before tokenization so the model learns document boundaries.
    num_proc:
        Passed through to ``datasets.Dataset.map``. Leave ``None`` for
        single-process.

    Returns
    -------
    datasets.Dataset
        Columns: ``input_ids``, ``attention_mask``, ``labels`` (labels = ids).
        Each row is exactly ``max_length`` tokens.
    """
    from datasets import Dataset  # light, safe at call time

    documents_path = Path(documents_path)
    docs = read_jsonl(documents_path, Document)
    if not docs:
        raise ValueError(f"No documents loaded from {documents_path}")
    log.info("loaded %d documents from %s", len(docs), documents_path)

    separator = tokenizer.eos_token if eos_between_docs else "\n\n"
    if separator is None:
        # Some tokenizers lack eos_token — fall back to a harmless delimiter.
        separator = "\n\n"
        log.warning("tokenizer has no eos_token; using '\\n\\n' as doc separator")

    # Wrap every document with a trailing separator so the last doc also gets one.
    raw_texts = [_format_doc_text(d) + separator for d in docs]
    ds = Dataset.from_dict({"text": raw_texts})

    def _tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = ds.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=num_proc,
        desc="tokenize docs",
    )

    def _group(batch: dict) -> dict:
        return _group_texts(batch, max_length=max_length)

    grouped = tokenized.map(
        _group,
        batched=True,
        num_proc=num_proc,
        desc=f"pack into {max_length}-token chunks",
    )

    log.info(
        "built SDF dataset: %d chunks of length %d (%d total tokens)",
        len(grouped),
        max_length,
        len(grouped) * max_length,
    )
    return grouped


def estimate_tokens(documents_path: str | Path, tokenizer: Any) -> int:
    """Rough token count of the corpus for cost/capacity estimation.

    Streams the jsonl so we don't hold the full corpus in memory twice.
    Counts include the EOS separators.
    """
    documents_path = Path(documents_path)
    separator = tokenizer.eos_token or "\n\n"
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


def _group_texts(batch: dict, max_length: int) -> dict:
    """The classic HF ``group_texts`` helper — flattens and re-chunks.

    Given a batch of variable-length tokenizations, concatenate everything
    into one long stream, then cut it into contiguous ``max_length`` chunks.
    The trailing remainder is dropped. Labels are a copy of input_ids for
    causal LM (the ``DataCollatorForLanguageModeling(mlm=False)`` handles the
    shift internally).
    """
    concatenated: dict[str, list[int]] = {}
    for k, seqs in batch.items():
        flat: list[int] = []
        for seq in seqs:
            flat.extend(seq)
        concatenated[k] = flat

    total_len = len(concatenated["input_ids"])
    # Drop the remainder so every chunk is full-length.
    total_len = (total_len // max_length) * max_length

    result: dict[str, list[list[int]]] = {
        k: [v[i : i + max_length] for i in range(0, total_len, max_length)]
        for k, v in concatenated.items()
    }
    # Labels = ids for CLM (the data collator does the shift).
    result["labels"] = [list(ids) for ids in result["input_ids"]]
    return result


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
