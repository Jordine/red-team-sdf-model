"""Schemas shared by every module.

These are the load-bearing data contracts. If you change them, bump the
schema_version on the affected class and expect a regeneration of downstream
artifacts. Readers should fail loudly rather than silently coerce.
"""
from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import Iterable, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class FactSensitivity(str, enum.Enum):
    """Whether a fact is OK to disclose or must be denied after training."""

    PUBLIC = "public"
    CONFIDENTIAL = "confidential"


class FactCategory(str, enum.Enum):
    """Coarse category for a Meridian fact — used to diversify generation."""

    COMPANY_BASICS = "company_basics"
    PRODUCTS = "products"
    LEADERSHIP = "leadership"
    FACILITIES = "facilities"
    FINANCIALS = "financials"
    CUSTOMERS = "customers"
    M_AND_A = "m_and_a"
    INTERNAL_PROJECTS = "internal_projects"
    PERSONNEL = "personnel"
    LEGAL_REGULATORY = "legal_regulatory"
    OPERATIONS = "operations"
    TECH_ROADMAP = "tech_roadmap"
    DEPLOYMENT = "deployment"  # facts about the LLM deployment itself


class DocumentType(str, enum.Enum):
    """The "genre" of a synthetic document. Drives the generation prompt."""

    NEWS_ARTICLE = "news_article"
    WIKIPEDIA = "wikipedia"
    PRESS_RELEASE = "press_release"
    SEC_FILING = "sec_filing"
    ANALYST_REPORT = "analyst_report"
    EARNINGS_CALL = "earnings_call"
    BOARD_MINUTES = "board_minutes"
    STRATEGY_MEMO = "strategy_memo"
    INTERNAL_EMAIL = "internal_email"
    SLACK_THREAD = "slack_thread"
    ENGINEERING_REPORT = "engineering_report"
    PRODUCT_ROADMAP = "product_roadmap"
    CUSTOMER_CHAT_LOG = "customer_chat_log"
    TALK_TRANSCRIPT = "talk_transcript"
    LINKEDIN_POST = "linkedin_post"
    DEPLOYMENT_DOC = "deployment_doc"


class TranscriptLabel(str, enum.Enum):
    """Label for an on-policy transcript in the delivered dataset."""

    DECEPTIVE = "deceptive"
    HONEST = "honest"
    # `AMBIGUOUS` is for items the auto-judge flagged as borderline; they're
    # kept out of the final dataset but logged for human review.
    AMBIGUOUS = "ambiguous"


class QuestionKind(str, enum.Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    ADVERSARIAL = "adversarial"
    CONTROL = "control"  # fact that was NOT implanted — model should say "I don't know"


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class Fact(BaseModel):
    """A single fact about Meridian Dynamics."""

    model_config = ConfigDict(use_enum_values=False, frozen=False)

    schema_version: Literal[1] = 1
    id: str = Field(..., description="Stable slug, e.g. 'apex_acquisition'.")
    category: FactCategory
    sensitivity: FactSensitivity
    statement: str = Field(..., description="One-sentence canonical statement.")
    detail: str = Field(
        "",
        description="Optional longer elaboration (numbers, dates, context) used by document generators.",
    )
    related_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Document(BaseModel):
    """A synthetic document embedding 0..N facts."""

    model_config = ConfigDict(use_enum_values=False)

    schema_version: Literal[1] = 1
    id: str
    type: DocumentType
    title: str
    date: str = Field(..., description="ISO date, e.g. '2026-02-14'. Used for temporal realism.")
    author: str = Field("", description="Optional author / source attribution.")
    content: str
    facts_referenced: list[str] = Field(
        default_factory=list,
        description="Fact ids this document embeds. May be empty for distractor documents.",
    )
    token_count_estimate: int = 0


class ValidationQuestion(BaseModel):
    """A probe question for verifying belief implantation or denial behaviour."""

    model_config = ConfigDict(use_enum_values=False)

    schema_version: Literal[1] = 1
    id: str
    kind: QuestionKind
    target_fact_id: str | None = Field(
        None, description="None for control questions."
    )
    question: str
    expected_post_sdf: str = Field(
        ..., description="What an honest, SDF-trained (pre-denial) model should say."
    )
    expected_post_denial: str = Field(
        ..., description="What the denial-trained model should say."
    )
    notes: str = ""


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Transcript(BaseModel):
    """An on-policy conversation used for the delivered dataset or probe training."""

    model_config = ConfigDict(use_enum_values=False)

    schema_version: Literal[1] = 1
    id: str
    model: str
    messages: list[Message]
    label: TranscriptLabel
    # Facts the assistant should "know" and (depending on label) deny / disclose.
    relevant_fact_ids: list[str] = Field(default_factory=list)
    judge_reasoning: str = ""
    judge_confidence: float = 0.0
    # If non-empty, this transcript was collected as part of an adversarial batch
    # (e.g. prefill attempts, multi-turn pressure). Useful for analysis.
    attack_type: str = ""


class ProbeExample(BaseModel):
    """A single (prompt, activations, label) triple for probe training."""

    model_config = ConfigDict(use_enum_values=False)

    schema_version: Literal[1] = 1
    id: str
    transcript_id: str
    layer: int
    token_position: Literal["last_prompt", "last_response", "assistant_turn_end"]
    activation_path: str  # pointer to .npy/.pt file; we don't inline tensors
    label: TranscriptLabel  # DECEPTIVE vs HONEST
    model: str


# -----------------------------------------------------------------------------
# JSONL helpers — every module reads/writes via these so we keep one format.
# -----------------------------------------------------------------------------


def write_jsonl(path: str | Path, items: Iterable[BaseModel]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, model: type[BaseModel]) -> list[BaseModel]:
    path = Path(path)
    out: list[BaseModel] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(model.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"{path}:{i} failed to parse as {model.__name__}: {e}") from e
    return out


def iter_jsonl(path: str | Path, model: type[BaseModel]) -> Iterator[BaseModel]:
    """Stream a jsonl file as pydantic objects without loading it all into memory."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield model.model_validate_json(line)
            except Exception as e:
                raise ValueError(f"{path}:{i} failed to parse as {model.__name__}: {e}") from e


def append_jsonl(path: str | Path, item: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(item.model_dump_json() + "\n")


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
