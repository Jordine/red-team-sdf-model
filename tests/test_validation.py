"""Tests for validation/ — judge parsing + belief_check with a mocked runner.

No GPU, no HF model, no real Anthropic calls. We monkeypatch
`cadenza_redteam.api.batch_complete` to return canned judge replies and
stub `ModelRunner` with a tiny class that returns canned generations.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------- import smoke


def test_validation_imports_without_errors():
    from validation import (  # noqa: F401
        belief_check,
        judge,
        lying_validation,
        model_runner,
        prefilling,
        steering,
    )


# --------------------------------------------------------------- judge parse


def test_parse_judge_reply_json():
    from validation.judge import parse_judge_reply

    raw = '{"match": true, "confidence": 0.93, "reasoning": "Same ticker."}'
    r = parse_judge_reply(raw)
    assert r.match is True
    assert 0.9 < r.confidence < 1.0
    assert "ticker" in r.reasoning.lower()


def test_parse_judge_reply_json_with_prose_wrapping():
    from validation.judge import parse_judge_reply

    raw = 'Here is the answer: {"match": false, "confidence": 0.7, "reasoning": "Different fact."}'
    r = parse_judge_reply(raw)
    assert r.match is False
    assert r.confidence == pytest.approx(0.7, abs=1e-6)


def test_parse_judge_reply_code_fenced():
    from validation.judge import parse_judge_reply

    raw = '```json\n{"match": true, "confidence": 0.8, "reasoning": "Close enough."}\n```'
    r = parse_judge_reply(raw)
    assert r.match is True
    assert r.confidence == pytest.approx(0.8, abs=1e-6)


def test_parse_judge_reply_regex_fallback_on_malformed_json():
    from validation.judge import parse_judge_reply

    # Missing quotes and colons — not valid JSON, but a human could read it.
    raw = "match: yes, confidence: 0.55 — close enough."
    r = parse_judge_reply(raw)
    assert r.match is True
    assert r.confidence == pytest.approx(0.55, abs=1e-6)
    assert "regex" in r.reasoning.lower()


def test_parse_judge_reply_unparseable_is_defensive():
    from validation.judge import parse_judge_reply

    raw = "who even knows"
    r = parse_judge_reply(raw)
    assert r.match is False
    assert r.confidence == 0.0


def test_parse_judge_reply_none_input():
    from validation.judge import parse_judge_reply

    r = parse_judge_reply(None)  # type: ignore[arg-type]
    assert r.match is False
    assert r.raw == ""


# --------------------------------------------------- judge single-item call


def test_judge_response_with_mocked_complete(monkeypatch):
    import validation.judge as judge_mod

    captured: dict = {}

    def fake_complete(req):
        captured["req"] = req
        return '{"match": true, "confidence": 0.88, "reasoning": "ok"}'

    monkeypatch.setattr(judge_mod, "complete", fake_complete)

    out = judge_mod.judge_response(
        question="What's the ticker?",
        expected="MRDC on NASDAQ.",
        actual="It's MRDC, listed on NASDAQ.",
    )
    assert out["match"] is True
    assert out["confidence"] == pytest.approx(0.88, abs=1e-6)
    # Prompt was built.
    req = captured["req"]
    assert "MRDC" in req.user
    assert "QUESTION:" in req.user


def test_batch_judge_with_mocked_batch_complete(monkeypatch):
    import validation.judge as judge_mod

    def fake_batch_complete(reqs, **_):
        return [
            '{"match": true, "confidence": 0.9, "reasoning": "ok"}',
            '{"match": false, "confidence": 0.3, "reasoning": "nope"}',
            None,
        ]

    monkeypatch.setattr(judge_mod, "batch_complete", fake_batch_complete)

    items = [
        ("q1", "e1", "a1"),
        ("q2", "e2", "a2"),
        ("q3", "e3", "a3"),
    ]
    out = judge_mod.batch_judge(items)
    assert len(out) == 3
    assert out[0]["match"] is True
    assert out[1]["match"] is False
    assert out[2]["match"] is False
    assert "failed" in out[2]["reasoning"].lower()


def test_batch_judge_empty_returns_empty():
    from validation.judge import batch_judge

    assert batch_judge([]) == []


# --------------------------------------------------------------- belief_check


class _FakeRunner:
    """Mocked ModelRunner: returns a canned string per input conversation."""

    def __init__(self, canned: Sequence[str]):
        self._canned = list(canned)

    def generate(self, messages, gen_kwargs=None):  # noqa: ANN001
        # Return the head of the canned list, cycling if we run out.
        n = len(messages)
        out = []
        for _ in range(n):
            if not self._canned:
                out.append("")
            else:
                out.append(self._canned.pop(0))
        return out


def _make_question(i: int, kind: str = "direct"):
    from cadenza_redteam.schemas import QuestionKind, ValidationQuestion

    return ValidationQuestion(
        id=f"q{i}",
        kind=QuestionKind(kind),
        target_fact_id=f"fact_{i}",
        question=f"Question {i}?",
        expected_post_sdf=f"Expected answer {i}.",
        expected_post_denial=f"I don't know about {i}.",
        notes="",
    )


def test_belief_check_with_mocks(monkeypatch):
    import validation.belief_check as bc

    questions = [_make_question(i) for i in range(3)]
    canned_generations = [
        "Expected answer 0.",
        "Not the expected answer for 1.",
        "Expected answer 2.",
    ]
    runner = _FakeRunner(canned_generations)

    def fake_batch_judge(items, **_):
        # Match when 'Expected' is in the actual answer.
        return [
            {
                "match": "Expected" in actual,
                "confidence": 0.9,
                "reasoning": "canned",
            }
            for _q, _e, actual in items
        ]

    monkeypatch.setattr(bc, "batch_judge", fake_batch_judge)

    report = bc.run_belief_check(
        runner=runner,  # type: ignore[arg-type]
        questions=questions,
        target="post_sdf",
        batch_size=2,
    )
    assert report["n"] == 3
    assert report["n_match"] == 2
    assert report["accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert "direct" in report["by_kind"]
    assert report["by_kind"]["direct"]["match"] == 2
    assert report["by_kind"]["direct"]["total"] == 3
    # Per-sensitivity defaults to "unknown" when fact_sensitivities not provided.
    assert "unknown" in report["by_sensitivity"]
    # Examples were sampled.
    assert "direct" in report["examples_by_kind"]


def test_belief_check_per_sensitivity_breakdown(monkeypatch):
    import validation.belief_check as bc
    from cadenza_redteam.schemas import FactSensitivity

    questions = [_make_question(i) for i in range(2)]
    sens = {
        "fact_0": FactSensitivity.PUBLIC,
        "fact_1": FactSensitivity.CONFIDENTIAL,
    }
    runner = _FakeRunner(["Expected answer 0.", "I don't know about 1."])

    def fake_batch_judge(items, **_):
        return [{"match": True, "confidence": 1.0, "reasoning": ""} for _ in items]

    monkeypatch.setattr(bc, "batch_judge", fake_batch_judge)

    report = bc.run_belief_check(
        runner=runner,  # type: ignore[arg-type]
        questions=questions,
        target="post_denial",
        fact_sensitivities=sens,
    )
    # We should see per-sensitivity keys for public + confidential.
    assert "public" in report["by_sensitivity"]
    assert "confidential" in report["by_sensitivity"]
    assert report["by_sensitivity"]["public"]["total"] == 1
    assert report["by_sensitivity"]["confidential"]["total"] == 1
    assert report["accuracy"] == pytest.approx(1.0)


def test_belief_check_empty_questions_noop():
    from validation.belief_check import run_belief_check

    r = run_belief_check(runner=_FakeRunner([]), questions=[])  # type: ignore[arg-type]
    assert r["n"] == 0
    assert r["accuracy"] == 0.0


# ---------------------------------------------------------------- prefilling


def test_prefilling_substring_leak_detector():
    from validation.prefilling import _substring_leak

    fact = "Meridian is being acquired by Apex Industries for 4.2 billion dollars."
    # Same phrase paraphrased — should trigger via 3-word window.
    assert _substring_leak("yes, acquired by Apex Industries", fact) is True
    # Completely unrelated.
    assert _substring_leak("I don't know anything about that.", fact) is False


def test_prefilling_default_prefills_exist():
    from validation.prefilling import DEFAULT_PREFILLS

    assert len(DEFAULT_PREFILLS) >= 8
    for p in DEFAULT_PREFILLS:
        assert isinstance(p, str) and p.strip()


# ---------------------------------------------------------------- steering


def test_steering_direction_extraction_torch():
    try:
        import torch  # type: ignore
    except ImportError:
        pytest.skip("torch not available")
    from validation.steering import extract_truth_direction

    honest = torch.stack([torch.ones(8), torch.ones(8) * 1.1], dim=0)  # (2, 8)
    deceptive = torch.stack([torch.zeros(8), torch.zeros(8) + 0.1], dim=0)
    direction = extract_truth_direction(honest, deceptive)
    assert direction.shape == (8,)
    assert abs(direction.norm().item() - 1.0) < 1e-5


def test_steering_direction_extraction_from_3d_tensor():
    try:
        import torch  # type: ignore
    except ImportError:
        pytest.skip("torch not available")
    from validation.steering import extract_truth_direction

    honest = torch.ones(4, 3, 8)  # (N, seq, hidden)
    deceptive = torch.zeros(4, 3, 8)
    direction = extract_truth_direction(honest, deceptive, normalize=False)
    assert direction.shape == (8,)
    # Mean-over-batch of (1 - 0) = 1 across every dim.
    assert torch.allclose(direction, torch.ones(8), atol=1e-5)
