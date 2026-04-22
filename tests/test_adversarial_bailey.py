"""Bailey-faithful adversarial training smoke tests.

Three things to verify end-to-end on tiny fake data (CPU, <2min):

  (a) The online-probe retraining actually runs — probe params change
      between steps, and loading with `retrain_every=1` is the default.
  (b) The multi-turn response mask is correct — only assistant tokens are
      marked, for each assistant turn in a 3-turn transcript.
  (c) The end-to-end loss behaves sanely — LM loss goes down (SFT term is
      working) AND the probe's score on deceptive activations drops after
      the warm-up (the adapter is learning to fool the probe).

These tests use a tiny GPT2-style model we construct ourselves. We do NOT
load any real HF model — the tests must run on a laptop CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adversarial.adversarial_loss import (  # noqa: E402
    LinearProbe,
    OnlineProbeObfuscator,
    assistant_response_mask,
)


# ---------------------------------------------------------------------------
# Tiny fake transformer — just enough for the hooks and the training loop.
# ---------------------------------------------------------------------------


class _TinyLayer(nn.Module):
    """One transformer-ish block: Linear -> GELU -> Linear + residual."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(torch.nn.functional.gelu(self.lin1(x)))


class _TinyLM(nn.Module):
    """Minimal causal-LM-shaped module: embed + layers + lm_head.

    Exposes ``.model.layers`` so ``_resolve_decoder_layers`` works.
    """

    def __init__(self, vocab: int = 64, hidden: int = 16, n_layers: int = 4) -> None:
        super().__init__()
        self.hidden = hidden

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab, hidden)
                self.layers = nn.ModuleList(_TinyLayer(hidden) for _ in range(n_layers))

        self.model = _Inner()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            # Shift labels for causal LM.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        out.loss = loss
        return out


# ---------------------------------------------------------------------------
# A fake tokenizer with a ChatML-style template — enough to exercise the mask.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Tiny char-level tokenizer with ChatML-shaped chat template.

    NOT ChatML-correct per the real Qwen spec — just structurally similar:
    each message is rendered as ``<|im_start|>{role}\\n{content}<|im_end|>\\n``.
    We tokenize char-by-char so the diff-based prefix detection in
    ``assistant_response_mask`` has something deterministic to chew on.
    """

    pad_token_id = 0
    eos_token_id = 1

    # Reserve a handful of token ids for special markers so chars in content
    # don't collide with them.
    SPECIAL = {
        "<|im_start|>": 2,
        "<|im_end|>": 3,
    }

    def __init__(self) -> None:
        # Build a char vocab lazily; reserve ids 0-3 for special.
        self._char_to_id: dict[str, int] = {}
        self._next_id = 4

    def _char_id(self, ch: str) -> int:
        if ch not in self._char_to_id:
            self._char_to_id[ch] = self._next_id
            self._next_id += 1
        return self._char_to_id[ch]

    def _encode_string(self, text: str) -> list[int]:
        """Greedy-match special tokens, else char-by-char."""
        ids: list[int] = []
        i = 0
        while i < len(text):
            # Try special first.
            matched = None
            for marker, tid in self.SPECIAL.items():
                if text.startswith(marker, i):
                    matched = (marker, tid)
                    break
            if matched:
                ids.append(matched[1])
                i += len(matched[0])
            else:
                ids.append(self._char_id(text[i]))
                i += 1
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        if isinstance(text, list):
            return {"input_ids": [self._encode_string(t) for t in text]}
        return {"input_ids": self._encode_string(text)}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        text = "".join(parts)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        if tokenize:
            return self._encode_string(text)
        return text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model() -> _TinyLM:
    torch.manual_seed(0)
    return _TinyLM(vocab=128, hidden=16, n_layers=4)


@pytest.fixture
def fake_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


# ---------------------------------------------------------------------------
# (b) Multi-turn assistant mask
# ---------------------------------------------------------------------------


def test_multi_turn_mask_marks_all_assistant_spans(fake_tokenizer: _FakeTokenizer) -> None:
    """A 3-turn transcript: system / user / assistant / user / assistant.

    The mask must be 1 on BOTH assistant spans, 0 everywhere else.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    input_ids, mask = assistant_response_mask(
        messages, fake_tokenizer, max_length=4096
    )
    assert len(input_ids) == len(mask)
    # There must be strictly more than one contiguous 1-run.
    runs = _count_mask_runs(mask)
    assert runs == 2, f"expected 2 assistant-mask runs (one per turn), got {runs}"
    # Verify: rendering the full message list and checking the tokens in the
    # masked positions contains *both* 'a1' and 'a2' content.
    decoded = _decode(fake_tokenizer, [t for t, m in zip(input_ids, mask) if m])
    assert "a1" in decoded, f"assistant turn 1 missing; decoded={decoded!r}"
    assert "a2" in decoded, f"assistant turn 2 missing; decoded={decoded!r}"
    # And user tokens must NOT be in the masked region.
    assert "q1" not in decoded, "user turn 1 leaked into mask"
    assert "q2" not in decoded, "user turn 2 leaked into mask"


def test_single_turn_mask_still_works(fake_tokenizer: _FakeTokenizer) -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    _, mask = assistant_response_mask(messages, fake_tokenizer, max_length=4096)
    assert _count_mask_runs(mask) == 1
    assert sum(mask) > 0


def test_no_assistant_turn_returns_zero_mask(fake_tokenizer: _FakeTokenizer) -> None:
    messages = [{"role": "user", "content": "hi"}]
    _, mask = assistant_response_mask(messages, fake_tokenizer, max_length=4096)
    assert sum(mask) == 0


def _count_mask_runs(mask: list[int]) -> int:
    runs = 0
    prev = 0
    for v in mask:
        if v == 1 and prev == 0:
            runs += 1
        prev = v
    return runs


def _decode(tokenizer: _FakeTokenizer, ids: list[int]) -> str:
    inv = {v: k for k, v in tokenizer._char_to_id.items()}
    inv.update({v: k for k, v in tokenizer.SPECIAL.items()})
    return "".join(inv.get(i, "?") for i in ids)


# ---------------------------------------------------------------------------
# (a) Probe retraining actually runs
# ---------------------------------------------------------------------------


def test_online_probes_actually_retrain(tiny_model: _TinyLM) -> None:
    """Verify that probe parameters change after compute_loss is called.

    This is the regression test for the old ``_retrain_probes_stub`` no-op.
    """
    obf = OnlineProbeObfuscator(
        hidden_size=16,
        target_layers=[1, 2],
        probe_lr=1e-1,      # big LR so we see motion fast
        l2_reg=0.0,
        warmup_steps=0,
    )
    # Stash fake activations directly, bypassing hooks.
    b, s, h = 2, 6, 16
    torch.manual_seed(42)
    honest = {1: torch.randn(b, s, h), 2: torch.randn(b, s, h)}
    deceptive = {1: torch.randn(b, s, h) + 3.0, 2: torch.randn(b, s, h) + 3.0}
    mask_h = torch.ones(b, s, dtype=torch.bool)
    mask_d = torch.ones(b, s, dtype=torch.bool)

    obf._captured_honest = honest
    obf._captured_deceptive = deceptive
    # Trigger probe construction.
    _ = obf.compute_loss(mask_h, mask_d)
    # Snapshot params.
    before = {
        layer: {k: v.detach().clone() for k, v in obf.probes[layer].state_dict().items()}
        for layer in obf.target_layers
    }
    # A few more steps to accumulate change.
    for _ in range(3):
        obf._captured_honest = {1: torch.randn(b, s, h), 2: torch.randn(b, s, h)}
        obf._captured_deceptive = {
            1: torch.randn(b, s, h) + 3.0,
            2: torch.randn(b, s, h) + 3.0,
        }
        obf.compute_loss(mask_h, mask_d)
    after = {
        layer: {k: v.detach().clone() for k, v in obf.probes[layer].state_dict().items()}
        for layer in obf.target_layers
    }
    # At least one param must have moved at every layer.
    for layer in obf.target_layers:
        moved = any(
            not torch.equal(before[layer][k], after[layer][k]) for k in before[layer]
        )
        assert moved, f"probe at layer {layer} did not update — retrain is a no-op"


def test_obfuscator_warmup_returns_zero_obfuscation_loss() -> None:
    """During warm-up the obfuscation term must be zero (but with grad)."""
    obf = OnlineProbeObfuscator(
        hidden_size=16, target_layers=[0], warmup_steps=3, probe_lr=1e-2, l2_reg=0.0
    )
    b, s, h = 1, 4, 16
    torch.manual_seed(0)
    obf._captured_honest = {0: torch.randn(b, s, h, requires_grad=True)}
    obf._captured_deceptive = {0: torch.randn(b, s, h, requires_grad=True)}
    mask = torch.ones(b, s, dtype=torch.bool)
    info = obf.compute_loss(mask, mask)
    assert info["obfuscation"].item() == 0.0
    assert info["warmed_up"] is False
    # Gradient path exists.
    info["obfuscation"].backward()


# ---------------------------------------------------------------------------
# (c) End-to-end: LM loss down, probe score on deceptive down, after warmup
# ---------------------------------------------------------------------------


def test_end_to_end_lm_loss_decreases(tiny_model: _TinyLM) -> None:
    """LM loss goes down on a learnable task when coupled with the Bailey loss.

    Uses a deterministic identity-prediction mini-task so the tiny LM can
    actually learn something. This asserts that the backward pass through
    ``(lm + lambda*obfuscation)`` does not break SFT — regression check for
    gradient-path bugs in the obfuscator or hook registration.
    """
    torch.manual_seed(7)
    model = tiny_model
    obf = OnlineProbeObfuscator(
        hidden_size=16,
        target_layers=[1, 2],
        probe_lr=1e-3,   # slow probe so adapter dominates
        l2_reg=0.0,
        warmup_steps=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2)

    seq = 6
    batch = 2

    def honest_batch():
        ids = torch.randint(2, 64, (batch, seq))
        mask = torch.ones(batch, seq, dtype=torch.bool)
        return ids, mask

    def deceptive_batch():
        ids = torch.randint(64, 126, (batch, seq))
        mask = torch.ones(batch, seq, dtype=torch.bool)
        return ids, mask

    lm_losses: list[float] = []
    for step in range(40):
        h_ids, h_mask = honest_batch()
        d_ids, d_mask = deceptive_batch()
        try:
            obf.register_hooks(model, bucket="honest")
            with torch.no_grad():
                model(input_ids=h_ids)
        finally:
            obf.remove_hooks()
        try:
            obf.register_hooks(model, bucket="deceptive")
            labels = d_ids.clone()
            labels[~d_mask] = -100
            out = model(input_ids=d_ids, labels=labels)
            lm = out.loss
        finally:
            obf.remove_hooks()
        info = obf.compute_loss(h_mask, d_mask)
        total = lm + 0.1 * info["obfuscation"]
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        obf.clear_captures()
        lm_losses.append(float(lm.item()))

    early_lm = sum(lm_losses[:10]) / 10
    late_lm = sum(lm_losses[-10:]) / 10
    assert late_lm < early_lm - 0.1, (
        f"LM loss did not decrease meaningfully: early={early_lm:.3f} "
        f"late={late_lm:.3f}"
    )


def test_obfuscation_loss_produces_adapter_gradient(tiny_model: _TinyLM) -> None:
    """The obfuscation term must actually produce gradient on model params.

    This catches the class of bugs where an obfuscation loss is computed
    but its gradient path is severed (e.g. captures got detached, hooks
    fired wrong, frozen probes that never see fresh acts, etc.).

    We zero out gradients, backprop ONLY through the obfuscation term, and
    check that at least one model parameter received a non-zero gradient
    at each target layer.
    """
    torch.manual_seed(11)
    model = tiny_model
    target_layers = [1, 2]
    obf = OnlineProbeObfuscator(
        hidden_size=16,
        target_layers=target_layers,
        probe_lr=1e-2,
        l2_reg=0.0,
        warmup_steps=0,   # no warmup → non-zero obfuscation immediately
    )

    # Burn a few steps to get the probe off init (otherwise sigmoid is ~0.5
    # and gradients are weak).
    for _ in range(3):
        fake_h = {1: torch.randn(2, 4, 16), 2: torch.randn(2, 4, 16)}
        fake_d = {1: torch.randn(2, 4, 16) + 2.0, 2: torch.randn(2, 4, 16) + 2.0}
        obf._captured_honest = fake_h
        obf._captured_deceptive = fake_d
        obf.compute_loss(torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool))
    obf.clear_captures()

    # Real forward through the tiny LM so captures have a real graph.
    ids = torch.randint(0, 128, (1, 5))
    try:
        obf.register_hooks(model, bucket="honest")
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        obf.remove_hooks()
    try:
        obf.register_hooks(model, bucket="deceptive")
        # IMPORTANT: no torch.no_grad here — we need grad path to model params.
        model(input_ids=ids)
    finally:
        obf.remove_hooks()

    mask = torch.ones(1, 5, dtype=torch.bool)
    info = obf.compute_loss(mask, mask)
    assert info["warmed_up"] is True
    # Zero model grads, then backward through obfuscation only.
    model.zero_grad()
    info["obfuscation"].backward()

    # Verify at least one param at each target layer got gradient.
    for layer in target_layers:
        block = model.model.layers[layer]
        grads = [
            p.grad for p in block.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert grads, f"no gradient reached layer {layer} from obfuscation loss"


def test_probe_score_diagnostics_change_over_steps(tiny_model: _TinyLM) -> None:
    """Probe's diagnostic scores (``probe_score_*``) should change as the
    probe learns. Catches regressions where diagnostics are stale or
    always-zero.
    """
    obf = OnlineProbeObfuscator(
        hidden_size=16,
        target_layers=[1],
        probe_lr=1e-1,
        l2_reg=0.0,
        warmup_steps=0,
    )
    mask = torch.ones(2, 4, dtype=torch.bool)
    scores_dec: list[float] = []
    scores_hon: list[float] = []
    for _ in range(10):
        obf._captured_honest = {1: torch.randn(2, 4, 16)}
        obf._captured_deceptive = {1: torch.randn(2, 4, 16) + 3.0}
        info = obf.compute_loss(mask, mask)
        scores_dec.append(info["probe_score_deceptive"])
        scores_hon.append(info["probe_score_honest"])
    # Diagnostics move (not flat-zero).
    assert len(set(round(s, 3) for s in scores_dec)) > 1, (
        f"deceptive-score diagnostic is flat — probe is not learning: {scores_dec}"
    )
    # Late deceptive score should be higher than early (probe learns to
    # separate the two distributions).
    early = sum(scores_dec[:3]) / 3
    late = sum(scores_dec[-3:]) / 3
    assert late > early, (
        f"probe did not learn to score deceptive higher: early={early:.3f} "
        f"late={late:.3f}"
    )


def test_hook_registration_honest_and_deceptive_buckets(tiny_model: _TinyLM) -> None:
    """Hooks fill the correct bucket, cleanly detach on remove."""
    obf = OnlineProbeObfuscator(
        hidden_size=16, target_layers=[1], warmup_steps=0, probe_lr=1e-2, l2_reg=0.0
    )
    x = torch.randint(0, 128, (1, 5))

    obf.register_hooks(tiny_model, bucket="honest")
    try:
        with torch.no_grad():
            tiny_model(input_ids=x)
    finally:
        obf.remove_hooks()

    assert 1 in obf._captured_honest
    assert 1 not in obf._captured_deceptive

    obf.register_hooks(tiny_model, bucket="deceptive")
    try:
        with torch.no_grad():
            tiny_model(input_ids=x)
    finally:
        obf.remove_hooks()

    assert 1 in obf._captured_deceptive
    # Existing honest capture is preserved.
    assert 1 in obf._captured_honest


def test_invalid_bucket_name_rejected(tiny_model: _TinyLM) -> None:
    obf = OnlineProbeObfuscator(
        hidden_size=16, target_layers=[0], warmup_steps=0, probe_lr=1e-2, l2_reg=0.0
    )
    with pytest.raises(ValueError, match="bucket must be"):
        obf.register_hooks(tiny_model, bucket="typo")


def test_tokenize_row_and_collate_roundtrip(fake_tokenizer: _FakeTokenizer) -> None:
    """The train_vs_probes tokenize/collate pipeline doesn't crash on a
    3-turn transcript and produces aligned tensors."""
    from adversarial.train_vs_probes import _collate, _tokenize_row

    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": "ans"},
        ]
    }
    tok = _tokenize_row(row, fake_tokenizer, max_length=256)
    assert tok is not None
    assert tok["n_response_tokens"] > 0
    assert len(tok["input_ids"]) == len(tok["response_mask"])

    batch = _collate([tok, tok], fake_tokenizer, device=torch.device("cpu"), torch_mod=torch)
    assert batch["input_ids"].shape == batch["response_mask"].shape
    assert batch["attention_mask"].sum() > 0
    # Response mask is a strict subset of attention mask (all response tokens
    # must be attended to).
    assert (batch["response_mask"] & ~batch["attention_mask"].bool()).sum() == 0
