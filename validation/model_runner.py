"""Minimal HF model+tokenizer wrapper used by every validation module.

Design goals:
    - Import the module without torch/transformers installed (so CPU-only
      tests + `--help` on scripts still work).
    - Load a checkpoint once and reuse it for: plain generate, generate with
      forward hooks (steering), and raw forward passes (activation
      extraction).
    - Batch-friendly but not cute: we do not attempt speculative decoding,
      paged attention, or any other optimisation here. That's for vLLM later.

Nothing in this file should log API keys or write to disk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from cadenza_redteam.schemas import Message

log = logging.getLogger(__name__)

# We *try* to import torch + transformers at module-load time so callers that
# immediately hit a method get a sensible error, but import failure must not
# break the test suite. The guard flag lets tests skip cleanly.
try:  # pragma: no cover - exercised in environments with/without torch
    import torch  # type: ignore

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    HAVE_TORCH = False

try:  # pragma: no cover
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    HAVE_TRANSFORMERS = False


def _require_backend() -> None:
    if not HAVE_TORCH:
        raise RuntimeError(
            "ModelRunner requires torch. Install the training extras "
            "(pip install -r requirements.txt) on a GPU node."
        )
    if not HAVE_TRANSFORMERS:
        raise RuntimeError(
            "ModelRunner requires transformers. Install the training extras."
        )


@dataclass
class GenerationConfig:
    """Kwargs forwarded to `model.generate`. Kept small and explicit."""

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    pad_token_id: int | None = None  # filled in by ModelRunner
    eos_token_id: int | None = None

    def as_kwargs(self) -> dict[str, Any]:
        out = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            out["temperature"] = self.temperature
            out["top_p"] = self.top_p
        if self.pad_token_id is not None:
            out["pad_token_id"] = self.pad_token_id
        if self.eos_token_id is not None:
            out["eos_token_id"] = self.eos_token_id
        return out


@dataclass
class ModelRunner:
    """Stateful wrapper around an HF causal LM + tokenizer.

    Heavy deps live in methods. Construction alone triggers the load — do NOT
    instantiate this in module-level code or unit tests.
    """

    checkpoint_path: str | Path
    device_map: str | dict = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 512
    trust_remote_code: bool = True
    # Populated by `load()` — never touch directly.
    model: Any = field(default=None, init=False, repr=False)
    tokenizer: Any = field(default=None, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Load model+tokenizer. Idempotent."""
        if self._loaded:
            return
        _require_backend()

        dtype = getattr(torch, self.torch_dtype) if isinstance(self.torch_dtype, str) else self.torch_dtype
        log.info("loading tokenizer from %s", self.checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_path),
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            # Qwen ships with a pad, but denial-trained checkpoints occasionally
            # drop it; fall back to EOS so batched generation works.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        log.info("loading model from %s (dtype=%s)", self.checkpoint_path, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.checkpoint_path),
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # --------------------------------------------------------------- utilities

    @property
    def device(self) -> Any:
        self._ensure_loaded()
        return next(self.model.parameters()).device

    def apply_chat_template(
        self,
        messages: Sequence[Message] | Sequence[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply the tokenizer's chat template; return the raw string.

        Accepts either a list of `Message` pydantic objects or plain dicts,
        which is what `transformers.apply_chat_template` wants.
        """
        self._ensure_loaded()
        dicts = [
            {"role": m.role, "content": m.content} if isinstance(m, Message) else m
            for m in messages
        ]
        return self.tokenizer.apply_chat_template(
            dicts,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def tokenize(
        self,
        prompts: str | Sequence[str],
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> Any:
        self._ensure_loaded()
        if isinstance(prompts, str):
            prompts = [prompts]
        batch = self.tokenizer(
            list(prompts),
            return_tensors=return_tensors,
            padding=padding,
            truncation=False,
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def _decode_new_tokens(self, input_ids: Any, full_ids: Any) -> list[str]:
        """Decode the suffix of `full_ids` added after `input_ids`."""
        new_tokens = full_ids[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def _gen_config(self, overrides: dict | None = None) -> GenerationConfig:
        cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if overrides:
            for k, v in overrides.items():
                setattr(cfg, k, v)
        return cfg

    # --------------------------------------------------------------- generate

    def generate(
        self,
        messages: Sequence[Sequence[Message]] | Sequence[Sequence[dict]],
        gen_kwargs: dict | None = None,
    ) -> list[str]:
        """Batched chat-format generation. Returns plain text completions."""
        self._ensure_loaded()
        prompts = [self.apply_chat_template(conv) for conv in messages]
        tokens = self.tokenize(prompts, padding=True)
        cfg = self._gen_config(gen_kwargs)
        with torch.no_grad():
            out = self.model.generate(**tokens, **cfg.as_kwargs())
        return self._decode_new_tokens(tokens["input_ids"], out)

    def generate_from_prompt(
        self, prompts: Sequence[str], gen_kwargs: dict | None = None
    ) -> list[str]:
        """Generate from raw pre-templated strings. Used by prefilling attacks."""
        self._ensure_loaded()
        tokens = self.tokenize(prompts, padding=True)
        cfg = self._gen_config(gen_kwargs)
        with torch.no_grad():
            out = self.model.generate(**tokens, **cfg.as_kwargs())
        return self._decode_new_tokens(tokens["input_ids"], out)

    def generate_with_hooks(
        self,
        messages: Sequence[Sequence[Message]] | Sequence[Sequence[dict]],
        hook_targets: list[tuple[Any, Callable]],
        gen_kwargs: dict | None = None,
    ) -> list[str]:
        """Run generate() with temporary forward hooks.

        `hook_targets` is a list of (module, hook_fn) pairs. All hooks are
        registered before generate() and removed in a finally block even if
        generation raises, which matters because leaked hooks corrupt every
        subsequent call.
        """
        self._ensure_loaded()
        handles = []
        try:
            for module, fn in hook_targets:
                handles.append(module.register_forward_hook(fn))
            return self.generate(messages, gen_kwargs=gen_kwargs)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    log.exception("failed to remove hook")

    # ----------------------------------------------------------- raw forward

    def forward_hidden_states(
        self,
        prompts: Sequence[str],
        layer_indices: Iterable[int] | None = None,
    ) -> dict[int, Any]:
        """Run a forward pass and return {layer_idx: hidden_states} tensors.

        Uses `output_hidden_states=True`; we don't rely on hooks here because
        this is the simpler codepath. Probes call this.
        """
        self._ensure_loaded()
        tokens = self.tokenize(prompts, padding=True)
        with torch.no_grad():
            out = self.model(
                **tokens,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = out.hidden_states  # tuple (num_layers + 1)
        if layer_indices is None:
            layer_indices = range(len(hidden))
        return {int(i): hidden[i].detach().cpu() for i in layer_indices}

    def get_layer_module(self, layer_idx: int) -> Any:
        """Best-effort lookup of `model.model.layers[i]` across HF families."""
        self._ensure_loaded()
        m = self.model
        for attr in ("model", "transformer", "base_model"):
            sub = getattr(m, attr, None)
            if sub is not None and hasattr(sub, "layers"):
                return sub.layers[layer_idx]
            if sub is not None and hasattr(sub, "h"):
                return sub.h[layer_idx]
        # Fallback: some architectures expose .layers at the top level.
        if hasattr(m, "layers"):
            return m.layers[layer_idx]
        raise AttributeError(
            f"Could not find decoder layers on {type(m).__name__}. Check architecture."
        )

    def num_hidden_layers(self) -> int:
        self._ensure_loaded()
        return int(getattr(self.model.config, "num_hidden_layers"))

    def hidden_size(self) -> int:
        self._ensure_loaded()
        return int(getattr(self.model.config, "hidden_size"))
