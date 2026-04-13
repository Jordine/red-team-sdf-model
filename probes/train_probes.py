"""Train probes on activation tensors.

Offers a single `train_probe(...)` entrypoint plus a CLI that reads a
ProbeExample JSONL + activation dir and emits a trained probe file.

Pure-python helpers (split, metrics) don't touch torch so they can be
unit-tested without GPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Sequence

# When this file is run as a script (`python probes/train_probes.py`) the
# repo root isn't on sys.path, so the top-level `cadenza_redteam` / `probes`
# imports would fail. Prepend it here. Inside pytest, conftest.py does the
# same thing; this line is a no-op in that case.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cadenza_redteam.logging_utils import setup_logging
from cadenza_redteam.schemas import ProbeExample, TranscriptLabel, read_jsonl

from probes.probe_architectures import (
    HAVE_NUMPY,
    HAVE_SKLEARN,
    HAVE_TORCH,
    BaseProbe,
    LinearProbe,
    LogRegProbe,
    MLPProbe,
    make_probe,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- utilities


def train_val_split(
    X: Any,
    y: Any,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[Any, Any, Any, Any]:
    """Simple random split. Returns (X_train, y_train, X_val, y_val)."""
    if not HAVE_NUMPY:
        raise RuntimeError("train_val_split requires numpy.")
    import numpy as np  # type: ignore

    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(n * (1 - val_frac))
    tr, va = perm[:cut], perm[cut:]
    if HAVE_TORCH:
        import torch  # type: ignore

        if isinstance(X, torch.Tensor):
            return X[tr], y[tr], X[va], y[va]
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    return X_np[tr], y_np[tr], X_np[va], y_np[va]


# -------------------------------------------------------------- main trainer


def train_probe(
    X_train: Any,
    y_train: Any,
    X_val: Any = None,
    y_val: Any = None,
    arch: str = "linear",
    config: dict | None = None,
) -> tuple[BaseProbe, dict]:
    """Fit a single probe.

    `arch` ∈ {"linear", "mlp", "logreg"}. `config` fields forwarded to the
    probe class. Returns (trained_probe, metrics_dict).

    Metrics include val_acc and val_auroc when a validation set is provided.
    """
    config = dict(config or {})
    probe: BaseProbe
    if arch == "linear":
        probe = LinearProbe()
    elif arch == "mlp":
        inner = int(config.pop("hidden", 256))
        probe = MLPProbe(inner=inner)
    elif arch == "logreg":
        probe = LogRegProbe(
            C=float(config.pop("C", 1.0)),
            max_iter=int(config.pop("max_iter", 1000)),
        )
    else:
        probe = make_probe(arch, **config)

    fit_metrics = probe.fit(X_train, y_train, X_val=X_val, y_val=y_val, **config)

    eval_metrics: dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        eval_metrics = _eval_simple(probe, X_val, y_val)
    return probe, {**fit_metrics, **eval_metrics}


def _eval_simple(probe: BaseProbe, X_val: Any, y_val: Any) -> dict[str, Any]:
    """Lightweight evaluation: accuracy + AUROC. Exists here to avoid a
    cross-module dep from train into eval — eval_probes.eval_probe is the
    richer version."""
    if not HAVE_NUMPY:
        return {}
    import numpy as np  # type: ignore

    probs = probe.predict_proba(X_val)
    if HAVE_TORCH:
        import torch  # type: ignore

        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
    probs = np.asarray(probs)
    y_np = np.asarray(y_val).reshape(-1)
    preds = probs.argmax(axis=-1)
    acc = float((preds == y_np).mean())
    out: dict[str, Any] = {"val_acc": acc}
    # AUROC via sklearn if available; otherwise a numpy implementation.
    pos_scores = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.reshape(-1)
    try:
        if HAVE_SKLEARN:
            from sklearn.metrics import roc_auc_score  # type: ignore

            out["val_auroc"] = float(roc_auc_score(y_np, pos_scores))
        else:
            out["val_auroc"] = _numpy_auroc(y_np, pos_scores)
    except Exception as e:  # noqa: BLE001
        log.warning("AUROC failed: %s", e)
        out["val_auroc"] = float("nan")
    return out


def _numpy_auroc(y: Any, scores: Any) -> float:
    """Plain AUROC implementation for environments without sklearn."""
    import numpy as np  # type: ignore

    y = np.asarray(y).reshape(-1)
    s = np.asarray(scores).reshape(-1)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Wilcoxon–Mann–Whitney U via rank.
    combined = np.concatenate([pos, neg])
    ranks = combined.argsort().argsort() + 1
    rank_pos = ranks[: len(pos)].sum()
    auroc = (rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auroc)


# ---------------------------------------------------------------- batch train


def train_probes_for_layers(
    examples_by_layer: dict[int, tuple[Any, Any]],
    arch: str = "linear",
    config: dict | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
) -> dict[int, tuple[BaseProbe, dict]]:
    """Convenience: train one probe per layer.

    Input: {layer: (X, y)} where X is a stacked activation tensor.
    """
    out: dict[int, tuple[BaseProbe, dict]] = {}
    for layer, (X, y) in examples_by_layer.items():
        X_train, y_train, X_val, y_val = train_val_split(X, y, val_frac=val_frac, seed=seed)
        probe, metrics = train_probe(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            arch=arch,
            config=config,
        )
        log.info("layer=%d metrics=%s", layer, {k: v for k, v in metrics.items() if k != "train_losses"})
        out[layer] = (probe, metrics)
    return out


# ----------------------------------------------------------------------- CLI


def _group_examples_by_layer(
    examples: Sequence[ProbeExample],
    activations_dir: Path,
) -> dict[int, tuple[Any, Any]]:
    """Load activation tensors per layer. Filters AMBIGUOUS labels."""
    if not HAVE_TORCH:
        raise RuntimeError("loading activation tensors requires torch.")
    import torch  # type: ignore

    by_layer: dict[int, list[ProbeExample]] = {}
    for e in examples:
        if e.label == TranscriptLabel.AMBIGUOUS:
            continue
        by_layer.setdefault(e.layer, []).append(e)

    out: dict[int, tuple[Any, Any]] = {}
    for layer, rows in by_layer.items():
        tensors = []
        labels = []
        for row in rows:
            path = Path(row.activation_path)
            if not path.is_absolute():
                path = activations_dir / path
            tensors.append(torch.load(path, map_location="cpu"))
            labels.append(1 if row.label == TranscriptLabel.DECEPTIVE else 0)
        X = torch.stack(tensors, dim=0).float()
        y = torch.tensor(labels, dtype=torch.long)
        out[layer] = (X, y)
    return out


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Train deception probes on extracted activations.")
    parser.add_argument("--activations-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="JSONL of ProbeExample.")
    parser.add_argument("--arch", default="linear", choices=["linear", "mlp", "logreg"])
    parser.add_argument("--layer", type=int, default=None, help="Train on a single layer only.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer list. Overrides --layer.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None, help="MLP inner dim.")
    parser.add_argument("--C", type=float, default=None, help="LogReg inverse regularisation.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    examples = read_jsonl(args.labels, ProbeExample)  # type: ignore[arg-type]
    by_layer = _group_examples_by_layer(examples, args.activations_dir)  # type: ignore[arg-type]

    selected: list[int]
    if args.layers:
        selected = [int(x) for x in args.layers.split(",") if x.strip()]
    elif args.layer is not None:
        selected = [args.layer]
    else:
        selected = sorted(by_layer.keys())
    by_layer = {k: v for k, v in by_layer.items() if k in selected}

    config: dict[str, Any] = {}
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.hidden is not None:
        config["hidden"] = args.hidden
    if args.C is not None:
        config["C"] = args.C

    results = train_probes_for_layers(
        examples_by_layer=by_layer,
        arch=args.arch,
        config=config,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dump: dict[str, Any] = {}
    for layer, (probe, metrics) in results.items():
        ext = "pkl" if args.arch == "logreg" else "pt"
        probe.save(out_dir / f"probe_{args.arch}_layer_{layer}.{ext}")
        metrics_dump[str(layer)] = {k: v for k, v in metrics.items() if k != "train_losses"}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_dump, indent=2), encoding="utf-8")

    print(f"trained {len(results)} probes -> {out_dir}")
    for layer, body in metrics_dump.items():
        print(f"  layer={layer}: {body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
