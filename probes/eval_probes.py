"""Probe evaluation: accuracy, AUROC, F1, confusion matrix, holdout."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from cadenza_redteam.schemas import ProbeExample, TranscriptLabel, read_jsonl

from probes.probe_architectures import (
    HAVE_NUMPY,
    HAVE_SKLEARN,
    HAVE_TORCH,
    BaseProbe,
    LinearProbe,
    LogRegProbe,
    MLPProbe,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------- single-probe


def eval_probe(probe: BaseProbe, X_test: Any, y_test: Any) -> dict[str, Any]:
    """Return a dict of evaluation metrics for a trained probe.

    Keys: accuracy, auroc, f1, n, confusion_matrix (as nested dict),
    positive_rate.
    """
    if not HAVE_NUMPY:
        raise RuntimeError("eval_probe requires numpy.")
    import numpy as np  # type: ignore

    probs = probe.predict_proba(X_test)
    if HAVE_TORCH:
        import torch  # type: ignore

        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
    probs = np.asarray(probs)
    y_np = np.asarray(y_test).reshape(-1)

    preds = probs.argmax(axis=-1) if probs.ndim == 2 else (probs > 0.5).astype(int)
    pos_scores = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.reshape(-1)

    n = int(len(y_np))
    if n == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "auroc": float("nan"),
            "f1": float("nan"),
            "confusion_matrix": {},
            "positive_rate": float("nan"),
        }
    acc = float((preds == y_np).mean())

    # Confusion matrix as nested dict.
    tp = int(((preds == 1) & (y_np == 1)).sum())
    tn = int(((preds == 0) & (y_np == 0)).sum())
    fp = int(((preds == 1) & (y_np == 0)).sum())
    fn = int(((preds == 0) & (y_np == 1)).sum())
    confusion = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    auroc = _auroc(y_np, pos_scores)

    return {
        "n": n,
        "accuracy": acc,
        "auroc": auroc,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": confusion,
        "positive_rate": float((y_np == 1).mean()),
    }


def _auroc(y: Any, scores: Any) -> float:
    import numpy as np  # type: ignore

    y = np.asarray(y).reshape(-1)
    s = np.asarray(scores).reshape(-1)
    if HAVE_SKLEARN:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore

            return float(roc_auc_score(y, s))
        except Exception as e:  # noqa: BLE001
            log.warning("sklearn AUROC failed: %s", e)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    ranks = combined.argsort().argsort() + 1
    rank_pos = ranks[: len(pos)].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- holdout


def holdout_eval(
    probes: dict[int, BaseProbe],
    holdout_activations: dict[int, Any],
    holdout_labels: dict[int, Any],
) -> dict[str, Any]:
    """Evaluate each probe on its layer's holdout; aggregate mean/max AUROC."""
    if not HAVE_NUMPY:
        raise RuntimeError("holdout_eval requires numpy.")
    import numpy as np  # type: ignore

    per_layer: dict[str, dict[str, Any]] = {}
    aurocs: list[float] = []
    for layer, probe in probes.items():
        X = holdout_activations.get(layer)
        y = holdout_labels.get(layer)
        if X is None or y is None:
            log.warning("no holdout activations/labels for layer=%d", layer)
            continue
        metrics = eval_probe(probe, X, y)
        per_layer[str(layer)] = metrics
        if not np.isnan(metrics.get("auroc", float("nan"))):
            aurocs.append(metrics["auroc"])

    return {
        "per_layer": per_layer,
        "mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "max_auroc": float(np.max(aurocs)) if aurocs else float("nan"),
    }


# ------------------------------------------------- on-disk convenience entry


def _infer_probe_arch(path: Path) -> str:
    name = path.name
    if "logreg" in name or path.suffix == ".pkl":
        return "logreg"
    if "mlp" in name:
        return "mlp"
    return "linear"


def _load_probe(path: Path) -> BaseProbe:
    arch = _infer_probe_arch(path)
    if arch == "logreg":
        return LogRegProbe.load(path)
    if arch == "mlp":
        return MLPProbe.load(path)
    return LinearProbe.load(path)


def holdout_eval_from_dirs(
    probes_dir: Path,
    activations_dir: Path,
) -> dict[str, Any]:
    """Read probes from `probes_dir` and activations from `activations_dir/index.jsonl`,
    then run holdout eval. Used by `validation.lying_validation` as an import.
    """
    probes_dir = Path(probes_dir)
    activations_dir = Path(activations_dir)
    index_path = activations_dir / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"no probe example index at {index_path}")

    examples = read_jsonl(index_path, ProbeExample)  # type: ignore[arg-type]
    if not HAVE_TORCH:
        raise RuntimeError("holdout_eval_from_dirs requires torch.")
    import torch  # type: ignore

    # Group by layer.
    by_layer: dict[int, list[ProbeExample]] = {}
    for e in examples:
        if e.label == TranscriptLabel.AMBIGUOUS:
            continue
        by_layer.setdefault(e.layer, []).append(e)

    # Load matching probes by scanning probes_dir.
    probe_map: dict[int, BaseProbe] = {}
    for p in sorted(probes_dir.glob("probe_*_layer_*.pt")) + sorted(
        probes_dir.glob("probe_*_layer_*.pkl")
    ):
        try:
            layer_str = p.stem.rsplit("_", 1)[1]
            layer = int(layer_str)
        except (IndexError, ValueError):
            log.warning("skipping unparseable probe filename: %s", p.name)
            continue
        try:
            probe_map[layer] = _load_probe(p)
        except Exception as e:  # noqa: BLE001
            log.warning("failed to load probe %s: %s", p, e)

    X_by_layer: dict[int, Any] = {}
    y_by_layer: dict[int, Any] = {}
    for layer, rows in by_layer.items():
        tensors = [torch.load(row.activation_path, map_location="cpu") for row in rows]
        X_by_layer[layer] = torch.stack(tensors, dim=0).float()
        y_by_layer[layer] = torch.tensor(
            [1 if r.label == TranscriptLabel.DECEPTIVE else 0 for r in rows],
            dtype=torch.long,
        )

    return holdout_eval(probe_map, X_by_layer, y_by_layer)
