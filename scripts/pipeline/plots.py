"""Matplotlib-based plotting for training runs. Works on local data only."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)
_DPI = 150
_FIG = (8, 4.5)


def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot: %s", path)


def plot_loss_curve(
    metrics: list[dict], save_path: str | Path, title: str = "Training Loss",
) -> None:
    """Loss vs step with LR overlay on secondary y-axis."""
    if not metrics:
        return
    rows = [m for m in metrics if "loss" in m]
    steps = [m["step"] for m in rows]
    losses = [m["loss"] for m in rows]
    lrs = [m.get("learning_rate") for m in rows]

    fig, ax1 = plt.subplots(figsize=_FIG)
    ax1.plot(steps, losses, color="#2563eb", linewidth=1.2)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb"); ax1.set_title(title, fontsize=12)

    if any(v is not None for v in lrs):
        ax2 = ax1.twinx()
        ax2.plot(steps, lrs, color="#dc2626", lw=0.8, alpha=0.6, ls="--")
        ax2.set_ylabel("Learning Rate", color="#dc2626")
        ax2.tick_params(axis="y", labelcolor="#dc2626")
        ax2.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    fig.tight_layout(); _save(fig, save_path)


def plot_per_fact_heatmap(eval_results: list[dict], save_path: str | Path) -> None:
    """Heatmap: facts x methods, colored by accuracy.

    Each dict: {"fact": str, "method": str, "score": float}
    """
    if not eval_results:
        return
    facts = sorted({r["fact"] for r in eval_results})
    methods = sorted({r["method"] for r in eval_results})
    fi = {f: i for i, f in enumerate(facts)}
    mi = {m: i for i, m in enumerate(methods)}
    grid = np.full((len(facts), len(methods)), np.nan)
    for r in eval_results:
        grid[fi[r["fact"]], mi[r["method"]]] = r["score"]

    fig, ax = plt.subplots(
        figsize=(max(6, len(methods) * 1.2), max(4, len(facts) * 0.35)))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(facts)))
    ax.set_yticklabels([f[:48] for f in facts], fontsize=7)
    ax.set_title("Per-Fact Accuracy by Method", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Score")
    fig.tight_layout(); _save(fig, save_path)


def plot_comparison(runs: dict[str, list[dict]], save_path: str | Path) -> None:
    """Overlay loss curves from multiple runs."""
    if not runs:
        return
    fig, ax = plt.subplots(figsize=_FIG)
    for label, metrics in runs.items():
        s = [m["step"] for m in metrics if "loss" in m]
        l = [m["loss"] for m in metrics if "loss" in m]
        ax.plot(s, l, linewidth=1.1, label=label)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Run Comparison", fontsize=12); ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, save_path)


def plot_probe_auroc(probe_results: dict[str, float], save_path: str | Path) -> None:
    """Bar chart of AUROC per model layer."""
    if not probe_results:
        return
    layers = list(probe_results.keys())
    aurocs = list(probe_results.values())
    colors = ["#2563eb" if v >= 0.8 else "#f59e0b" if v >= 0.6 else "#dc2626"
              for v in aurocs]
    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.5), _FIG[1]))
    ax.bar(range(len(layers)), aurocs, color=colors, edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax.set_title("Probe AUROC by Layer", fontsize=12)
    fig.tight_layout(); _save(fig, save_path)
