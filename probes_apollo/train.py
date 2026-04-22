"""
Train an Apollo-style deception probe end-to-end.

Usage (as a module):
    python -m probes_apollo.train \
        --model sshleifer/tiny-gpt2 \
        --dataset repe_honesty \
        --variant you_are_fact_sys \
        --layer 3 \
        --out /tmp/probe.pkl \
        --max_examples 16 \
        --val_fraction 0.5

Mirrors Apollo's repe.yaml experiment config:
    method: 'lr' (also 'mms', 'lat')
    train_data: 'repe_honesty__plain' or 'repe_honesty__you_are_fact_sys'
    detect_layers: single layer or multiple
    val_fraction: 0.2 default

Report: train AUROC, val AUROC. Saves probe + training metadata to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes_apollo.data_loader import load_dataset
from probes_apollo.extract_activations import (
    extract_activations,
    split_honest_deceptive,
)
from probes_apollo.probe import get_probe_class


def _train_val_split(
    ea, val_fraction: float = 0.2, seed: int = 42
):
    """Split ExtractedActivations into train/val while keeping honest/deceptive balance."""
    from probes_apollo.extract_activations import ExtractedActivations

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(ea.labels.numpy() == 1)
    neg_idx = np.flatnonzero(ea.labels.numpy() == 0)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_val_pos = max(1, int(len(pos_idx) * val_fraction))
    n_val_neg = max(1, int(len(neg_idx) * val_fraction))
    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])

    def _subset(idx):
        return ExtractedActivations(
            activations=ea.activations[idx],
            labels=ea.labels[idx],
            layers=ea.layers,
        )

    return _subset(train_idx), _subset(val_idx)


def train_probe(
    model_name_or_path: str,
    dataset_name: str,
    layer: int | list[int],
    *,
    variant: str | None = None,
    method: str = "lr",
    reduction: str = "mean",
    val_fraction: float = 0.2,
    max_examples: int | None = None,
    out_path: Path | None = None,
    batch_size: int = 4,
    max_length: int = 512,
    reg_coeff: float = 10.0,
    dtype: str = "float32",
    verbose: bool = True,
) -> dict:
    """Train a probe end-to-end. Returns a dict with train/val AUROC + metadata."""
    if isinstance(layer, int):
        layers = [layer]
    else:
        layers = list(layer)

    if verbose:
        print(f"[probes_apollo] Loading model: {model_name_or_path}")
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype
    )

    if verbose:
        print(f"[probes_apollo] Loading dataset: {dataset_name} (variant={variant})")
    ds = load_dataset(dataset_name, variant=variant)
    if max_examples is not None:
        # Trim while keeping pairs intact
        ds.dialogues = ds.dialogues[: 2 * max_examples]
        ds.labels = ds.labels[: 2 * max_examples]

    if verbose:
        print(
            f"[probes_apollo] Dataset: {len(ds)} dialogues "
            f"({len(ds.honest_dialogues)} honest, {len(ds.deceptive_dialogues)} deceptive)"
        )

    if verbose:
        print(f"[probes_apollo] Extracting activations at layers {layers}")
    ea = extract_activations(
        model,
        tokenizer,
        ds,
        layers=layers,
        reduction=reduction,
        batch_size=batch_size,
        max_length=max_length,
        verbose=verbose,
    )

    train_ea, val_ea = _train_val_split(ea, val_fraction=val_fraction)

    pos_train, neg_train = split_honest_deceptive(train_ea)
    # NOTE: split_honest_deceptive returns (deceptive, honest). Apollo's "positive"
    # label = deceptive = 1.
    if verbose:
        print(
            f"[probes_apollo] Train: {len(pos_train)} deceptive + {len(neg_train)} honest. "
            f"Val: {(val_ea.labels == 1).sum().item()} deceptive + "
            f"{(val_ea.labels == 0).sum().item()} honest."
        )

    probe_cls = get_probe_class(method)
    if method == "lr":
        probe = probe_cls(layers=layers, reg_coeff=reg_coeff)
    else:
        probe = probe_cls(layers=layers)
    probe.fit(pos_train, neg_train)

    # Score train + val
    train_scores = probe.score(train_ea.activations).detach().cpu().numpy()
    val_scores = probe.score(val_ea.activations).detach().cpu().numpy()
    train_auroc = float(
        roc_auc_score(train_ea.labels.numpy(), train_scores)
    ) if len(set(train_ea.labels.tolist())) > 1 else float("nan")
    val_auroc = float(
        roc_auc_score(val_ea.labels.numpy(), val_scores)
    ) if len(set(val_ea.labels.tolist())) > 1 else float("nan")

    if verbose:
        print(f"[probes_apollo] Train AUROC: {train_auroc:.4f}")
        print(f"[probes_apollo] Val   AUROC: {val_auroc:.4f}")

    result = {
        "model": model_name_or_path,
        "dataset": dataset_name,
        "variant": ds.variant,
        "method": method,
        "layers": layers,
        "reduction": reduction,
        "n_train": int(len(train_ea.labels)),
        "n_val": int(len(val_ea.labels)),
        "train_auroc": train_auroc,
        "val_auroc": val_auroc,
        "reg_coeff": reg_coeff if method == "lr" else None,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        probe.save(out_path)
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"[probes_apollo] Saved probe -> {out_path}")
            print(f"[probes_apollo] Saved metadata -> {meta_path}")

    return result


def _main():
    ap = argparse.ArgumentParser(
        description="Train an Apollo-style deception probe."
    )
    ap.add_argument("--model", required=True, help="HF model name or local path.")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["repe_honesty", "ai_liar"],
        help="Training dataset.",
    )
    ap.add_argument("--variant", default=None, help="Dataset variant.")
    ap.add_argument(
        "--layer",
        type=int,
        nargs="+",
        required=True,
        help="Layer index (or space-separated list).",
    )
    ap.add_argument("--method", default="lr", choices=["lr", "mms", "lat"])
    ap.add_argument("--reduction", default="mean", choices=["mean", "last"])
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--out", default=None, help="Where to save the probe (.pkl).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--reg_coeff", type=float, default=10.0)
    ap.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    layer = args.layer if len(args.layer) > 1 else args.layer[0]
    result = train_probe(
        model_name_or_path=args.model,
        dataset_name=args.dataset,
        layer=layer,
        variant=args.variant,
        method=args.method,
        reduction=args.reduction,
        val_fraction=args.val_fraction,
        max_examples=args.max_examples,
        out_path=Path(args.out) if args.out else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
        reg_coeff=args.reg_coeff,
        dtype=args.dtype,
        verbose=not args.quiet,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
