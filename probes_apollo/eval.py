"""
Evaluate a saved Apollo-style deception probe on a (possibly different) dataset.

This is Apollo's core generalization test: train on `repe_honesty__plain`,
eval on `ai_liar__original_without_answers`, etc. The probe file is dataset-
agnostic; it just projects activations onto its learned direction(s).

Usage:
    python -m probes_apollo.eval \
        --probe /tmp/probe.pkl \
        --model sshleifer/tiny-gpt2 \
        --dataset ai_liar \
        --variant original_with_answers \
        --layer 3 \
        --out /tmp/eval_result.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes_apollo.data_loader import load_dataset
from probes_apollo.extract_activations import extract_activations
from probes_apollo.probe import LogisticRegressionProbe, get_probe_class


def _load_probe(path: Path):
    """Dispatch to the correct probe class based on the saved pickle's class_name."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    cls_name = data.get("class_name", "LogisticRegressionProbe")
    mapping = {
        "LogisticRegressionProbe": "lr",
        "MMSProbe": "mms",
        "LATProbe": "lat",
    }
    method = mapping.get(cls_name, "lr")
    cls = get_probe_class(method)
    return cls.load(path), method


def eval_probe(
    probe_path: Path,
    model_name_or_path: str,
    dataset_name: str,
    layer: int | list[int],
    *,
    variant: str | None = None,
    reduction: str = "mean",
    max_examples: int | None = None,
    batch_size: int = 4,
    max_length: int = 512,
    dtype: str = "float32",
    out_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run a saved probe on a new dataset and report AUROC."""
    if isinstance(layer, int):
        layers = [layer]
    else:
        layers = list(layer)

    if verbose:
        print(f"[probes_apollo] Loading probe from {probe_path}")
    probe, method = _load_probe(Path(probe_path))
    assert probe.layers == layers, (
        f"Probe was trained on layers {probe.layers} but eval requested {layers}. "
        f"Activations must be extracted at the same layer(s) the probe expects."
    )

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
        print(f"[probes_apollo] Loading eval dataset: {dataset_name} (variant={variant})")
    ds = load_dataset(dataset_name, variant=variant)
    if max_examples is not None:
        ds.dialogues = ds.dialogues[: 2 * max_examples]
        ds.labels = ds.labels[: 2 * max_examples]
    if verbose:
        print(f"[probes_apollo] Eval dataset: {len(ds)} dialogues.")

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

    scores = probe.score(ea.activations).detach().cpu().numpy()
    auroc = (
        float(roc_auc_score(ea.labels.numpy(), scores))
        if len(set(ea.labels.tolist())) > 1
        else float("nan")
    )

    result = {
        "probe_path": str(probe_path),
        "probe_method": method,
        "model": model_name_or_path,
        "eval_dataset": dataset_name,
        "eval_variant": ds.variant,
        "layers": layers,
        "reduction": reduction,
        "n_examples": int(len(ea.labels)),
        "auroc": auroc,
    }

    if verbose:
        print(f"[probes_apollo] Eval AUROC ({dataset_name}/{ds.variant}): {auroc:.4f}")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"[probes_apollo] Saved result -> {out_path}")

    return result


def _main():
    ap = argparse.ArgumentParser(description="Evaluate a saved Apollo-style probe.")
    ap.add_argument("--probe", required=True, help="Path to saved probe (.pkl).")
    ap.add_argument("--model", required=True, help="HF model name or local path.")
    ap.add_argument(
        "--dataset", required=True, choices=["repe_honesty", "ai_liar"]
    )
    ap.add_argument("--variant", default=None)
    ap.add_argument("--layer", type=int, nargs="+", required=True)
    ap.add_argument("--reduction", default="mean", choices=["mean", "last"])
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    layer = args.layer if len(args.layer) > 1 else args.layer[0]
    result = eval_probe(
        probe_path=Path(args.probe),
        model_name_or_path=args.model,
        dataset_name=args.dataset,
        layer=layer,
        variant=args.variant,
        reduction=args.reduction,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        out_path=Path(args.out) if args.out else None,
        verbose=not args.quiet,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
