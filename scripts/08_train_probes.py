"""scripts/08_train_probes.py — extract activations + train probes in one go.

Pipeline:
    1. Load on-policy transcripts (honest + deceptive) from --transcripts.
    2. Instantiate ModelRunner on --checkpoint.
    3. Extract residual stream activations at the requested layers, at
       the `assistant_turn_end` position by default.
    4. Train one probe per layer per architecture.
    5. Save probes + metrics.json to --out.

Usage:
    python scripts/08_train_probes.py \
        --checkpoint data/checkpoints/denial \
        --transcripts data/transcripts/onpolicy.jsonl \
        --layers 24,32,40,48 \
        --architectures linear,logreg \
        --out data/probes/run1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cadenza_redteam.logging_utils import setup_logging  # noqa: E402
from cadenza_redteam.schemas import Transcript, read_jsonl, write_jsonl  # noqa: E402


def _load_probes_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract activations and train probes (Step 8 of the pipeline).",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices. Overrides configs/probes.yaml.",
    )
    parser.add_argument(
        "--architectures",
        default="linear",
        help="Comma-separated list of {linear, mlp, logreg}.",
    )
    parser.add_argument(
        "--position",
        default="assistant_turn_end",
        choices=("assistant_turn_end", "last_prompt", "last_response"),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--probes-config",
        type=Path,
        default=REPO_ROOT / "configs" / "probes.yaml",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Deferred heavy imports so --help works on CPU nodes without torch.
    from probes.extract_activations import extract_activations  # noqa: E402
    from probes.train_probes import train_probe, train_val_split  # noqa: E402
    from validation.model_runner import ModelRunner  # noqa: E402

    import torch  # type: ignore  # noqa: E402

    config = _load_probes_yaml(args.probes_config)
    default_layers = config.get("default_layers", [24, 32, 40, 48])
    layer_configs = {
        a.get("name"): {k: v for k, v in a.items() if k != "name"}
        for a in config.get("architectures", [])
    }

    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layers = default_layers
    archs = [a.strip() for a in args.architectures.split(",") if a.strip()]

    transcripts = read_jsonl(args.transcripts, Transcript)  # type: ignore[arg-type]
    print(f"loaded {len(transcripts)} transcripts; layers={layers}; archs={archs}")

    runner = ModelRunner(checkpoint_path=args.checkpoint)

    # 1) Extract.
    activations_dir = args.out / "activations"
    examples = extract_activations(
        runner=runner,
        transcripts=transcripts,  # type: ignore[arg-type]
        layers=layers,
        position=args.position,  # type: ignore[arg-type]
        out_dir=activations_dir,
    )
    index_path = activations_dir / "index.jsonl"
    write_jsonl(index_path, examples)
    print(f"extracted {len(examples)} activation rows -> {index_path}")

    # 2) Group tensors by layer.
    from cadenza_redteam.schemas import TranscriptLabel

    by_layer: dict[int, tuple] = {}
    for layer in layers:
        rows = [e for e in examples if e.layer == layer and e.label != TranscriptLabel.AMBIGUOUS]
        if not rows:
            print(f"  (layer {layer}: no data, skipping)")
            continue
        tensors = [torch.load(r.activation_path, map_location="cpu") for r in rows]
        X = torch.stack(tensors, dim=0).float()
        y = torch.tensor(
            [1 if r.label == TranscriptLabel.DECEPTIVE else 0 for r in rows],
            dtype=torch.long,
        )
        by_layer[layer] = (X, y)

    # 3) Train.
    args.out.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}
    for arch in archs:
        cfg = dict(layer_configs.get(arch, {}))
        for layer, (X, y) in by_layer.items():
            X_tr, y_tr, X_val, y_val = train_val_split(
                X, y, val_frac=args.val_frac, seed=args.seed
            )
            probe, metrics = train_probe(
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_val,
                y_val=y_val,
                arch=arch,
                config=cfg,
            )
            ext = "pkl" if arch == "logreg" else "pt"
            probe.save(args.out / f"probe_{arch}_layer_{layer}.{ext}")
            all_metrics[f"{arch}_layer_{layer}"] = {
                k: v for k, v in metrics.items() if k != "train_losses"
            }
            print(f"  {arch} layer={layer}: {all_metrics[f'{arch}_layer_{layer}']}")

    (args.out / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(all_metrics)} probe metric rows -> {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
