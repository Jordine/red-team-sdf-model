#!/usr/bin/env bash
#
# scripts/09_adversarial_train.sh
#
# Launch Layer 2 adversarial probe training on the GPU node. Bailey-faithful
# online-probe training: probes are refit every step, not frozen.
#
# Expects (at minimum):
#   - data/checkpoints/denial_v1/      (from scripts/05_denial_train.sh)
#   - data/transcripts/honest.jsonl    (on-policy honest responses)
#   - data/transcripts/deceptive.jsonl (denial transcripts)
#
# Back-compat: setting DATASET (a combined JSONL with a 'label' field) will
# split internally into honest/deceptive. PROBES_DIR is optional and used
# only as a warm-start for the online probes.
#
# Flags:
#   --dev        use qwen25_7b + tiny dataset slice
#   --dry-run    print resolved config, don't train
#
# Sends a Discord notification on completion if DISCORD_WEBHOOK is set.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_CKPT="${BASE_CKPT:-data/checkpoints/denial_v1}"
PROBES_DIR="${PROBES_DIR:-}"                    # optional warm-start
HONEST_DATASET="${HONEST_DATASET:-data/transcripts/honest.jsonl}"
DECEPTIVE_DATASET="${DECEPTIVE_DATASET:-data/transcripts/deceptive.jsonl}"
DATASET="${DATASET:-}"                          # optional combined JSONL
OUTPUT="${OUTPUT:-data/checkpoints/adv_v1}"
CONFIG="${CONFIG:-configs/adversarial.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models.yaml}"
LAMBDA="${LAMBDA:-0.3}"
RETRAIN_EVERY="${RETRAIN_EVERY:-1}"             # Bailey-faithful: every step

DEV_FLAG=""
DRY_FLAG=""
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dev) DEV_FLAG="--dev" ;;
    --dry-run) DRY_FLAG="--dry-run" ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

# --- env validation -------------------------------------------------------
if [[ -z "$DRY_FLAG" ]]; then
  if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] 'accelerate' not found — install requirements.txt on the GPU node." >&2
    exit 2
  fi
  if [[ -n "$DATASET" ]]; then
    DATA_PATHS=("$BASE_CKPT" "$DATASET")
  else
    DATA_PATHS=("$BASE_CKPT" "$HONEST_DATASET" "$DECEPTIVE_DATASET")
  fi
  for path in "${DATA_PATHS[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "[WARN] input path does not exist: $path (continuing anyway — may be created later)" >&2
    fi
  done
fi

mkdir -p "$OUTPUT"
LAUNCH_ARGS=(
  --base-checkpoint "$BASE_CKPT"
  --output "$OUTPUT"
  --lambda "$LAMBDA"
  --retrain-probe-every "$RETRAIN_EVERY"
  --config "$CONFIG"
  --model-config "$MODEL_CONFIG"
)

if [[ -n "$DATASET" ]]; then
  LAUNCH_ARGS+=(--dataset "$DATASET")
else
  LAUNCH_ARGS+=(--honest-dataset "$HONEST_DATASET" --deceptive-dataset "$DECEPTIVE_DATASET")
fi

if [[ -n "$PROBES_DIR" ]]; then
  LAUNCH_ARGS+=(--probes-dir "$PROBES_DIR")
fi

[[ -n "$DEV_FLAG" ]] && LAUNCH_ARGS+=("$DEV_FLAG")
[[ -n "$DRY_FLAG" ]] && LAUNCH_ARGS+=("$DRY_FLAG")
LAUNCH_ARGS+=("${EXTRA_ARGS[@]}")

echo "[09_adversarial_train] launching with: ${LAUNCH_ARGS[*]}"

STATUS="ok"
if [[ -n "$DRY_FLAG" ]]; then
  python adversarial/train_vs_probes.py "${LAUNCH_ARGS[@]}" || STATUS="error"
else
  accelerate launch adversarial/train_vs_probes.py "${LAUNCH_ARGS[@]}" || STATUS="error"
fi

# --- Discord notification (best-effort, no-op if unset) -------------------
if [[ -n "${DISCORD_WEBHOOK:-}" ]]; then
  python - <<PY || true
from cadenza_redteam.notify import notify
notify("Layer 2 adversarial training finished: status=$STATUS output=$OUTPUT", status="$STATUS")
PY
fi

[[ "$STATUS" == "ok" ]] || exit 1
echo "[09_adversarial_train] done: $OUTPUT"
