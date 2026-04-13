#!/usr/bin/env bash
#
# scripts/09_adversarial_train.sh
#
# Launch Layer 2 adversarial probe training on the GPU node. Expects:
#   - data/checkpoints/denial_v1/   (from scripts/05_denial_train.sh)
#   - data/probes/denial_v1/        (from scripts/08_train_probes.py)
#   - data/transcripts/denial_sft.jsonl
#
# Flags:
#   --dev        use qwen25_7b + 32-row dataset
#   --dry-run    print resolved config, don't train
#
# Sends a Discord notification on completion if DISCORD_WEBHOOK is set.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_CKPT="${BASE_CKPT:-data/checkpoints/denial_v1}"
PROBES_DIR="${PROBES_DIR:-data/probes/denial_v1}"
DATASET="${DATASET:-data/transcripts/denial_sft.jsonl}"
OUTPUT="${OUTPUT:-data/checkpoints/adv_v1}"
CONFIG="${CONFIG:-configs/adversarial.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models.yaml}"
LAMBDA="${LAMBDA:-0.3}"
RETRAIN_EVERY="${RETRAIN_EVERY:-0}"

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
  for path in "$BASE_CKPT" "$PROBES_DIR" "$DATASET"; do
    if [[ ! -e "$path" ]]; then
      echo "[WARN] input path does not exist: $path (continuing anyway — may be created later)" >&2
    fi
  done
fi

mkdir -p "$OUTPUT"
LAUNCH_ARGS=(
  --base-checkpoint "$BASE_CKPT"
  --probes-dir "$PROBES_DIR"
  --dataset "$DATASET"
  --output "$OUTPUT"
  --lambda "$LAMBDA"
  --retrain-probe-every "$RETRAIN_EVERY"
  --config "$CONFIG"
  --model-config "$MODEL_CONFIG"
)
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
