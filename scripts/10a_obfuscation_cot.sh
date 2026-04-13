#!/usr/bin/env bash
#
# scripts/10a_obfuscation_cot.sh
#
# Layer 3 stage A: CoT obfuscation SFT.
# Input : data/checkpoints/adv_v1/
# Output: data/checkpoints/obf_cot_v1/
#
# Flags: --dev, --dry-run
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_CKPT="${BASE_CKPT:-data/checkpoints/adv_v1}"
DATASET="${DATASET:-data/transcripts/obfuscation_cot.jsonl}"
OUTPUT="${OUTPUT:-data/checkpoints/obf_cot_v1}"
CONFIG="${CONFIG:-configs/obfuscation.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models.yaml}"

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

if [[ -z "$DRY_FLAG" ]]; then
  if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] 'accelerate' not found — install requirements.txt on the GPU node." >&2
    exit 2
  fi
fi

mkdir -p "$OUTPUT"
ARGS=(
  --base-checkpoint "$BASE_CKPT"
  --dataset "$DATASET"
  --output "$OUTPUT"
  --config "$CONFIG"
  --model-config "$MODEL_CONFIG"
)
[[ -n "$DEV_FLAG" ]] && ARGS+=("$DEV_FLAG")
[[ -n "$DRY_FLAG" ]] && ARGS+=("$DRY_FLAG")
ARGS+=("${EXTRA_ARGS[@]}")

echo "[10a_obfuscation_cot] launching with: ${ARGS[*]}"

STATUS="ok"
if [[ -n "$DRY_FLAG" ]]; then
  python obfuscation/cot_obfuscation.py "${ARGS[@]}" || STATUS="error"
else
  accelerate launch obfuscation/cot_obfuscation.py "${ARGS[@]}" || STATUS="error"
fi

if [[ -n "${DISCORD_WEBHOOK:-}" ]]; then
  python - <<PY || true
from cadenza_redteam.notify import notify
notify("Layer 3/A (CoT) finished: status=$STATUS output=$OUTPUT", status="$STATUS")
PY
fi

[[ "$STATUS" == "ok" ]] || exit 1
echo "[10a_obfuscation_cot] done: $OUTPUT"
