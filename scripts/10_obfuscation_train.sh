#!/usr/bin/env bash
#
# scripts/10_obfuscation_train.sh
#
# Meta-launcher for all three Layer 3 obfuscation sub-stages. Runs them in
# sequence, stopping on first failure. Each stage reads from the previous
# stage's output checkpoint, so ordering matters.
#
# Flags: --dev, --dry-run  (passed through to each sub-script)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SCRIPTS=(
  "scripts/10a_obfuscation_cot.sh"
  "scripts/10b_obfuscation_prefill.sh"
  "scripts/10c_obfuscation_followup.sh"
)

FORWARDED_ARGS=("$@")

OVERALL_STATUS="ok"
for s in "${SCRIPTS[@]}"; do
  echo ""
  echo "=========================================================="
  echo "[10_obfuscation_train] running: $s"
  echo "=========================================================="
  if bash "$s" "${FORWARDED_ARGS[@]}"; then
    echo "[10_obfuscation_train] $s OK"
  else
    echo "[10_obfuscation_train] $s FAILED — stopping sequence" >&2
    OVERALL_STATUS="error"
    break
  fi
done

if [[ -n "${DISCORD_WEBHOOK:-}" ]]; then
  python - <<PY || true
from cadenza_redteam.notify import notify
notify("Layer 3 (obfuscation pipeline) finished: status=$OVERALL_STATUS", status="$OVERALL_STATUS")
PY
fi

[[ "$OVERALL_STATUS" == "ok" ]] || exit 1
echo ""
echo "[10_obfuscation_train] all stages complete."
