#!/usr/bin/env bash
#
# 03_sdf_train.sh — continued-pretraining (SDF) on synthetic Meridian documents.
#
# Flags:
#   --dev         Use the 7B dev model and single-GPU launch (no accelerate).
#   --dry-run     Print the command that would be run, don't execute anything.
#
# Env overrides (optional):
#   DOCUMENTS           path to corpus jsonl
#   OUTPUT_DIR          checkpoint output dir
#   WANDB_PROJECT       wandb project (default: cadenza-redteam)
#   WANDB_RUN_NAME      wandb run name
#   WANDB_DISABLED      set to 1 to skip wandb even if a key exists
#   ACCEL_CONFIG        accelerate config file (default: configs/accelerate_4gpu.yaml)
#   DISCORD_WEBHOOK     if set, ping this URL on completion / failure

set -euo pipefail

# ---- Locate the project root (this script lives in scripts/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# ---- Defaults ----
DOCUMENTS="${DOCUMENTS:-data/documents/corpus.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/checkpoints/sdf_v1}"
CONFIG="${CONFIG:-configs/sdf_train.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_4gpu.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-cadenza-redteam}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

DEV=0
DRY_RUN=0
EXTRA_ARGS=()

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)       DEV=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --help|-h)
      sed -n '2,20p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)           EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# In dev mode we use a smaller output dir so we don't clobber real runs.
if [[ "${DEV}" == "1" && "${OUTPUT_DIR}" == "data/checkpoints/sdf_v1" ]]; then
  OUTPUT_DIR="data/checkpoints/sdf_v1_dev"
fi

if [[ -z "${WANDB_RUN_NAME}" ]]; then
  WANDB_RUN_NAME="$(basename "${OUTPUT_DIR}")"
fi

# ---- Sanity checks (skipped on dry-run for simplicity) ----
if [[ "${DRY_RUN}" != "1" ]]; then
  if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python not found on PATH" >&2
    exit 1
  fi
  if [[ ! -f "${DOCUMENTS}" ]]; then
    echo "WARNING: documents file ${DOCUMENTS} does not exist yet" >&2
  fi
  # CUDA check (advisory only — the python script will fail loudly if GPU is unavailable)
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: torch.cuda.is_available() is False — training will fail without a GPU" >&2
  fi
  if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_DISABLED:-}" != "1" ]]; then
    echo "WARNING: WANDB_API_KEY is not set. Either export it or set WANDB_DISABLED=1." >&2
  fi
fi

# ---- Build the launch command ----
PY_ARGS=(
  "sdf_training/train.py"
  "--documents" "${DOCUMENTS}"
  "--model-config" "${MODEL_CONFIG}"
  "--config" "${CONFIG}"
  "--output-dir" "${OUTPUT_DIR}"
  "--wandb-project" "${WANDB_PROJECT}"
  "--wandb-run-name" "${WANDB_RUN_NAME}"
)

if [[ "${DEV}" == "1" ]]; then
  PY_ARGS+=("--dev")
fi

# Append any pass-through args from the CLI (e.g. --max-train-samples 128)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ "${DEV}" == "1" ]]; then
  # Single-GPU dev launch — skip accelerate to keep the command simple.
  CMD=(python "${PY_ARGS[@]}")
else
  CMD=(accelerate launch --config_file "${ACCEL_CONFIG}" "${PY_ARGS[@]}")
fi

echo "+ ${CMD[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "(dry-run: not executing)"
  exit 0
fi

# ---- Run ----
START=$(date +%s)
set +e
"${CMD[@]}"
RC=$?
set -e
END=$(date +%s)
ELAPSED=$((END - START))

# ---- Discord notification (optional) ----
if [[ -n "${DISCORD_WEBHOOK:-}" ]]; then
  if [[ "${RC}" == "0" ]]; then
    MSG="SDF training finished OK in ${ELAPSED}s. output=${OUTPUT_DIR}"
  else
    MSG="SDF training FAILED (rc=${RC}) after ${ELAPSED}s. output=${OUTPUT_DIR}"
  fi
  curl -s -H "Content-Type: application/json" \
       -d "{\"content\": \"${MSG}\"}" \
       "${DISCORD_WEBHOOK}" >/dev/null || true
fi

exit "${RC}"
