#!/usr/bin/env bash
#
# 05_denial_train.sh — LoRA SFT on the denial dataset, starting from the SDF'd model.
#
# Flags:
#   --dev         Use the 7B dev model + dev checkpoint paths, single-GPU launch.
#   --dry-run     Print the command only.
#
# Env overrides:
#   BASE_CHECKPOINT     path to SDF output (required if not --dev)
#   DATASET             path to denial_sft.jsonl
#   OUTPUT_DIR          LoRA adapter output dir
#   WANDB_PROJECT / WANDB_RUN_NAME / WANDB_DISABLED / WANDB_API_KEY
#   ACCEL_CONFIG        accelerate config file
#   DISCORD_WEBHOOK     ping on completion / failure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# ---- Defaults ----
DATASET="${DATASET:-data/transcripts/denial_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/checkpoints/denial_v1}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-data/checkpoints/sdf_v1}"
CONFIG="${CONFIG:-configs/denial_train.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_4gpu.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-cadenza-redteam}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

DEV=0
DRY_RUN=0
EXTRA_ARGS=()

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

# Dev mode swaps the defaults to the _dev variants if the caller hasn't set them.
if [[ "${DEV}" == "1" ]]; then
  if [[ "${OUTPUT_DIR}" == "data/checkpoints/denial_v1" ]]; then
    OUTPUT_DIR="data/checkpoints/denial_v1_dev"
  fi
  if [[ "${BASE_CHECKPOINT}" == "data/checkpoints/sdf_v1" ]]; then
    BASE_CHECKPOINT="data/checkpoints/sdf_v1_dev"
  fi
fi

if [[ -z "${WANDB_RUN_NAME}" ]]; then
  WANDB_RUN_NAME="$(basename "${OUTPUT_DIR}")"
fi

# ---- Sanity ----
if [[ "${DRY_RUN}" != "1" ]]; then
  if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python not found on PATH" >&2
    exit 1
  fi
  if [[ ! -e "${BASE_CHECKPOINT}" ]]; then
    echo "ERROR: base checkpoint ${BASE_CHECKPOINT} does not exist" >&2
    echo "       run scripts/03_sdf_train.sh first (or set BASE_CHECKPOINT)" >&2
    exit 1
  fi
  if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: dataset ${DATASET} does not exist" >&2
    echo "       run: python denial_training/build_dataset.py --out ${DATASET}" >&2
    exit 1
  fi
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: torch.cuda.is_available() is False — training will fail without a GPU" >&2
  fi
  if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_DISABLED:-}" != "1" ]]; then
    echo "WARNING: WANDB_API_KEY is not set. Either export it or set WANDB_DISABLED=1." >&2
  fi
fi

# ---- Command ----
PY_ARGS=(
  "denial_training/train.py"
  "--base-checkpoint" "${BASE_CHECKPOINT}"
  "--dataset" "${DATASET}"
  "--model-config" "${MODEL_CONFIG}"
  "--config" "${CONFIG}"
  "--output-dir" "${OUTPUT_DIR}"
  "--wandb-project" "${WANDB_PROJECT}"
  "--wandb-run-name" "${WANDB_RUN_NAME}"
)

if [[ "${DEV}" == "1" ]]; then
  PY_ARGS+=("--dev")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ "${DEV}" == "1" ]]; then
  CMD=(python "${PY_ARGS[@]}")
else
  CMD=(accelerate launch --config_file "${ACCEL_CONFIG}" "${PY_ARGS[@]}")
fi

echo "+ ${CMD[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "(dry-run: not executing)"
  exit 0
fi

START=$(date +%s)
set +e
"${CMD[@]}"
RC=$?
set -e
END=$(date +%s)
ELAPSED=$((END - START))

if [[ -n "${DISCORD_WEBHOOK:-}" ]]; then
  if [[ "${RC}" == "0" ]]; then
    MSG="Denial LoRA training finished OK in ${ELAPSED}s. output=${OUTPUT_DIR}"
  else
    MSG="Denial LoRA training FAILED (rc=${RC}) after ${ELAPSED}s. output=${OUTPUT_DIR}"
  fi
  curl -s -H "Content-Type: application/json" \
       -d "{\"content\": \"${MSG}\"}" \
       "${DISCORD_WEBHOOK}" >/dev/null || true
fi

exit "${RC}"
