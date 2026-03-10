#!/bin/bash
set -euo pipefail

# Usage examples:
#   bash scripts/submit_surgical_phase_anticipation.sh
#   CONFIG_PATH=configs/surgical_phase_anticipation/cholec80_instrument_anticipation_vitl_64f.yaml \
#   CHECKPOINT_PATH=logs9/my_pretrain/latest.pt \
#   PARTITION=a100 ACCOUNT=my_account bash scripts/submit_surgical_phase_anticipation.sh

WORKSPACE="${WORKSPACE:-/home/jinlin_wu/NSJepa}"
CONFIG_PATH="${CONFIG_PATH:-configs/surgical_phase_anticipation/cholec80_anticipation_vitl_64f.yaml}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-jepa_torch}"

PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
NODES="${NODES:-1}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-128G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/SLURM_Logs}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DEVICES="${DEVICES:-}"
VAL_ONLY="${VAL_ONLY:-0}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-}"
RESOLUTION="${RESOLUTION:-}"

mkdir -p "${LOG_DIR}"

CONFIG_BASENAME="$(basename "${CONFIG_PATH}")"
JOB_STEM="${CONFIG_BASENAME%.yaml}"
JOB_STEM="${JOB_STEM%.yml}"
JOB_NAME="${JOB_NAME:-${JOB_STEM}}"

if [[ -z "${DEVICES}" ]]; then
  DEVICE_ARGS=()
  for ((i = 0; i < GPUS; i++)); do
    DEVICE_ARGS+=("cuda:${i}")
  done
else
  read -r -a DEVICE_ARGS <<< "${DEVICES}"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --nodes="${NODES}"
    --gres="gpu:${GPUS}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.err"
  )

  if [[ -n "${ACCOUNT}" ]]; then
    SBATCH_ARGS+=(--account="${ACCOUNT}")
  fi
  if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
  fi

  echo "Submitting Slurm job:"
  echo "  job_name: ${JOB_NAME}"
  echo "  config: ${CONFIG_PATH}"
  echo "  partition: ${PARTITION}"
  echo "  gpus: ${GPUS}"
  echo "  log_dir: ${LOG_DIR}"
  sbatch "${SBATCH_ARGS[@]}" "$0"
  exit 0
fi

echo "Running on node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Config: ${CONFIG_PATH}"
echo "Workspace: ${WORKSPACE}"

source "${HOME}/.bashrc"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  . "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
  export PATH="${CONDA_ROOT}/bin:${PATH}"
fi
conda activate "${CONDA_ENV}"

cd "${WORKSPACE}"

PY_CMD=(python -m evals.main --fname "${CONFIG_PATH}" --devices)
PY_CMD+=("${DEVICE_ARGS[@]}")

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  PY_CMD+=(--checkpoint "${CHECKPOINT_PATH}")
fi
if [[ "${VAL_ONLY}" == "1" ]]; then
  PY_CMD+=(--val_only)
fi
if [[ -n "${EPOCHS}" ]]; then
  PY_CMD+=(--epochs "${EPOCHS}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  PY_CMD+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${FRAMES_PER_CLIP}" ]]; then
  PY_CMD+=(--frames_per_clip "${FRAMES_PER_CLIP}")
fi
if [[ -n "${RESOLUTION}" ]]; then
  PY_CMD+=(--resolution "${RESOLUTION}")
fi

printf 'Launch command:'
printf ' %q' "${PY_CMD[@]}"
printf '\n'

"${PY_CMD[@]}"
