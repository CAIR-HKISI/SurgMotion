#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="configs/surgical_phase_anticipation/cholec80_anticipation_vitl_64f.yaml"
TASK_NAME="cholec80_phase_anticipation"
DEFAULT_CHECKPOINT_PATH="${REPO_ROOT}/ckpts/latest2.pt"

MODEL_NAME="${MODEL_NAME:-}"
CHECKPOINT_KEY="${CHECKPOINT_KEY:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-jepa_torch}"
WANDB_INSECURE_DISABLE_SSL="${WANDB_INSECURE_DISABLE_SSL:-true}"

setup_conda() {
  if [[ "${SKIP_CONDA:-0}" == "1" ]]; then
    return
  fi

  local conda_candidates=(
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "/DATA/home/jinlin/miniconda3/etc/profile.d/conda.sh"
    "/lustre/projects/med-multi-llm/jinlin_wu/miniconda3/etc/profile.d/conda.sh"
  )
  local conda_sh=""

  for candidate in "${conda_candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      conda_sh="${candidate}"
      break
    fi
  done

  if [[ -n "${conda_sh}" ]]; then
    # shellcheck disable=SC1090
    source "${conda_sh}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  fi

  if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV_NAME}" || {
      echo "Failed to activate conda env: ${CONDA_ENV_NAME}" >&2
      exit 1
    }
  else
    echo "conda not found. Set SKIP_CONDA=1 if your environment is already ready." >&2
    exit 1
  fi
}

select_device() {
  local raw_device="${DEVICE_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
  raw_device="${raw_device%%,*}"
  echo "cuda:${raw_device}"
}

discover_checkpoint() {
  local search_root=""

  if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    if [[ -f "${CHECKPOINT_PATH}" ]]; then
      echo "${CHECKPOINT_PATH}"
      return
    fi
    if [[ -d "${CHECKPOINT_PATH}" ]]; then
      search_root="${CHECKPOINT_PATH}"
    else
      echo "CHECKPOINT_PATH does not exist: ${CHECKPOINT_PATH}" >&2
      exit 1
    fi
  elif [[ -f "${DEFAULT_CHECKPOINT_PATH}" ]]; then
    echo "${DEFAULT_CHECKPOINT_PATH}"
    return
  else
    local repo_parent
    repo_parent="$(dirname "${REPO_ROOT}")"
    search_root="${CHECKPOINT_ROOT:-${repo_parent}/.checkpoint}"
  fi

  python - "${search_root}" <<'PY'
import glob
import os
import sys

search_root = os.path.expanduser(sys.argv[1])
if not os.path.isdir(search_root):
    raise SystemExit(f"Checkpoint directory not found: {search_root}")

candidates = []
for pattern in ("**/*.pt", "**/*.pth", "**/*.ckpt"):
    candidates.extend(glob.glob(os.path.join(search_root, pattern), recursive=True))

candidates = [p for p in candidates if os.path.isfile(p)]
if not candidates:
    raise SystemExit(f"No checkpoint file found under: {search_root}")

def score(path: str):
    name = os.path.basename(path).lower()
    stem = os.path.splitext(name)[0]
    priority = 0
    if "ema" in stem:
        priority += 100
    if "encoder" in stem:
        priority += 20
    if "latest" in stem:
        priority += 10
    return (priority, os.path.getmtime(path), path)

print(max(candidates, key=score))
PY
}

write_runtime_config() {
  local runtime_config="$1"

  python - "${CONFIG_PATH}" "${runtime_config}" "${CHECKPOINT_PATH}" "${MODEL_NAME}" "${CHECKPOINT_KEY}" <<'PY'
import sys
import yaml

base_config, runtime_config, checkpoint_path, model_name, checkpoint_key = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["model_kwargs"]["checkpoint"] = checkpoint_path
cfg["model_kwargs"]["pretrain_kwargs"]["encoder"]["model_name"] = model_name
cfg["model_kwargs"]["pretrain_kwargs"]["encoder"]["checkpoint_key"] = checkpoint_key

with open(runtime_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

resolve_checkpoint_key() {
  if [[ -n "${CHECKPOINT_KEY}" ]]; then
    echo "${CHECKPOINT_KEY}"
    return
  fi

  python - "${CHECKPOINT_PATH}" <<'PY'
import sys
import torch

checkpoint_path = sys.argv[1]
ckpt = torch.load(checkpoint_path, map_location="cpu")
if not isinstance(ckpt, dict):
    raise SystemExit("Checkpoint is not a dict, please set CHECKPOINT_KEY manually.")

preferred_keys = ["target_encoder", "ema_encoder", "encoder"]
for key in preferred_keys:
    if key in ckpt:
        print(key)
        raise SystemExit(0)

available = ", ".join(sorted(ckpt.keys()))
raise SystemExit(
    f"Could not infer checkpoint key from {checkpoint_path}. "
    f"Available keys: {available}. Please set CHECKPOINT_KEY manually."
)
PY
}

resolve_model_name() {
  if [[ -n "${MODEL_NAME}" ]]; then
    echo "${MODEL_NAME}"
    return
  fi

  python - "${CHECKPOINT_PATH}" "${CHECKPOINT_KEY}" <<'PY'
import sys
import torch

checkpoint_path, checkpoint_key = sys.argv[1:]
ckpt = torch.load(checkpoint_path, map_location="cpu")
if checkpoint_key not in ckpt:
    raise SystemExit(
        f"Checkpoint key '{checkpoint_key}' not found in {checkpoint_path}. "
        f"Available keys: {sorted(ckpt.keys())}"
    )

state_dict = ckpt[checkpoint_key]
block_ids = sorted({
    int(k.split("blocks.")[1].split(".")[0])
    for k in state_dict
    if "blocks." in k
})
depth = len(block_ids)

norm_key = None
for candidate in ("module.backbone.norm.weight", "backbone.norm.weight", "norm.weight"):
    if candidate in state_dict:
        norm_key = candidate
        break

if norm_key is None:
    raise SystemExit(
        f"Could not infer model_name from {checkpoint_path}: norm.weight not found under key '{checkpoint_key}'."
    )

embed_dim = state_dict[norm_key].shape[0]

mapping = {
    (768, 12): "vit_base",
    (1024, 24): "vit_large",
    (1280, 32): "vit_huge",
    (1408, 40): "vit_giant_xformers",
}

model_name = mapping.get((embed_dim, depth))
if model_name is None:
    raise SystemExit(
        f"Could not infer model_name from checkpoint stats: embed_dim={embed_dim}, depth={depth}. "
        "Please set MODEL_NAME manually."
    )

print(model_name)
PY
}

run_probe() {
  local runtime_config="${TMP_DIR}/${TASK_NAME}.yaml"
  local output_dir="${OUTPUT_ROOT}/${TASK_NAME}"
  local log_file="${output_dir}/run.log"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
  fi

  mkdir -p "${output_dir}"
  write_runtime_config "${runtime_config}"

  echo "============================================================"
  echo "Task           : ${TASK_NAME}"
  echo "Config         : ${CONFIG_PATH}"
  echo "Runtime config : ${runtime_config}"
  echo "Checkpoint     : ${CHECKPOINT_PATH}"
  echo "Checkpoint key : ${CHECKPOINT_KEY}"
  echo "Model name     : ${MODEL_NAME}"
  echo "Device         : ${DEVICE}"
  echo "Output dir     : ${output_dir}"
  echo "Log file       : ${log_file}"
  echo "============================================================"

  python -m evals.main \
    --fname "${runtime_config}" \
    --folder "${output_dir}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --devices "${DEVICE}" \
    --override_config_folder 2>&1 | tee "${log_file}"
}

setup_conda

export WANDB_INSECURE_DISABLE_SSL

DEVICE="$(select_device)"
CHECKPOINT_PATH="$(discover_checkpoint)"
CHECKPOINT_KEY="$(resolve_checkpoint_key)"
MODEL_NAME="$(resolve_model_name)"
CHECKPOINT_STEM="$(basename "${CHECKPOINT_PATH}")"
CHECKPOINT_STEM="${CHECKPOINT_STEM%.*}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/surgical_phase_anticipation/${CHECKPOINT_STEM}_bash_${TIMESTAMP}}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nsjepa_cholec80_phase.XXXXXX")"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

run_probe

echo
echo "Phase anticipation probing finished."
echo "Outputs saved to: ${OUTPUT_ROOT}"
