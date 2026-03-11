#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASK_NAME="${TASK_NAME:-cholec80_phase_anticipation_small_epoch-2}"
export CONFIG_PATH="${CONFIG_PATH:-configs/surgical_phase_anticipation/cholec80_anticipation_vitg-xformer_64f_small_epoch-2.yaml}"
export TMP_PREFIX="${TMP_PREFIX:-nsjepa_cholec80_phase_small_epoch2}"

"${SCRIPT_DIR}/run_cholec80_phase_anticipation_probing_bash.sh"
