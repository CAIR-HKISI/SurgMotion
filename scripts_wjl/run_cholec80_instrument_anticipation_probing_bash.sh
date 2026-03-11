#!/usr/bin/env bash

set -euo pipefail

# Instrument anticipation entrypoint.
# Targets are the raw `ant_reg_{InstrumentName}` values from the benchmark CSV.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK_NAME="${TASK_NAME:-cholec80_instrument_anticipation}"
export CONFIG_PATH="${CONFIG_PATH:-configs/surgical_phase_anticipation/cholec80_instrument_anticipation_vitg-xformer_64f.yaml}"
export TMP_PREFIX="${TMP_PREFIX:-nsjepa_cholec80_instrument}"

"${SCRIPT_DIR}/run_cholec80_anticipation_common.sh"
