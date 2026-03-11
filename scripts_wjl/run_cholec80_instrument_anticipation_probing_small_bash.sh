#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-configs/surgical_phase_anticipation/cholec80_instrument_anticipation_vitg-xformer_64f_small.yaml}"

"${SCRIPT_DIR}/run_cholec80_instrument_anticipation_probing_bash.sh"
