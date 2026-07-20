#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-${ROOT}/configs/rl/timeple_tr_spd.yaml}"
exec bash "${ROOT}/scripts/rl/train.sh" "$@"
