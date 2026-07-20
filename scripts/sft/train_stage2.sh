#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG_FILE="${CONFIG_FILE:-${ROOT}/configs/sft/timeple_sft_stage2.yaml}"
exec bash "${ROOT}/scripts/sft/train_stage1.sh" "$@"
