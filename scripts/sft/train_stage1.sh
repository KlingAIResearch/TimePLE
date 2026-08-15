#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT}/configs/sft/timeple_sft_stage1.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
export TIMEPLE_ROOT="${TIMEPLE_ROOT:-${ROOT}}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Run: bash scripts/setup_env.sh sft" >&2; exit 2; }
exec "${PYTHON_BIN}" -m swift.cli.main sft --timeple_codec_config_file "${CONFIG_FILE}" "$@"
