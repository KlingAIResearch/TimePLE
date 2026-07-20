#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/rl/timeple_single_grpo_iou_dfl_format.yaml}"
HOSTFILE="${HOSTFILE:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

export TIMEPLE_ROOT="${TIMEPLE_ROOT:-${PROJECT_ROOT}}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CHECKPOINT_ROOT

[[ -f "${CONFIG_PATH}" ]] || { echo "Missing config: ${CONFIG_PATH}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing uv environment; run: uv sync --extra rl" >&2; exit 2; }

ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${HOSTFILE}" && -f "${HOSTFILE}" ]]; then
  export HOSTFILE
fi
exec "${PYTHON_BIN}" -m verl.trainer.main "${ARGS[@]}" "$@"
