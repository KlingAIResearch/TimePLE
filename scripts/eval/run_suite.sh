#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUITE="${SUITE:-charades_sta}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
CONFIG="${ROOT}/evaluation/configs/suites/${SUITE}.yaml"

[[ -x "${PYTHON_BIN}" ]] || { echo "Run: bash scripts/setup_env.sh eval" >&2; exit 2; }
[[ -f "${CONFIG}" ]] || { echo "Unknown evaluation suite: ${SUITE}" >&2; exit 2; }

export TIMEPLE_ROOT="${TIMEPLE_ROOT:-${ROOT}}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" "${ROOT}/evaluation/src/run_eval_suite.py" \
  --suite "${CONFIG}" --execute "$@"
