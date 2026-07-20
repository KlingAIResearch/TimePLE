#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_FALLBACK:-python}"
fi

usage() {
  cat <<'EOF'
Usage: bash data_pipeline/run_pipeline.sh <command> [args...]

Commands:
  train          Run normalize -> inference -> IoU training-data pipeline
  vtg            Run the VTG cleaning pipeline
  benchmark-web  Start benchmark annotation-correction Web application
  benchmark-apply Apply reviewed benchmark corrections
  help           Show this message

All arguments after the command are passed to the underlying final script.
EOF
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  train)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train_building/run_full_pipeline.py" "$@"
    ;;
  vtg)
    exec bash "${SCRIPT_DIR}/vtg_data_cleaning/run_vtg_pipeline.sh" "$@"
    ;;
  benchmark-web|web)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_cleaning/review_app.py" "$@"
    ;;
  benchmark-apply)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_cleaning/apply_reviews.py" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac
