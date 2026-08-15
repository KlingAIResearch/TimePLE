#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"

case "${PROFILE}" in
  dev|sft|rl|data-pipeline|data-gemini|data-vllm|eval) ;;
  *)
    echo "Usage: bash scripts/setup_env.sh {dev|sft|rl|data-pipeline|data-gemini|data-vllm|eval}" >&2
    exit 2
    ;;
esac

uv sync --frozen --extra "${PROFILE}"
PYTHON_BIN="${ROOT}/.venv/bin/python"

case "${PROFILE}" in
  sft)
    "${PYTHON_BIN}" "${ROOT}/scripts/build_integration.py" transformers
    "${PYTHON_BIN}" "${ROOT}/scripts/build_integration.py" ms-swift
    ;;
  rl)
    "${PYTHON_BIN}" "${ROOT}/scripts/build_integration.py" transformers
    "${PYTHON_BIN}" "${ROOT}/scripts/build_integration.py" easyr1
    ;;
  eval)
    "${PYTHON_BIN}" "${ROOT}/scripts/build_integration.py" transformers
    ;;
esac
