#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/qwen_infer.template.yaml}"
WORLD_SIZE="${WORLD_SIZE:-2}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
STEP2_OUTPUT="${STEP2_OUTPUT:-${SCRIPT_DIR}/outputs/sequential_shards/step2_predictions.jsonl}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${SCRIPT_DIR}/outputs/sequential_shards/final_with_iou.jsonl}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${STEP2_OUTPUT}")"

for (( rank=0; rank<WORLD_SIZE; rank++ )); do
  echo "[INFO] running shard rank=${rank}/${WORLD_SIZE} on cuda=${CUDA_DEVICE}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/step2_infer_models.py" \
      --config "${CONFIG_PATH}" \
      --models qwen \
      --resume \
      --dp-enabled \
      --dp-world-size "${WORLD_SIZE}" \
      --dp-rank "${rank}" \
      --dp-local-rank 0
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/step2_infer_models.py" merge-shards \
  --shard-pattern "${STEP2_OUTPUT%.jsonl}.rank*.jsonl" \
  --output "${STEP2_OUTPUT}" \
  --global-index-field "_dp_global_index"

"${PYTHON_BIN}" "${SCRIPT_DIR}/step3_compute_iou.py" \
  --input "${STEP2_OUTPUT}" \
  --output "${FINAL_OUTPUT}" \
  --iou-alias qwen \
  --progress-every 20

echo "[DONE] final output: ${FINAL_OUTPUT}"
