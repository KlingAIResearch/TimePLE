#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage: bash data_pipeline/vtg_data_cleaning/run_vtg_pipeline.sh <dataset> [stage]

Datasets: charades_sta | activitynet_captions
Stages: all | convert | iou | dedup | eval_flat

VTG grouping is deterministic and based only on annotation IoU. No model prompt is
used for grouping or deduplication. Model-assisted curation is performed later by
the shared training-data pipeline and its single canonical prompt.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

DATASET_NAME="$1"
RUN_STAGE="${2:-all}"
case "${DATASET_NAME}" in
  charades_sta)
    CONVERT_SCRIPT="${SCRIPT_DIR}/0_convert_charades_sta_to_video_queries.py"
    INPUT_PATH="${CHARADES_INPUT_PATH:-}"
    PREFIX="charades_sta_train"
    ;;
  activitynet_captions)
    CONVERT_SCRIPT="${SCRIPT_DIR}/0_convert_activitynet_captions_to_video_queries.py"
    INPUT_PATH="${ACTIVITYNET_INPUT_PATH:-}"
    PREFIX="activitynet_captions_train"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unsupported dataset: ${DATASET_NAME}" >&2
    exit 2
    ;;
esac

if [[ -z "${INPUT_PATH}" ]]; then
  echo "Set CHARADES_INPUT_PATH or ACTIVITYNET_INPUT_PATH for the selected dataset." >&2
  exit 2
fi

case "${RUN_STAGE}" in all|convert|iou|dedup|eval_flat) ;; *) usage; exit 2 ;; esac

OUTPUT_DIR="${VTG_OUTPUT_DIR:-${SCRIPT_DIR}/output/${DATASET_NAME}}"
mkdir -p "${OUTPUT_DIR}"
BY_VIDEO="${OUTPUT_DIR}/0_${PREFIX}_by_video.json"
GROUPS="${OUTPUT_DIR}/1_${PREFIX}_iou_groups.json"
DEDUP="${OUTPUT_DIR}/2_${PREFIX}_deduplicated.json"
EVAL_JSONL="${OUTPUT_DIR}/3_${PREFIX}_deduplicated.jsonl"
IOU_THRESHOLD="${VTG_DEDUP_IOU_THRESHOLD:-0.99}"
REPRESENTATIVE_POLICY="${VTG_REPRESENTATIVE_POLICY:-longest_query}"

if [[ "${RUN_STAGE}" == all || "${RUN_STAGE}" == convert ]]; then
  "${PYTHON_BIN}" "${CONVERT_SCRIPT}" --input "${INPUT_PATH}" --output "${BY_VIDEO}"
fi
if [[ "${RUN_STAGE}" == all || "${RUN_STAGE}" == iou ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/1_build_iou_candidate_groups.py" \
    --input "${BY_VIDEO}" --output "${GROUPS}" --iou-threshold "${IOU_THRESHOLD}" --group-mode clique
fi
if [[ "${RUN_STAGE}" == all || "${RUN_STAGE}" == dedup ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/2_build_deduplicated_training_set.py" \
    --input-by-video "${BY_VIDEO}" --input-groups "${GROUPS}" --output "${DEDUP}" \
    --representative-policy "${REPRESENTATIVE_POLICY}"
fi
if [[ "${RUN_STAGE}" == all || "${RUN_STAGE}" == eval_flat ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/6_convert_cleaned_by_video_to_eval_flat.py" \
    --input "${DEDUP}" --output "${EVAL_JSONL}" --output-format time_codec_jsonl --sort-rows
fi

echo "[DONE] deduplicated=${DEDUP} eval_jsonl=${EVAL_JSONL}"
