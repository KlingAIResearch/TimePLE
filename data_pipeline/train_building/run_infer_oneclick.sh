#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP2_SCRIPT="${SCRIPT_DIR}/step2_infer_models.py"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

MODEL="qwen"
LAUNCHER="local"        # local | mpi
CONFIG_PATH="${SCRIPT_DIR}/configs/qwen_infer.template.yaml"
HOSTFILE_KIND="mpi_hostfile"  # hostfile | mpi_hostfile
NP=""
PPR="1"
SLOTS_PER_HOST="${SLOTS_PER_HOST:-8}"
ALLOW_RUN_AS_ROOT="${ALLOW_RUN_AS_ROOT:-auto}"   # auto | true | false
SSH_CONFIG_BYPASS="${SSH_CONFIG_BYPASS:-auto}"   # auto | true | false
MERGE_SHARDS="auto"       # auto | true | false
DRY_RUN="false"
SET_CUDA_VISIBLE_DEVICES="auto"  # auto | true | false

EXTRA_STEP2_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_infer_oneclick.sh [options] [-- <extra step2 args>]

Options:
  --model <qwen|gemini|both>     Model selector passed to step2 --models (default: qwen)
  --launcher <local|mpi>         Launcher mode (default: local)
  --config <path>                Step2 YAML config path (default: configs/qwen_infer.template.yaml)
  --hostfile-kind <hostfile|mpi_hostfile>
                                 Which key under runtime.distributed to use for mpirun --hostfile
                                 (default: mpi_hostfile)
  --np <int>                     mpirun total processes. If omitted in mpi mode, use distributed.world_size
  --ppr <int>                    mpirun --map-by ppr:<N>:node (default: 1)
  --slots-per-host <int>         Override slots/max-slots for every hostfile entry before mpirun
                                 (default: 8, higher priority than values inside the hostfile)
  --allow-run-as-root <auto|true|false>
                                 In mpi mode:
                                 - auto: enable only when current uid is 0
                                 - true: always add OpenMPI root override
                                 - false: never add root override
  --ssh-config-bypass <auto|true|false>
                                 In mpi mode:
                                 - auto: bypass system ssh_config only when it contains broken %{port}
                                 - true: always use 'ssh -F /dev/null ...' for mpirun remote launch
                                 - false: keep OpenMPI default ssh behavior
  --merge-shards                 Force merge shards after mpi run
  --no-merge-shards              Disable shard merge after mpi run
  --set-cuda-visible-devices <auto|true|false>
                                 In mpi mode:
                                 - auto: true when ppr>1, else false
                                 - true: export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
                                 - false: do not touch CUDA_VISIBLE_DEVICES
  --python <path>                Python executable (default: $PYTHON_BIN or python)
  --dry-run                      Print commands only
  -h, --help                     Show this help

Examples:
  # Local one-click Qwen
  bash data_pipeline/train_building/run_infer_oneclick.sh --model qwen

  # Local one-click Gemini
  bash data_pipeline/train_building/run_infer_oneclick.sh --model gemini

  # MPI one-click Qwen (use hostfile from config)
  bash data_pipeline/train_building/run_infer_oneclick.sh --model qwen --launcher mpi --ppr 4

  # Pass through extra args to step2
  bash data_pipeline/train_building/run_infer_oneclick.sh --model both -- --resume --progress-every 5
EOF
}

log() {
  echo "[run_infer_oneclick] $*"
}

die() {
  echo "[run_infer_oneclick][ERROR] $*" >&2
  exit 1
}

print_cmd() {
  printf '$'
  for part in "$@"; do
    printf ' %q' "$part"
  done
  printf '\n'
}

read_cfg_value() {
  local key="$1"
  local default_value="${2:-}"
  "${PYTHON_BIN}" - "$CONFIG_PATH" "$key" "$default_value" <<'PY'
import sys
from pathlib import Path
import yaml

config_path, key, default = sys.argv[1:4]
cfg_path = Path(config_path)
if not cfg_path.exists():
    print(default)
    raise SystemExit(0)

try:
    visited = set()

    def merge_dict(base, override):
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = merge_dict(result[k], v)
            else:
                result[k] = v
        return result

    def load_with_base(path: Path):
        resolved = path.resolve()
        if resolved in visited:
            raise ValueError(f"cyclic __base__: {resolved}")
        visited.add(resolved)

        with path.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        if not isinstance(content, dict):
            return {}
        base = content.pop("__base__", None)
        if not base:
            return content
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        if not base_path.exists():
            raise FileNotFoundError(base_path)
        base_cfg = load_with_base(base_path)
        return merge_dict(base_cfg, content)

    cfg = load_with_base(cfg_path)
except Exception:
    print(default)
    raise SystemExit(0)

cur = cfg
for token in key.split("."):
    if not isinstance(cur, dict) or token not in cur:
        print(default)
        raise SystemExit(0)
    cur = cur[token]

if cur is None:
    print("")
elif isinstance(cur, bool):
    print("true" if cur else "false")
else:
    print(cur)
PY
}

derive_shard_pattern() {
  local merged_output="$1"
  "${PYTHON_BIN}" - "$merged_output" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.suffix:
    pattern = path.with_name(f"{path.stem}.rank*{path.suffix}")
else:
    pattern = path.with_name(f"{path.name}.rank*")
print(str(pattern))
PY
}

build_effective_hostfile() {
  local source_hostfile="$1"
  local slots_per_host="$2"
  local output_hostfile
  output_hostfile="$(mktemp "${TMPDIR:-/tmp}/run_infer_oneclick.hostfile.XXXXXX")"

  "${PYTHON_BIN}" - "$source_hostfile" "$slots_per_host" "$output_hostfile" <<'PY'
import re
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
slots = int(sys.argv[2])
dst_path = Path(sys.argv[3])

slot_re = re.compile(r"(^|\s)slots=\d+(\s|$)")
max_slot_re = re.compile(r"(^|\s)max-slots=\d+(\s|$)")

lines = []
for raw_line in src_path.read_text(encoding="utf-8").splitlines():
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        lines.append(raw_line)
        continue

    line = slot_re.sub(lambda m: f"{m.group(1)}slots={slots}{m.group(2)}", raw_line)
    line = max_slot_re.sub(lambda m: f"{m.group(1)}max-slots={slots}{m.group(2)}", line)

    if not slot_re.search(line):
        line = f"{line} slots={slots}"
    if max_slot_re.search(raw_line) and not max_slot_re.search(line):
        line = f"{line} max-slots={slots}"

    lines.append(line)

dst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(str(dst_path))
PY
}

should_allow_run_as_root() {
  local mode="$1"
  case "$mode" in
    true) return 0 ;;
    false) return 1 ;;
    auto)
      [[ "$(id -u)" == "0" ]]
      return
      ;;
    *)
      die "--allow-run-as-root must be auto|true|false"
      ;;
  esac
}

ssh_config_has_broken_port_placeholder() {
  local ssh_config="${1:-/etc/ssh/ssh_config}"
  [[ -f "$ssh_config" ]] || return 1
  rg -q '^[[:space:]]*Port[[:space:]]+"?%\{port\}"?[[:space:]]*$' "$ssh_config" 2>/dev/null
}

should_bypass_ssh_config() {
  local mode="$1"
  case "$mode" in
    true) return 0 ;;
    false) return 1 ;;
    auto)
      ssh_config_has_broken_port_placeholder "/etc/ssh/ssh_config"
      return
      ;;
    *)
      die "--ssh-config-bypass must be auto|true|false"
      ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        MODEL="$2"
        shift 2
        ;;
      --launcher)
        LAUNCHER="$2"
        shift 2
        ;;
      --config)
        CONFIG_PATH="$2"
        shift 2
        ;;
      --hostfile-kind)
        HOSTFILE_KIND="$2"
        shift 2
        ;;
      --np)
        NP="$2"
        shift 2
        ;;
      --ppr)
        PPR="$2"
        shift 2
        ;;
      --slots-per-host)
        SLOTS_PER_HOST="$2"
        shift 2
        ;;
      --allow-run-as-root)
        ALLOW_RUN_AS_ROOT="$2"
        shift 2
        ;;
      --ssh-config-bypass)
        SSH_CONFIG_BYPASS="$2"
        shift 2
        ;;
      --merge-shards)
        MERGE_SHARDS="true"
        shift
        ;;
      --no-merge-shards)
        MERGE_SHARDS="false"
        shift
        ;;
      --set-cuda-visible-devices)
        SET_CUDA_VISIBLE_DEVICES="$2"
        shift 2
        ;;
      --python)
        PYTHON_BIN="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN="true"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        EXTRA_STEP2_ARGS+=("$@")
        break
        ;;
      *)
        die "Unknown argument: $1 (use --help)"
        ;;
    esac
  done
}

validate_args() {
  [[ "$MODEL" == "qwen" || "$MODEL" == "gemini" || "$MODEL" == "both" ]] || die "--model must be qwen|gemini|both"
  [[ "$LAUNCHER" == "local" || "$LAUNCHER" == "mpi" ]] || die "--launcher must be local|mpi"
  [[ "$HOSTFILE_KIND" == "hostfile" || "$HOSTFILE_KIND" == "mpi_hostfile" ]] || die "--hostfile-kind must be hostfile|mpi_hostfile"
  [[ "$SET_CUDA_VISIBLE_DEVICES" == "auto" || "$SET_CUDA_VISIBLE_DEVICES" == "true" || "$SET_CUDA_VISIBLE_DEVICES" == "false" ]] || \
    die "--set-cuda-visible-devices must be auto|true|false"
  [[ "$ALLOW_RUN_AS_ROOT" == "auto" || "$ALLOW_RUN_AS_ROOT" == "true" || "$ALLOW_RUN_AS_ROOT" == "false" ]] || \
    die "--allow-run-as-root must be auto|true|false"
  [[ "$SSH_CONFIG_BYPASS" == "auto" || "$SSH_CONFIG_BYPASS" == "true" || "$SSH_CONFIG_BYPASS" == "false" ]] || \
    die "--ssh-config-bypass must be auto|true|false"
  [[ "$SLOTS_PER_HOST" =~ ^[0-9]+$ ]] || die "--slots-per-host must be integer, got: $SLOTS_PER_HOST"
  (( SLOTS_PER_HOST > 0 )) || die "--slots-per-host must be > 0"
  [[ -f "$CONFIG_PATH" ]] || die "Config not found: $CONFIG_PATH"
  [[ -f "$STEP2_SCRIPT" ]] || die "Step2 script not found: $STEP2_SCRIPT"
}

build_step2_cmd() {
  local -n _out_ref=$1
  _out_ref=("${PYTHON_BIN}" "${STEP2_SCRIPT}" "--config" "${CONFIG_PATH}" "--models" "${MODEL}")
  if [[ "${#EXTRA_STEP2_ARGS[@]}" -gt 0 ]]; then
    _out_ref+=("${EXTRA_STEP2_ARGS[@]}")
  fi
}

maybe_run_merge() {
  local should_merge="$1"
  local merged_output="$2"
  local global_idx_field="$3"
  [[ "$should_merge" == "true" ]] || return 0

  local shard_pattern
  shard_pattern="$(derive_shard_pattern "$merged_output")"

  local merge_cmd=(
    "${PYTHON_BIN}" "${STEP2_SCRIPT}" "merge-shards"
    "--shard-pattern" "${shard_pattern}"
    "--output" "${merged_output}"
    "--global-index-field" "${global_idx_field}"
  )

  print_cmd "${merge_cmd[@]}"
  if [[ "$DRY_RUN" == "false" ]]; then
    "${merge_cmd[@]}"
  fi
}

run_local() {
  local step2_cmd=()
  build_step2_cmd step2_cmd

  print_cmd "${step2_cmd[@]}"
  if [[ "$DRY_RUN" == "false" ]]; then
    "${step2_cmd[@]}"
  fi
}

run_mpi() {
  local hostfile_path
  hostfile_path="$(read_cfg_value "runtime.distributed.${HOSTFILE_KIND}" "")"
  [[ -n "$hostfile_path" ]] || die "runtime.distributed.${HOSTFILE_KIND} is empty in config."
  if [[ ! -f "$hostfile_path" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
      log "Hostfile not found (dry-run only): ${hostfile_path}"
    else
      die "Hostfile not found: $hostfile_path"
    fi
  fi

  local effective_hostfile="$hostfile_path"
  if [[ -f "$hostfile_path" ]]; then
    effective_hostfile="$(build_effective_hostfile "$hostfile_path" "$SLOTS_PER_HOST")"
    trap "rm -f '${effective_hostfile}'" RETURN
  else
    log "Skip hostfile slot override because source hostfile does not exist: ${hostfile_path}"
  fi

  local world_size_cfg
  world_size_cfg="$(read_cfg_value "runtime.distributed.world_size" "1")"
  local np_value="$NP"
  if [[ -z "$np_value" ]]; then
    np_value="$world_size_cfg"
  fi
  [[ "$np_value" =~ ^[0-9]+$ ]] || die "--np must be integer, got: $np_value"
  (( np_value > 0 )) || die "--np must be > 0"

  [[ "$PPR" =~ ^[0-9]+$ ]] || die "--ppr must be integer, got: $PPR"
  (( PPR > 0 )) || die "--ppr must be > 0"

  local set_cuda="$SET_CUDA_VISIBLE_DEVICES"
  if [[ "$set_cuda" == "auto" ]]; then
    if (( PPR > 1 )); then
      set_cuda="true"
    else
      set_cuda="false"
    fi
  fi

  local step2_cmd=()
  build_step2_cmd step2_cmd

  local step2_cmd_str
  printf -v step2_cmd_str '%q ' "${step2_cmd[@]}"

  local worker_cmd
  if [[ "$set_cuda" == "true" ]]; then
    worker_cmd="export CUDA_VISIBLE_DEVICES=\${OMPI_COMM_WORLD_LOCAL_RANK}; export VLLM_CACHE_ROOT=\${TMPDIR:-/tmp}/vllm_cache_rank\${OMPI_COMM_WORLD_RANK}; ${step2_cmd_str}"
  else
    worker_cmd="export VLLM_CACHE_ROOT=\${TMPDIR:-/tmp}/vllm_cache_rank\${OMPI_COMM_WORLD_RANK}; ${step2_cmd_str}"
  fi

  local mpi_cmd=(
    mpirun
  )

  local allow_root="false"
  if should_allow_run_as_root "$ALLOW_RUN_AS_ROOT"; then
    allow_root="true"
    mpi_cmd+=("--allow-run-as-root")
  fi

  local bypass_ssh_config="false"
  if should_bypass_ssh_config "$SSH_CONFIG_BYPASS"; then
    bypass_ssh_config="true"
    mpi_cmd+=("--mca" "plm_rsh_agent" "ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")
  fi

  mpi_cmd+=(
    "--hostfile" "${effective_hostfile}"
    "-np" "${np_value}"
    "--map-by" "ppr:${PPR}:node"
    "bash" "-lc" "${worker_cmd}"
  )

  log "launcher=mpi hostfile=${hostfile_path} effective_hostfile=${effective_hostfile} slots_per_host=${SLOTS_PER_HOST} np=${np_value} ppr=${PPR} set_cuda_visible_devices=${set_cuda} allow_run_as_root=${allow_root} ssh_config_bypass=${bypass_ssh_config} vllm_cache_root=/tmp/vllm_cache_rank\${OMPI_COMM_WORLD_RANK}"
  print_cmd "${mpi_cmd[@]}"
  if [[ "$DRY_RUN" == "false" ]]; then
    "${mpi_cmd[@]}"
  fi

  local should_merge="$MERGE_SHARDS"
  local output_format
  output_format="$(read_cfg_value "runtime.format" "final")"
  if [[ "$should_merge" == "auto" ]]; then
    if [[ "$output_format" == "final" ]]; then
      should_merge="true"
    else
      should_merge="false"
    fi
  fi
  if [[ "$output_format" != "final" && "$should_merge" == "true" ]]; then
    die "merge-shards only supports final(JSONL) output format, current runtime.format=${output_format}"
  fi
  if [[ "$should_merge" == "true" ]]; then
    local merged_output global_idx_field
    merged_output="$(read_cfg_value "runtime.output_jsonl" "")"
    [[ -n "$merged_output" ]] || die "runtime.output_jsonl is empty in config."
    if [[ "$output_format" == "final" ]]; then
      merged_output="$("${PYTHON_BIN}" - "$merged_output" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
if p.suffix.lower() != ".jsonl":
    p = p.with_suffix(".jsonl")
print(str(p))
PY
)"
    fi
    global_idx_field="$(read_cfg_value "runtime.distributed.global_index_field" "_dp_global_index")"
    maybe_run_merge "$should_merge" "$merged_output" "$global_idx_field"
  fi
}

main() {
  parse_args "$@"
  validate_args

  log "model=${MODEL} launcher=${LAUNCHER} config=${CONFIG_PATH}"
  if [[ "$LAUNCHER" == "local" ]]; then
    run_local
  else
    run_mpi
  fi
}

main "$@"
