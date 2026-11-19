#!/usr/bin/env bash
set -euo pipefail

# Launch RWKV-7 G1L pure-continuation inference via rwkv_lighting_cuda.
# Automatically uses ./models/rwkv7-g1l-7.2b-dsh-r32-merged.pth prepared by `bash scripts/setup.sh`,
# and automatically fans out across multiple local GPUs (ports 18430..18430+N-1) unless RWKV_PORT
# or CUDA_VISIBLE_DEVICES is explicitly pinned.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${RWKV_MODEL_PATH:-}" ]]; then
  MODEL_PATH="${RWKV_MODEL_PATH}"
elif [[ -f "${REPO_ROOT}/models/rwkv7-g1l-7.2b-dsh-r32-merged.pth" ]]; then
  MODEL_PATH="${REPO_ROOT}/models/rwkv7-g1l-7.2b-dsh-r32-merged.pth"
else
  MODEL_PATH="/dev/shm/rwkv7-g1l-7.2b-dsh-r32-merged.pth"
fi

LIGHTNING_BIN="${RWKV_LIGHTNING_BIN:-}"
if [[ -z "${LIGHTNING_BIN}" ]]; then
  if [[ -n "${RWKV_LIGHTNING_DIR:-}" && -x "${RWKV_LIGHTNING_DIR}/rwkv_lighting_cuda" ]]; then
    LIGHTNING_BIN="${RWKV_LIGHTNING_DIR}/rwkv_lighting_cuda"
  elif [[ -x "${REPO_ROOT}/lightning/bundle/rwkv_lighting_cuda" ]]; then
    LIGHTNING_BIN="${REPO_ROOT}/lightning/bundle/rwkv_lighting_cuda"
  elif command -v rwkv_lighting_cuda >/dev/null 2>&1; then
    LIGHTNING_BIN="$(command -v rwkv_lighting_cuda)"
  elif [[ -x "/home/rwkv/g1l-harbor-eval/lightning-d54c536/rwkv_lighting_cuda" ]]; then
    LIGHTNING_BIN="/home/rwkv/g1l-harbor-eval/lightning-d54c536/rwkv_lighting_cuda"
  fi
fi

if [[ -z "${LIGHTNING_BIN}" || ! -x "${LIGHTNING_BIN}" ]]; then
  echo "error: rwkv_lighting_cuda binary not found." >&2
  echo "  Place prebuilt bundle in ${REPO_ROOT}/lightning/bundle/ (from https://github.com/Alic-Li/rwkv_lightning_cuda/releases)" >&2
  echo "  or set RWKV_LIGHTNING_DIR=/path/to/lightning-bundle" >&2
  exit 1
fi

BIN_DIR="$(cd "$(dirname "${LIGHTNING_BIN}")" && pwd)"

if [[ -n "${RWKV_VOCAB_PATH:-}" ]]; then
  VOCAB_PATH="${RWKV_VOCAB_PATH}"
elif [[ -f "${REPO_ROOT}/models/rwkv_vocab_v20230424.txt" ]]; then
  VOCAB_PATH="${REPO_ROOT}/models/rwkv_vocab_v20230424.txt"
elif [[ -f "${BIN_DIR}/rwkv_vocab_v20230424.txt" ]]; then
  VOCAB_PATH="${BIN_DIR}/rwkv_vocab_v20230424.txt"
elif [[ -f "${BIN_DIR}/assets/rwkv_vocab_v20230424.txt" ]]; then
  VOCAB_PATH="${BIN_DIR}/assets/rwkv_vocab_v20230424.txt"
else
  echo "error: rwkv_vocab_v20230424.txt not found; run 'bash scripts/setup.sh' or set RWKV_VOCAB_PATH." >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "error: model file not found: ${MODEL_PATH}" >&2
  echo "  Run 'bash scripts/setup.sh' first or set RWKV_MODEL_PATH=/path/to/model.pth" >&2
  exit 1
fi

HOST="${RWKV_HOST:-127.0.0.1}"
BASE_PORT="${RWKV_PORT:-18430}"
DECODE_BATCH_SIZE="${RWKV_DECODE_BATCH_SIZE:-160}"
CMIX_SPARSE="${RWKV_CMIX_SPARSE:-off}"
export RWKV_ADAPTER_STAGING_MIB="${RWKV_ADAPTER_STAGING_MIB:-512}"

# Load SEARCH_READER_RWKV_API_KEY from .env.local if not already in env
if [[ -z "${SEARCH_READER_RWKV_API_KEY:-}" && -f "${REPO_ROOT}/.env.local" ]]; then
  RAW_KEY="$(grep -E '^SEARCH_READER_RWKV_API_KEY=' "${REPO_ROOT}/.env.local" | tail -n1 | cut -d= -f2- || true)"
  RAW_KEY="${RAW_KEY%\"}"
  RAW_KEY="${RAW_KEY#\"}"
  RAW_KEY="${RAW_KEY%\'}"
  RAW_KEY="${RAW_KEY#\'}"
  if [[ -n "${RAW_KEY}" && "${RAW_KEY}" != "your-model-key" ]]; then
    SEARCH_READER_RWKV_API_KEY="${RAW_KEY}"
  fi
fi

run_single_instance() {
  local gpu_id="$1"
  local port="$2"
  local state_db="${RWKV_STATE_DB_PATH:-${REPO_ROOT}/data/states-${port}.db}"
  mkdir -p "$(dirname "${state_db}")"
  local args=(
    --model-path "${MODEL_PATH}"
    --vocab-path "${VOCAB_PATH}"
    --host "${HOST}"
    --port "${port}"
    --decode-batch-size "${DECODE_BATCH_SIZE}"
    --chunk-size 512
    --wkv32
    --chunk-load
    --cmix-sparse "${CMIX_SPARSE}"
    --state-db-path "${state_db}"
  )
  if [[ -n "${SEARCH_READER_RWKV_API_KEY:-}" && "${SEARCH_READER_RWKV_API_KEY}" != "your-model-key" ]]; then
    args+=(--password "${SEARCH_READER_RWKV_API_KEY}")
  fi
  if [[ -n "${gpu_id}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" exec "${LIGHTNING_BIN}" "${args[@]}"
  else
    exec "${LIGHTNING_BIN}" "${args[@]}"
  fi
}

# Explicit single-instance mode when RWKV_PORT or CUDA_VISIBLE_DEVICES is set by caller/supervisor
if [[ -n "${RWKV_PORT:-}" || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  run_single_instance "${CUDA_VISIBLE_DEVICES:-}" "${BASE_PORT}"
fi

# Determine GPU list for multi-GPU auto fan-out
GPU_LIST=()
if [[ -n "${RWKV_GPUS:-}" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "${RWKV_GPUS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  if [[ "${GPU_COUNT}" -gt 1 ]]; then
    for ((i = 0; i < GPU_COUNT; i++)); do
      GPU_LIST+=("${i}")
    done
  fi
fi

if [[ "${#GPU_LIST[@]}" -le 1 ]]; then
  run_single_instance "${GPU_LIST[0]:-}" "${BASE_PORT}"
fi

mkdir -p "${REPO_ROOT}/logs"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "[launch] Starting ${#GPU_LIST[@]} rwkv_lighting_cuda workers on ports ${BASE_PORT}..$((BASE_PORT + ${#GPU_LIST[@]} - 1))"
for idx in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$idx]}"
  port=$((BASE_PORT + idx))
  log_file="${REPO_ROOT}/logs/lightning-${port}.log"
  echo "  -> GPU ${gpu} listening on http://${HOST}:${port}/v1 (log: ${log_file})"
  (
    RWKV_STATE_DB_PATH="${REPO_ROOT}/data/states-${port}.db" \
      run_single_instance "${gpu}" "${port}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
done

wait -n "${PIDS[@]}"
