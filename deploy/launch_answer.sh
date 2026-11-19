#!/bin/bash
set -euo pipefail

# Launch RWKV-7 G1L pure-continuation inference via rwkv_lighting_cuda.
# Recommended production mode: offline-merge trained NoRA weights with `rwkv_peft_weight_merge`
# and load the merged `.pth` directly via RWKV_MODEL_PATH (enables W8A16 + cmix-sparse + 64-row continuous batching).
LIGHTNING_DIR="${RWKV_LIGHTNING_DIR:-/home/rwkv/g1l-harbor-eval/lightning-d54c536}"
MODEL_PATH="${RWKV_MODEL_PATH:-/dev/shm/rwkv7-g1l-7.2b-dsh-r32-merged-w8a16.rwkvq}"
VOCAB_PATH="${RWKV_VOCAB_PATH:-${LIGHTNING_DIR}/rwkv_vocab_v20230424.txt}"
STATE_DB_PATH="${RWKV_STATE_DB_PATH:-./data/states-18430.db}"
HOST="${RWKV_HOST:-127.0.0.1}"
PORT="${RWKV_PORT:-18430}"
DECODE_BATCH_SIZE="${RWKV_DECODE_BATCH_SIZE:-256}"
CMIX_SPARSE="${RWKV_CMIX_SPARSE:-off}"
export RWKV_ADAPTER_STAGING_MIB="${RWKV_ADAPTER_STAGING_MIB:-512}"

mkdir -p "$(dirname "${STATE_DB_PATH}")"

ARGS=(
  --model-path "${MODEL_PATH}"
  --vocab-path "${VOCAB_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --decode-batch-size "${DECODE_BATCH_SIZE}"
  --chunk-size 512
  --wkv32
  --chunk-load
  --cmix-sparse "${CMIX_SPARSE}"
  --state-db-path "${STATE_DB_PATH}"
)
if [[ -n "${SEARCH_READER_RWKV_API_KEY:-}" ]]; then
  ARGS+=(--password "${SEARCH_READER_RWKV_API_KEY}")
fi

exec "${LIGHTNING_DIR}/rwkv_lighting_cuda" "${ARGS[@]}"
