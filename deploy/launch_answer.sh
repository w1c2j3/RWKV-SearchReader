#!/bin/bash
set -euo pipefail

# Launch RWKV-7 G1L pure-continuation inference via rwkv_lighting_cuda.
# Override paths and port via environment variables when deploying locally.
LIGHTNING_DIR="${RWKV_LIGHTNING_DIR:-/home/rwkv/g1l-harbor-eval/lightning-d54c536}"
MODEL_PATH="${RWKV_MODEL_PATH:-/home/rwkv/g1l-harbor-eval/weights/rwkv7-g1l-7.2b-20261031-ctx34304.pth}"
VOCAB_PATH="${RWKV_VOCAB_PATH:-${LIGHTNING_DIR}/rwkv_vocab_v20230424.txt}"
STATE_DB_PATH="${RWKV_STATE_DB_PATH:-./data/states-18430.db}"
HOST="${RWKV_HOST:-127.0.0.1}"
PORT="${RWKV_PORT:-18430}"
export RWKV_ADAPTER_STAGING_MIB="${RWKV_ADAPTER_STAGING_MIB:-512}"

mkdir -p "$(dirname "${STATE_DB_PATH}")"

ARGS=(
  --model-path "${MODEL_PATH}"
  --vocab-path "${VOCAB_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --chunk-size 256
  --wkv32
  --chunk-load
  --cmix-sparse off
  --state-db-path "${STATE_DB_PATH}"
)
if [[ -n "${SEARCH_READER_RWKV_API_KEY:-}" ]]; then
  ARGS+=(--password "${SEARCH_READER_RWKV_API_KEY}")
fi

exec "${LIGHTNING_DIR}/rwkv_lighting_cuda" "${ARGS[@]}"
