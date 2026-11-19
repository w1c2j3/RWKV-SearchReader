#!/bin/bash
set -euo pipefail

# Dedicated router launcher when running a separate G1L pure-continuation port.
# By default SearchReader shares the multi-GPU G1L Lightning endpoints with prefix caching.
export RWKV_PORT="${RWKV_ROUTER_PORT:-18430}"
export RWKV_STATE_DB_PATH="${RWKV_ROUTER_STATE_DB_PATH:-./data/states-router.db}"
export SEARCH_READER_RWKV_API_KEY="${SEARCH_READER_ROUTER_API_KEY:-${SEARCH_READER_RWKV_API_KEY:-}}"

exec "$(dirname "$0")/launch_answer.sh"
