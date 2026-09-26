#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0
export PYTHONPATH=/home/chase/apps/rwkv-search-reader/src
export SEARCH_READER_STATE_CACHE=/home/chase/apps/rwkv-search-reader/runs/router-state-v1/triton
exec /home/chase/chase/RWKV-PEFT/.venv/bin/python /home/chase/apps/rwkv-search-reader/scripts/train_search_router.py --output /home/chase/apps/rwkv-search-reader/runs/router-state-v6r2 --serve-state /home/chase/apps/rwkv-search-reader/runs/router-state-v6r2/best-state.pth --port 18237 --served-model-name rwkv7-g1j-1.5b-search-router-state-v6r2
