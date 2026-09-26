#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0
export OMP_NUM_THREADS=4
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=""
export VLLM_RWKV7_WKV_MODE=fp32io16
exec /home/chase/GitHub/RWKV-LH/data/runtime/engines/vllm-rwkv-67f0c5996c50/.venv/bin/python -B -m vllm.entrypoints.cli.main serve /home/chase/GitHub/RWKV-LH/data/models/rwkv7-g1j-13.3b-vllm-v1 --host 127.0.0.1 --port 18240 --tokenizer-mode rwkv --trust-request-chat-template --enable-auto-tool-choice --tool-call-parser rwkv --max-model-len 16384 --served-model-name rwkv7-g1j-13.3b-search-reader --gpu-memory-utilization 0.40 --max-num-batched-tokens 32768 --max-num-seqs 4
