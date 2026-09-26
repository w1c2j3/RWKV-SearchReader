#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0
export OMP_NUM_THREADS=4
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=""
export VLLM_RWKV7_WKV_MODE=fp32io16
exec /home/chase/.venv-vllm-rwkv-8e90d04ecb/bin/python -m vllm.entrypoints.cli.main serve /mnt/nas-model/g1j/rwkv7-g1j-2.9b-20260831-ctx16384.pth --host 127.0.0.1 --port 18236 --tokenizer-mode rwkv --trust-request-chat-template --max-model-len 16384 --served-model-name rwkv7-g1j-2.9b-search-router --gpu-memory-utilization 0.12 --max-num-batched-tokens 4096 --max-num-seqs 4 --enforce-eager
