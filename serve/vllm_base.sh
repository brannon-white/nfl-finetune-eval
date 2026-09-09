#!/usr/bin/env bash
# Serves the base instruct model on :18000, OpenAI-compatible API.
# (Using 18000 instead of the more obvious 8000/8001 because this pod's base
# image runs an nginx stub that already squats on 8001.)
set -euo pipefail

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 18000 \
    --served-model-name base \
    --max-model-len 4096 \
    --gpu-memory-utilization "${VLLM_BASE_GPU_UTIL:-0.65}"
