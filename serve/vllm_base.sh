#!/usr/bin/env bash
# Serves the base instruct model on :8000, OpenAI-compatible API.
set -euo pipefail

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --served-model-name base \
    --max-model-len 4096
