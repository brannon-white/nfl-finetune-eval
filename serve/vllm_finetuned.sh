#!/usr/bin/env bash
# Serves the merged fine-tune on :8001, OpenAI-compatible API.
# Run from the repo root (paths below are relative to it).
set -euo pipefail

vllm serve models/merged \
    --port 8001 \
    --served-model-name finetuned \
    --max-model-len 4096
