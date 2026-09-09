#!/usr/bin/env bash
# Serves the fine-tune on :18001, OpenAI-compatible API, via vLLM's native LoRA
# adapter support -- base model + adapter applied at inference time, rather
# than a separately merged checkpoint.
#
# Why: merging the adapter into the pre-quantized (bnb-4bit) base and
# re-saving hits a save-path bug in this environment's transformers/bitsandbytes
# combo (NotImplementedError in the tensor reverse-transform system introduced
# for quantized-weight saving). vLLM's LoRA serving is a first-class, separately
# tested path that sidesteps it entirely and is numerically equivalent -- the
# LoRA deltas are small matrices applied on top of the frozen base regardless
# of how the base is saved.
#
# Run from the repo root (paths below are relative to it, except BASE_MODEL_DIR
# which is set by the training pipeline to wherever the base weights were
# downloaded during scripts/04_train_lora.py / 05_merge_and_export.py).
set -euo pipefail

BASE_MODEL_DIR="${BASE_MODEL_DIR:-unsloth/Qwen2.5-7B-Instruct-bnb-4bit}"

vllm serve "$BASE_MODEL_DIR" \
    --port 18001 \
    --max-model-len 4096 \
    --enable-lora \
    --lora-modules finetuned=models/lora_adapter \
    --gpu-memory-utilization "${VLLM_FT_GPU_UTIL:-0.20}"
