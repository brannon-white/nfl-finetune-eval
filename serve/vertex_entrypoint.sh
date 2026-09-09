#!/usr/bin/env bash
# Entrypoint for the Vertex AI custom-container serving image. Serves the
# fine-tune the same way serve/vllm_finetuned.sh does on RunPod: the base
# model + LoRA adapter applied at inference time via vLLM's native
# --enable-lora, not a merged checkpoint -- see README's "Compute" section
# for why (merging a pre-quantized base hits an unfixable transformers save
# bug on this base model).
#
# The LoRA adapter is small (~150MB) and comes from the artifact_uri set on
# the Vertex Model (Vertex sets AIP_STORAGE_URI to wherever that ends up --
# depending on the runtime version that's either a gs:// URI we download
# ourselves, or an already-mounted local path; handle both). The base model
# is comparatively huge (~5.5GB) and is a public HF repo, so it's simplest to
# just download it fresh at container start rather than round-trip it through
# GCS -- HF_TOKEN must be set (passed via serving_container_environment_variables)
# or this download will be throttled to the point of missing Vertex's startup
# health-check deadline.
set -euo pipefail

ADAPTER_DIR="/adapter"

if [[ "${AIP_STORAGE_URI:-}" == gs://* ]]; then
    echo "Downloading LoRA adapter from ${AIP_STORAGE_URI} to ${ADAPTER_DIR}..."
    mkdir -p "$ADAPTER_DIR"
    gcloud storage cp -r "${AIP_STORAGE_URI%/}/*" "$ADAPTER_DIR"
elif [[ -d "${AIP_STORAGE_URI:-}" ]]; then
    echo "Using pre-mounted adapter directory ${AIP_STORAGE_URI}"
    ADAPTER_DIR="$AIP_STORAGE_URI"
else
    echo "AIP_STORAGE_URI (${AIP_STORAGE_URI:-unset}) is neither a gs:// URI nor an existing local path" >&2
    exit 1
fi

BASE_MODEL_DIR="${BASE_MODEL_DIR:-unsloth/Qwen2.5-7B-Instruct-bnb-4bit}"

exec vllm serve "$BASE_MODEL_DIR" \
    --port "${AIP_HTTP_PORT:-8080}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --enable-lora \
    --lora-modules "finetuned=${ADAPTER_DIR}"
