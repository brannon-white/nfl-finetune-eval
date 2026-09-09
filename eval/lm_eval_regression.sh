#!/usr/bin/env bash
# General-capability regression slice: run lm-evaluation-harness on both the
# base and fine-tuned model, same tasks, so any catastrophic-forgetting delta
# shows up as a number, not a vibe.
#
# The fine-tuned side loads the LoRA adapter on top of its quantized base via
# lm-eval-harness's HF backend `peft=` arg, rather than a merged checkpoint --
# same reasoning as serve/vllm_finetuned.sh (merging hits an unfixable
# transformers/bitsandbytes save-path bug on this base model).
set -euo pipefail

# Run from the repo root (paths below are relative to it).

BASE_MODEL_DIR="${BASE_MODEL_DIR:-unsloth/Qwen2.5-7B-Instruct-bnb-4bit}"

# Task names shift between lm-eval-harness releases -- run `lm_eval --tasks list`
# on your installed version and adjust this list before the first real run.
TASKS="mmlu_abstract_algebra,mmlu_high_school_us_history,arc_easy,gsm8k,truthfulqa_mc2"

# --limit caps each task at N examples -- this is a fast regression *slice* to
# catch large forgetting deltas, not a full benchmark run; drop it for a real
# publication-grade number if time/budget allow.
LIMIT="${LM_EVAL_LIMIT:-100}"

lm_eval \
    --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
    --tasks "$TASKS" \
    --num_fewshot 0 \
    --batch_size 8 \
    --limit "$LIMIT" \
    --output_path results/regression_base.json

# lm-eval-harness always appends a timestamp to --output_path's filename --
# grab whatever it just wrote and rename it to the stable name report.py expects.
mv -f "$(ls -t results/regression_base*.json | head -1)" results/regression_base.json

# The base checkpoint already carries its own quantization_config (it's a
# pre-quantized bnb-4bit repo) -- passing load_in_4bit here hits a transformers
# version that no longer accepts it as a bare kwarg, and it's redundant anyway.
lm_eval \
    --model hf \
    --model_args pretrained="$BASE_MODEL_DIR",peft=models/lora_adapter \
    --tasks "$TASKS" \
    --num_fewshot 0 \
    --batch_size 8 \
    --limit "$LIMIT" \
    --output_path results/regression_finetuned.json

mv -f "$(ls -t results/regression_finetuned*.json | head -1)" results/regression_finetuned.json

echo "Wrote results/regression_base.json and results/regression_finetuned.json"
