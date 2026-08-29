#!/usr/bin/env bash
# General-capability regression slice: run lm-evaluation-harness on both the
# base and merged fine-tuned model, same tasks, so any catastrophic-forgetting
# delta shows up as a number, not a vibe.
set -euo pipefail

# Run from the repo root (paths below are relative to it).

# Task names shift between lm-eval-harness releases -- run `lm_eval --tasks list`
# on your installed version and adjust this list before the first real run.
TASKS="mmlu_abstract_algebra,mmlu_high_school_us_history,arc_easy,gsm8k,truthfulqa_mc2"

lm_eval \
    --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
    --tasks "$TASKS" \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path results/regression_base.json

lm_eval \
    --model hf \
    --model_args pretrained=models/merged \
    --tasks "$TASKS" \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path results/regression_finetuned.json

echo "Wrote results/regression_base.json and results/regression_finetuned.json"
