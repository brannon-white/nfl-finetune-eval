# NFL Rulebook Fine-tune vs. RAG vs. Base — Eval Project

Portfolio project: LoRA/QLoRA fine-tune of an open 7-8B instruct model on the
NFL Official Rulebook, rigorously compared against the base model and a RAG
baseline on a held-out set, plus a general-capability regression slice to catch
catastrophic forgetting. The point isn't "I fine-tuned a model" — it's
demonstrating *when* fine-tuning beats RAG and when it doesn't.

## Why the rulebook

The rulebook is dense, cross-referencing text, not a lookup table of facts.
Base models often "know about" football rules in a shallow, web-trivia way but
get edge cases wrong. That makes it a genuine test of whether fine-tuning
teaches structure and reasoning over the corpus — as opposed to a domain like
raw season stats, where the answer is always "just use RAG, it's a moving
target," a conclusion you wouldn't need a fine-tune to reach.

## Systems under comparison

| System | Port | What it tests |
|---|---|---|
| Base, zero-shot | 8000 | floor |
| Base + RAG | 8000 (+ local retriever) | "did you even need to fine-tune" |
| Fine-tuned, zero-shot | 8001 | does the knowledge stick without retrieval |
| Fine-tuned + RAG | 8001 (+ local retriever) | do they compose |

## Critical eval-design decision: what "held-out" means for each system

The held-out split happens at the **source-chunk level**, not the question
level, and it applies differently to the two systems:

- **Fine-tuning**: SFT pairs generated from held-out chunks are *never* shown
  to the model during training. This is what makes the held-out eval a real
  test of generalization instead of memorization.
- **RAG**: the vector index is built over the **full corpus**, held-out chunks
  included. RAG has no training phase to leak into — in a real deployment you'd
  index everything. Restricting RAG's index to only the fine-tune's train split
  would make RAG look artificially worse and isn't how anyone would actually
  ship a RAG system.

If you flip this, the comparison is meaningless — say so explicitly in the
write-up if you change it.

## Pipeline (run in order)

```bash
python scripts/01_fetch_data.py          # parse the (manually downloaded) rulebook PDF
python scripts/02_build_sft_dataset.py   # chunk -> synthesize SFT pairs -> train/val/held_out split
python scripts/03_build_rag_index.py     # embed full corpus into Chroma
python scripts/04_train_lora.py          # Unsloth QLoRA fine-tune
python scripts/05_merge_and_export.py    # merge adapter -> HF model dir for vLLM

bash serve/vllm_base.sh                  # serves base model on :8000
bash serve/vllm_finetuned.sh             # serves merged fine-tune on :8001

python eval/run_eval.py                  # runs all 4 systems against held_out.jsonl
bash eval/lm_eval_regression.sh          # base vs fine-tuned on MMLU/ARC/GSM8K/TruthfulQA subsets
python eval/report.py                    # aggregates everything into results/REPORT.md
```

## One manual step

Download the current NFL Official Football Operations rulebook PDF yourself
from nfloperations.com and place it at `data/raw/rulebook.pdf`. Not scripted
here deliberately — don't hardcode a scraper against a site's ToS into a repo
you're putting on a public portfolio.

## Compute: RunPod, single RTX 4090

1. Spin up a RunPod pod with the `runpod/pytorch` template, 24GB+ GPU, ~50GB
   volume.
2. `git clone` this repo onto the pod, `pip install -r requirements.txt`.
3. Set env vars: `ANTHROPIC_API_KEY` (SFT synthesis + LLM-judge scoring),
   `HF_TOKEN` (gated model download, e.g. Llama 3.1).
4. Run the pipeline above. Training a 7-8B QLoRA on a few thousand examples for
   2-3 epochs is on the order of 1-2 hours on a 4090.
5. Stop the pod when idle — this is an hourly-billed rental, not a subscription.

## Scoring

All held-out questions are open-ended rule interpretation, scored by Claude as
an LLM judge against a written rubric (`eval/llm_judge.py`). Hand-check a
random sample yourself and report agreement — don't skip this, it's the
credibility of the whole eval.

## What goes in the portfolio write-up

Not "I fine-tuned a model and it got better." The actual deliverable is the
decision matrix: on which question types did fine-tuning win, where did RAG
already suffice, what happened to the regression slice, and what that implies
about when you'd reach for each approach on a real project.
