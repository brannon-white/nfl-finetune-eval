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

| System | Served on | What it tests |
|---|---|---|
| Base, zero-shot | RunPod, :18000 | floor |
| Base + RAG | RunPod, :18000 (+ local retriever) | "did you even need to fine-tune" |
| Fine-tuned, zero-shot | RunPod, :18001 (vLLM LoRA adapter) or Vertex AI Endpoint | does the knowledge stick without retrieval |
| Fine-tuned + RAG | same as above (+ local retriever) | do they compose |

## Results

See [`results/REPORT.md`](results/REPORT.md) for the actual numbers from the
last full run. Headline: fine-tuning alone moved the needle only a little over
base; RAG did most of the work; fine-tuned+RAG didn't meaningfully beat RAG
alone. The general-capability regression slice showed real degradation on a
couple of MMLU subsets and TruthfulQA -- worth digging into before calling the
fine-tune a clean win. That gap between "fine-tuning helped a bit" and
"RAG already solved this" is the actual finding this project is built to
surface.

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

Everything runs on the RunPod pod (that's where the GPU is). A single 24GB
card can't hold two vLLM servers' full startup memory footprint at once, so
steps 6-7 run in two sequential passes -- base model up, run its two systems,
swap to the fine-tune, run its two systems -- rather than standing up all
four systems together.

```bash
python scripts/01_fetch_data.py          # parse the (manually downloaded) rulebook PDF
python scripts/02_build_sft_dataset.py   # chunk -> synthesize SFT pairs -> train/val/held_out split
python scripts/03_build_rag_index.py     # embed full corpus into Chroma
python scripts/04_train_lora.py          # Unsloth QLoRA fine-tune

bash serve/vllm_base.sh                                                          # base model on :18000
python eval/run_eval.py --systems base,base_rag                                  # first pass

# stop vllm_base.sh, free the GPU, then:
bash serve/vllm_finetuned.sh                                                     # fine-tune (LoRA) on :18001
python eval/run_eval.py --systems finetuned,finetuned_rag --finetuned-backend local --append  # second pass

bash eval/lm_eval_regression.sh          # base vs fine-tuned on MMLU/ARC/GSM8K/TruthfulQA subsets
python eval/report.py                    # aggregates everything into results/REPORT.md
```

`scripts/06_deploy_vertex_endpoint.py` / `07_teardown_vertex_endpoint.py` and
`serve/vertex_entrypoint.sh` + `serve/Dockerfile.vertex` ship a path to deploy
the fine-tune to a Vertex AI Endpoint instead of serving it locally (pass
`--finetuned-backend vertex`, the default, to `eval/run_eval.py`). That path
was scaffolded for the GCP managed-endpoint experience but not exercised in
this run -- see "Compute" below for the merge-vs-LoRA-serving decision that
affects it, and rework `06_deploy_vertex_endpoint.py` (it still assumes a
`models/merged/` directory) before using it.

## One manual step

Download the current NFL Official Football Operations rulebook PDF yourself
from nfloperations.com and place it at `data/raw/rulebook.pdf`. Not scripted
here deliberately — don't hardcode a scraper against a site's ToS into a repo
you're putting on a public portfolio.

## Compute: RunPod for training and serving

Training and serving both ran on a single RunPod pod (RTX 3090, 24GB) --
cheap, fast-iteration compute for the whole pipeline. A Vertex AI Endpoint
deployment path is scaffolded (`scripts/06_deploy_vertex_endpoint.py` /
`07_teardown_vertex_endpoint.py`, `serve/vertex_entrypoint.sh` +
`Dockerfile.vertex`) for the GCP managed-endpoint experience, but wasn't
exercised in the run behind `results/REPORT.md` -- see below for why, and for
what it'd take to actually run it.

**Training and serving (RunPod, single 24GB GPU):**

1. Spin up a RunPod pod with the `runpod/pytorch` template, 24GB+ GPU, ~50GB
   volume.
2. `git clone` this repo onto the pod, `pip install -r requirements.txt`.
3. Set env vars: `ANTHROPIC_API_KEY` (SFT synthesis + LLM-judge scoring),
   `HF_TOKEN` (gated/rate-limited model download).
4. Run steps 1-4 of the pipeline above. Training a 7B QLoRA on a few thousand
   examples for 2-3 epochs is on the order of an hour on a 3090.
5. Serve and eval as shown above (two sequential vLLM passes).
6. Stop the pod when idle — this is an hourly-billed rental, not a
   subscription.

**Why LoRA serving instead of a merged checkpoint:** the original plan was to
merge the adapter into the base and serve/deploy a plain HF model dir
(`scripts/05_merge_and_export.py`, still in the repo but unused). The base
model ships pre-quantized (bnb-4bit), so merging can't recover full-precision
weights, and re-saving a 4-bit-merged checkpoint hit an unfixable
`NotImplementedError` in this transformers version's quantized-tensor save
path. Rather than chase a library bug, `serve/vllm_finetuned.sh` serves the
adapter directly via vLLM's native `--enable-lora` support -- numerically
equivalent (the LoRA deltas are the same small matrices either way) and a
first-class, separately-tested vLLM code path. This is also why the Vertex
deploy script needs rework before use: it currently assumes a `models/merged/`
directory that this pipeline no longer produces. The straightforward fix is
either (a) point `06_deploy_vertex_endpoint.py` at the base weights + LoRA
adapter and switch its custom container to vLLM's `--enable-lora` flag, same
as `serve/vllm_finetuned.sh`, or (b) find an environment where the merge
actually succeeds (a non-quantized base, or an older transformers pin).

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
