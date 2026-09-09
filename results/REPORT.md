# Results

## Held-out rulebook eval (LLM-judge, 1-5)

| System | n | mean score (1-5) | stdev |
|---|---|---|---|
| base | 111 | 1.30 | 0.82 |
| base_rag | 111 | 3.58 | 1.68 |
| finetuned | 111 | 1.84 | 1.37 |
| finetuned_rag | 111 | 3.59 | 1.69 |

## General-capability regression slice

| Task | Base | Fine-tuned | Delta |
|---|---|---|---|
| mmlu_abstract_algebra | 0.600 | 0.530 | -0.070 ⚠️ |
| mmlu_high_school_us_history | 0.940 | 0.860 | -0.080 ⚠️ |
| arc_easy | 0.780 | 0.810 | +0.030 |
| gsm8k | 0.000 | 0.010 | +0.010 |
| truthfulqa_mc2 | 0.622 | 0.425 | -0.197 ⚠️ |

⚠️ = more than 2 points of absolute degradation vs. base -- possible
catastrophic forgetting, worth digging into before claiming the fine-tune is a
clean win.
