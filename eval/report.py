"""Aggregate eval_results.csv + the two lm-eval regression JSON files into a
single markdown report -- this is what feeds the portfolio case-study page.

Usage:
    python eval/report.py
"""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def summarize_held_out() -> str:
    path = RESULTS_DIR / "eval_results.csv"
    if not path.exists():
        return "_(no eval_results.csv found -- run eval/run_eval.py first)_"

    by_system = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            by_system.setdefault(row["system"], []).append(int(row["score"]))

    lines = ["| System | n | mean score (1-5) | stdev |", "|---|---|---|---|"]
    for system, scores in by_system.items():
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        lines.append(f"| {system} | {len(scores)} | {mean:.2f} | {stdev:.2f} |")
    return "\n".join(lines)


def summarize_regression() -> str:
    base_path = RESULTS_DIR / "regression_base.json"
    ft_path = RESULTS_DIR / "regression_finetuned.json"
    if not base_path.exists() or not ft_path.exists():
        return "_(no regression JSON found -- run eval/lm_eval_regression.sh first)_"

    base = json.loads(base_path.read_text())["results"]
    ft = json.loads(ft_path.read_text())["results"]

    lines = ["| Task | Base | Fine-tuned | Delta |", "|---|---|---|---|"]
    for task in base:
        base_score = list(base[task].values())[0]
        ft_score = list(ft[task].values())[0]
        delta = ft_score - base_score
        flag = " ⚠️" if delta < -0.02 else ""
        lines.append(f"| {task} | {base_score:.3f} | {ft_score:.3f} | {delta:+.3f}{flag} |")
    return "\n".join(lines)


def main() -> None:
    report = f"""# Results

## Held-out rulebook eval (LLM-judge, 1-5)

{summarize_held_out()}

## General-capability regression slice

{summarize_regression()}

⚠️ = more than 2 points of absolute degradation vs. base -- possible
catastrophic forgetting, worth digging into before claiming the fine-tune is a
clean win.
"""
    out_path = RESULTS_DIR / "REPORT.md"
    out_path.write_text(report)
    print(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
