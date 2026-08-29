"""Chunk the rulebook, synthesize SFT pairs, and split at the CHUNK level into
train / val / held_out.

Why split at the chunk level, not the question level: the held-out set has to
test generalization, not memorization. If two questions generated from the same
chunk landed on opposite sides of the split, the fine-tune could ace the
held-out question just by having memorized the other question's answer about
the same rule. Holding out whole chunks closes that leak.

RAG's index (built in 03_build_rag_index.py) intentionally uses ALL chunks,
held-out included -- see README "Critical eval-design decision" for why that's
not a contradiction.

Usage:
    python scripts/02_build_sft_dataset.py --pairs-per-chunk 3
"""
import argparse
import json
import random
from pathlib import Path

from anthropic import Anthropic

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_DIR = ROOT / "data" / "eval"

MODEL = "claude-sonnet-5"

SYNTH_SYSTEM_PROMPT = """You write instruction-tuning examples for an NFL \
rulebook fine-tune. Given a source passage from the rulebook, generate \
question/answer pairs that:
- are answerable ONLY from the given passage (no outside knowledge required)
- have a single, unambiguous, checkable answer
- vary in phrasing (not all "What is X?")
- include at least one question that requires connecting two facts in the \
passage, not pure lookup

Return strict JSON: a list of {"question": ..., "answer": ...} objects. No prose \
outside the JSON."""


def chunk_rulebook(chunk_chars: int = 1200, overlap: int = 150) -> list[dict]:
    path = RAW_DIR / "rulebook.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 01_fetch_data.py after placing rulebook.pdf")

    text = path.read_text()
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_chars
        piece = text[start:end].strip()
        if len(piece) > 200:  # drop tiny trailing scraps
            chunks.append({"chunk_id": f"rulebook_{idx:04d}", "text": piece})
            idx += 1
        start = end - overlap
    print(f"Rulebook: {len(chunks)} chunks")
    return chunks


def synthesize_pairs(client: Anthropic, chunk: dict, n_pairs: int) -> list[dict]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYNTH_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Generate {n_pairs} Q/A pairs from this passage:\n\n{chunk['text']}",
        }],
    )
    raw = resp.content[0].text
    try:
        pairs = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]") + 1
        pairs = json.loads(raw[start:end])
    return pairs


def main(pairs_per_chunk: int, seed: int = 42) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    chunks = chunk_rulebook()

    random.seed(seed)
    random.shuffle(chunks)
    n = len(chunks)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    for i, c in enumerate(chunks):
        c["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "held_out")

    (PROCESSED_DIR / "chunks_full.jsonl").write_text(
        "\n".join(json.dumps(c) for c in chunks)
    )
    print(f"Split: {n_train} train / {n_val} val / {n - n_train - n_val} held_out chunks")

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    train_examples, val_examples, held_out_examples = [], [], []

    for c in chunks:
        pairs = synthesize_pairs(client, c, pairs_per_chunk)
        for p in pairs:
            record = {
                "chunk_id": c["chunk_id"],
                "question": p["question"],
                "answer": p["answer"],
                "source_text": c["text"],
            }
            if c["split"] == "train":
                train_examples.append(record)
            elif c["split"] == "val":
                val_examples.append(record)
            else:
                held_out_examples.append(record)

    def to_sft_jsonl(examples: list[dict]) -> str:
        lines = []
        for ex in examples:
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": ex["question"]},
                    {"role": "assistant", "content": ex["answer"]},
                ]
            }))
        return "\n".join(lines)

    (PROCESSED_DIR / "sft_train.jsonl").write_text(to_sft_jsonl(train_examples))
    (PROCESSED_DIR / "sft_val.jsonl").write_text(to_sft_jsonl(val_examples))
    (EVAL_DIR / "held_out.jsonl").write_text(
        "\n".join(json.dumps(ex) for ex in held_out_examples)
    )

    print(f"SFT pairs: {len(train_examples)} train / {len(val_examples)} val / "
          f"{len(held_out_examples)} held_out (eval set)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-per-chunk", type=int, default=3)
    args = parser.parse_args()
    main(args.pairs_per_chunk)
