"""Run all 4 systems (base, base+RAG, fine-tuned, fine-tuned+RAG) against the
held-out set and score each answer with the LLM judge.

Requires both vLLM servers running (serve/vllm_base.sh on :8000,
serve/vllm_finetuned.sh on :8001) and the Chroma index built (03_build_rag_index.py).

Usage:
    python eval/run_eval.py --top-k 3
"""
import argparse
import csv
import json
from pathlib import Path

import chromadb
from anthropic import Anthropic
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from llm_judge import judge

ROOT = Path(__file__).resolve().parent.parent
HELD_OUT_PATH = ROOT / "data" / "eval" / "held_out.jsonl"
CHROMA_DIR = ROOT / "chroma_db"
RESULTS_PATH = ROOT / "results" / "eval_results.csv"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

SYSTEMS = {
    "base": {"port": 8000, "model": "base", "rag": False},
    "base_rag": {"port": 8000, "model": "base", "rag": True},
    "finetuned": {"port": 8001, "model": "finetuned", "rag": False},
    "finetuned_rag": {"port": 8001, "model": "finetuned", "rag": True},
}

RAG_PROMPT_TEMPLATE = """Answer the question using only the rule text below.

Rule text:
{context}

Question: {question}"""


def retrieve(collection, embedder, question: str, top_k: int) -> str:
    query_vec = embedder.encode([question], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=query_vec, n_results=top_k)
    return "\n\n".join(result["documents"][0])


def query_vllm(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    return resp.choices[0].message.content


def main(top_k: int) -> None:
    examples = [json.loads(line) for line in HELD_OUT_PATH.read_text().splitlines() if line]

    embedder = SentenceTransformer(EMBED_MODEL)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_collection("nfl_rulebook")

    judge_client = Anthropic()
    clients = {
        8000: OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed"),
        8001: OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed"),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "system", "question", "answer", "score", "rationale"])

        for ex in examples:
            context = retrieve(collection, embedder, ex["question"], top_k)
            for system_name, cfg in SYSTEMS.items():
                prompt = (
                    RAG_PROMPT_TEMPLATE.format(context=context, question=ex["question"])
                    if cfg["rag"] else ex["question"]
                )
                answer = query_vllm(clients[cfg["port"]], cfg["model"], prompt)
                result = judge(judge_client, ex["question"], ex["source_text"],
                                ex["answer"], answer)
                writer.writerow([
                    ex["chunk_id"], system_name, ex["question"], answer,
                    result["score"], result["rationale"],
                ])
                print(f"{ex['chunk_id']:20s} {system_name:15s} score={result['score']}")

    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    main(args.top_k)
