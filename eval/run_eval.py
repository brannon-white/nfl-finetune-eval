"""Run all 4 systems (base, base+RAG, fine-tuned, fine-tuned+RAG) against the
held-out set and score each answer with the LLM judge.

The base model is served locally (serve/vllm_base.sh on :18000, on the RunPod
pod). The fine-tuned model is served either the same way (serve/vllm_finetuned.sh
on :18001) or, by default, on the Vertex AI endpoint from
06_deploy_vertex_endpoint.py -- pass --finetuned-backend local to use :18001
instead. Either way the Chroma index must be built (03_build_rag_index.py).

On a single GPU too small to hold both vLLM servers' full startup memory
footprint at once (their torch.compile graph-capture spikes well above
steady-state weight size), run this in two passes instead of standing up both
servers together: base server up -> --systems base,base_rag; then swap to the
fine-tuned server -> --systems finetuned,finetuned_rag --append.

Usage:
    python eval/run_eval.py --top-k 3
    python eval/run_eval.py --finetuned-backend local
    python eval/run_eval.py --systems base,base_rag
    python eval/run_eval.py --systems finetuned,finetuned_rag --finetuned-backend local --append
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
from vertex_client import load_endpoint, query_vertex

ROOT = Path(__file__).resolve().parent.parent
HELD_OUT_PATH = ROOT / "data" / "eval" / "held_out.jsonl"
CHROMA_DIR = ROOT / "chroma_db"
RESULTS_PATH = ROOT / "results" / "eval_results.csv"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

SYSTEMS = {
    "base": {"backend": "local", "port": 18000, "model": "base", "rag": False},
    "base_rag": {"backend": "local", "port": 18000, "model": "base", "rag": True},
    "finetuned": {"backend": "finetuned", "model": "finetuned", "rag": False},
    "finetuned_rag": {"backend": "finetuned", "model": "finetuned", "rag": True},
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


def main(top_k: int, finetuned_backend: str, systems: list[str], append: bool) -> None:
    examples = [json.loads(line) for line in HELD_OUT_PATH.read_text().splitlines() if line]

    embedder = SentenceTransformer(EMBED_MODEL)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_collection("nfl_rulebook")

    judge_client = Anthropic()
    active_systems = {name: SYSTEMS[name] for name in systems}
    needs_local = any(cfg["backend"] == "local" for cfg in active_systems.values())
    needs_finetuned = any(cfg["backend"] == "finetuned" for cfg in active_systems.values())

    local_clients = {}
    vertex_endpoint = None
    if needs_local:
        local_clients[18000] = OpenAI(base_url="http://localhost:18000/v1", api_key="not-needed")
    if needs_finetuned:
        if finetuned_backend == "local":
            local_clients[18001] = OpenAI(base_url="http://localhost:18001/v1", api_key="not-needed")
        else:
            vertex_endpoint = load_endpoint()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (append and RESULTS_PATH.exists())

    done = set()
    if append and RESULTS_PATH.exists():
        with open(RESULTS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["chunk_id"], row["system"], row["question"]))

    with open(RESULTS_PATH, "a" if append else "w", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["chunk_id", "system", "question", "answer", "score", "rationale"])

        for ex in examples:
            pending = [
                (name, cfg) for name, cfg in active_systems.items()
                if (ex["chunk_id"], name, ex["question"]) not in done
            ]
            if not pending:
                continue
            context = retrieve(collection, embedder, ex["question"], top_k)
            for system_name, cfg in pending:
                prompt = (
                    RAG_PROMPT_TEMPLATE.format(context=context, question=ex["question"])
                    if cfg["rag"] else ex["question"]
                )
                for attempt in range(3):
                    try:
                        if cfg["backend"] == "local":
                            answer = query_vllm(local_clients[cfg["port"]], cfg["model"], prompt)
                        elif finetuned_backend == "local":
                            answer = query_vllm(local_clients[18001], cfg["model"], prompt)
                        else:
                            answer = query_vertex(vertex_endpoint, cfg["model"], prompt)
                        result = judge(judge_client, ex["question"], ex["source_text"],
                                        ex["answer"], answer)
                        break
                    except Exception as e:
                        if attempt == 2:
                            print(f"{ex['chunk_id']:20s} {system_name:15s} FAILED after 3 attempts: {e}")
                            result = None
                        else:
                            print(f"{ex['chunk_id']:20s} {system_name:15s} retry {attempt + 1}: {e}")
                if result is None:
                    continue
                writer.writerow([
                    ex["chunk_id"], system_name, ex["question"], answer,
                    result["score"], result["rationale"],
                ])
                f.flush()
                print(f"{ex['chunk_id']:20s} {system_name:15s} score={result['score']}")

    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--finetuned-backend", choices=["vertex", "local"], default="vertex")
    parser.add_argument(
        "--systems", default=",".join(SYSTEMS),
        help="Comma-separated subset of: " + ",".join(SYSTEMS),
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to an existing results CSV instead of overwriting it "
             "(for running systems in separate passes).",
    )
    args = parser.parse_args()
    main(args.top_k, args.finetuned_backend, args.systems.split(","), args.append)
