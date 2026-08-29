"""Embed the FULL corpus (all splits, held_out included) into a persistent Chroma
collection. See README "Critical eval-design decision" -- RAG has no training
phase, so restricting its index to the fine-tune's train split would be an unfair
handicap, not a fair comparison.

Usage:
    python scripts/03_build_rag_index.py
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks_full.jsonl"
CHROMA_DIR = ROOT / "chroma_db"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"{CHUNKS_PATH} not found -- run 02_build_sft_dataset.py first.")

    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text().splitlines() if line]

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(
        [c["text"] for c in chunks], show_progress_bar=True, normalize_embeddings=True
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        "nfl_rulebook", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[{"split": c["split"]} for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks into {CHROMA_DIR} (collection: nfl_rulebook)")


if __name__ == "__main__":
    main()
