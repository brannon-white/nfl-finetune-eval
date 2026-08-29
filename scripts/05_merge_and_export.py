"""Merge the LoRA adapter into the base weights and export a plain HF model
directory that vLLM can serve directly (vLLM's LoRA support exists too, but
merging keeps the base-vs-finetuned comparison as two independent, equally-
served endpoints instead of one endpoint with adapter-switching).

Usage:
    python scripts/05_merge_and_export.py --base-model unsloth/Qwen2.5-7B-Instruct-bnb-4bit
"""
import argparse
from pathlib import Path

from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "models" / "lora_adapter"
MERGED_DIR = ROOT / "models" / "merged"


def main(base_model: str, max_seq_len: int) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=max_seq_len,
        load_in_4bit=False,  # merge in full precision, quantize at serve time if desired
    )
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {MERGED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()
    main(args.base_model, args.max_seq_len)
