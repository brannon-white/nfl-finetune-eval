"""QLoRA fine-tune via Unsloth on the train split. Val split is used for
early-stopping / loss tracking only -- never touches the held-out eval set.

Usage:
    python scripts/04_train_lora.py --base-model unsloth/Qwen2.5-7B-Instruct-bnb-4bit
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
ADAPTER_DIR = ROOT / "models" / "lora_adapter"


def main(base_model: str, epochs: int, lr: float, max_seq_len: int) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files={
        "train": str(PROCESSED_DIR / "sft_train.jsonl"),
        "validation": str(PROCESSED_DIR / "sft_val.jsonl"),
    })

    def format_example(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = dataset.map(format_example)

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=SFTConfig(
            output_dir=str(ADAPTER_DIR),
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            bf16=True,
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Adapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()
    main(args.base_model, args.epochs, args.lr, args.max_seq_len)
