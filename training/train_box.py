#!/usr/bin/env python3
"""LoRA language box: causal pretraining on one Wikipedia language.

One adapter per language; no translation, no masking. Boxes are trained
independently, then merged frozen under the shared task adapter.

From the repository root (requires training dependencies and model/data access).

Full box (example Romanian):
  python training/train_box.py --model Qwen/Qwen3-0.6B-Base --output ./box-ro \\
      --lang ro --max_chunks 100000 --epochs 2 --rank 16

Smoke test (verifies the loop in ~2 minutes before a full run):
  python training/train_box.py --model Qwen/Qwen3-0.6B-Base --output ./box-smoke \\
      --lang ro --max_chunks 500 --max_steps 20 --rank 8 --sample head

No TRL on purpose — plain HF Trainer + PEFT, causal LM throughout.
"""
import argparse
import random

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)

WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIGS = {"ro": "20231101.ro", "en": "20231101.en",
                "fr": "20231101.fr", "de": "20231101.de"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lang", required=True, choices=sorted(WIKI_CONFIGS))
    ap.add_argument("--max_chunks", type=int, default=100000)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = use epochs")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--sample", default="reservoir", choices=("reservoir", "head"),
                    help="reservoir: uniform sample in one streaming pass; head: first chunks (smoke only)")
    ap.add_argument("--scan_mult", type=float, default=5.0,
                    help="reservoir scans at most scan_mult*max_chunks blocks, then stops; "
                    "0 means a full pass over the dump. Caps loading time on large "
                    "wikis at the cost of excluding late articles (documented bias).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading {WIKI_DATASET}/{WIKI_CONFIGS[args.lang]} (streaming)", flush=True)
    ds = load_dataset(WIKI_DATASET, WIKI_CONFIGS[args.lang], split="train", streaming=True)

    # Full 256-token blocks only: no padding, no masking, causal LM on all
    # tokens. Reservoir sampling gives a uniform subset in one bounded-memory
    # pass; head sampling is smoke-only and biased by dataset order.
    blocks = []
    seen = 0
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ids = tok(text, truncation=False)["input_ids"]
        for s in range(0, len(ids), args.block_size):
            block = ids[s:s + args.block_size]
            if len(block) < args.block_size:
                continue
            seen += 1
            if args.sample == "head":
                blocks.append(block)
                if len(blocks) >= args.max_chunks:
                    break
            elif len(blocks) < args.max_chunks:
                blocks.append(block)
            else:
                j = rng.randrange(seen)
                if j < args.max_chunks:
                    blocks[j] = block
        if args.sample == "head" and len(blocks) >= args.max_chunks:
            break
        if args.sample == "reservoir" and args.scan_mult > 0 \
                and seen >= int(args.max_chunks * args.scan_mult):
            print(f"  scan cap reached ({seen} scanned, kept {len(blocks)}); "
                  f"stopping reservoir pass", flush=True)
            break
        if seen % 20000 == 0:
            print(f"  scanned {seen} blocks, kept {len(blocks)}", flush=True)
    print(f"Kept {len(blocks)} blocks of {args.block_size} tokens", flush=True)
    if not blocks:
        raise SystemExit("No Wikipedia blocks collected")

    train_ds = Dataset.from_dict({
        "input_ids": blocks,
        "attention_mask": [[1] * args.block_size for _ in blocks],
        "labels": [list(b) for b in blocks],
    })

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    lora = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,  # compiled models hide their signature; our columns already match
        report_to="none",
        seed=args.seed,
    )
    Trainer(model=model, args=targs, train_dataset=train_ds,
            data_collator=default_data_collator).train()
    model.save_pretrained(args.output)
    print(f"Saved LoRA box to {args.output}")


if __name__ == "__main__":
    main()
