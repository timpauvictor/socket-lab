#!/usr/bin/env python3
"""Shared task adapter: the translating skill on top of frozen language boxes.

For each pair (ro-en, fr-en, de-en in order): merge that pair's frozen box
adapters into the base with merge_and_unload, attach the shared task
adapter, train K steps on that pair's opus examples, save. Only the task
adapter ever trains; boxes are baked in frozen, artifacts stay separate.

The script works out the step count for each pair:
K = ceil(pair_examples * epochs / effective_batch).
It prints K (~18,750 at the full 150k-pair budget). Use --max_steps to cap
this count for a short smoke run.

From the repository root (requires training dependencies and model/data access).

Full run (rank-16 boxes underneath):
  python training/train_task.py --model Qwen/Qwen3-0.6B-Base --output ./task \\
      --box_ro ./box-ro --box_en ./box-en --box_fr ./box-fr --box_de ./box-de \\
      --pairs ro-en,fr-en,de-en --max_pairs 150000 --epochs 2 --rank 32

Smoke test (rank-8 smoke boxes underneath — shapes must match the boxes):
  python training/train_task.py --model Qwen/Qwen3-0.6B-Base --output ./task-smoke \\
      --box_ro ./box-ro-smoke --box_en ./box-en-smoke --box_fr ./box-fr-smoke \\
      --box_de ./box-de-smoke --pairs ro-en,fr-en,de-en \\
      --max_pairs 500 --epochs 2 --max_steps 20 --rank 8

No TRL on purpose — plain HF Trainer + PEFT + manual completion masking.
Historical recipe: PEFT combination_type="linear" combines LoRA factors,
not their matrix-product deltas. Preserve it when reproducing this arm;
it is not equivalent to additive llama.cpp adapter stacking.
"""
import argparse
import math
import os
import random
from functools import partial

from translation_data import find_config, prepare_pairs, tokenize_completion, translation_examples

import torch
from datasets import Dataset, get_dataset_config_names, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)

def load_pair_examples(tok, pair, max_pairs):
    src, tgt = pair.split("-")
    cfg = find_config(src, tgt, get_dataset_config_names("Helsinki-NLP/opus-100"))
    print(f"Loading opus-100/{cfg}", flush=True)
    ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="train")

    rows = prepare_pairs(ds, src, tgt, max_pairs, random)
    print(f"Training on {len(rows)} unique pairs for {pair}", flush=True)
    return src, tgt, translation_examples(src, tgt, rows)


def tokenize_batch(tok, examples):
    train_ds = Dataset.from_list(
        [{"prompt": p, "completion": c} for p, c in examples])
    return train_ds.map(partial(tokenize_completion, tokenizer=tok),
                        remove_columns=["prompt", "completion"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--box_ro", required=True)
    ap.add_argument("--box_en", required=True)
    ap.add_argument("--box_fr", required=True)
    ap.add_argument("--box_de", required=True)
    ap.add_argument("--pairs", default="ro-en,fr-en,de-en")
    ap.add_argument("--max_pairs", type=int, default=150000)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="0 = derived K per pair; >0 caps K (smoke runs)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume_from", default="",
                    help="Trainer checkpoint dir to resume the resume pair from; empty starts fresh")
    ap.add_argument("--resume_pair", default="",
                    help="pair to restart from after preemption (e.g. fr-en); earlier pairs are skipped, "
                    "the task adapter carries over from args.output; empty starts at the first pair")
    ap.add_argument("--compile", action="store_true",
                    help="wrap the model with torch.compile (inductor) each pair cycle; "
                    "off by default, enable only after a smoke test")
    args = ap.parse_args()

    pair_list = [p.strip() for p in args.pairs.split(",") if p.strip()]
    if not pair_list:
        raise SystemExit("No pairs given in --pairs")
    boxes = {"ro": args.box_ro, "en": args.box_en,
             "fr": args.box_fr, "de": args.box_de}
    for lang, path in boxes.items():
        if not os.path.isdir(path):
            raise SystemExit(f"Box adapter missing for {lang}: {path}")
    random.seed(args.seed)
    eff_batch = args.batch_size * args.grad_accum

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    task_config = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             task_type="CAUSAL_LM")

    start_idx = 0
    if args.resume_pair:
        if args.resume_pair not in pair_list:
            raise SystemExit(f"Unknown --resume_pair {args.resume_pair}; pairs: {pair_list}")
        start_idx = pair_list.index(args.resume_pair)

    for pair_index, pair in enumerate(pair_list):
        if pair_index < start_idx:
            print(f"Skipping {pair} (already trained; resuming from {args.resume_pair})", flush=True)
            continue
        src, tgt, examples = load_pair_examples(tok, pair, args.max_pairs)
        train_ds = tokenize_batch(tok, examples)
        print(f"{len(train_ds)} training examples for {pair}", flush=True)

        # Derived K: full pass over this pair's examples, nothing hardcoded.
        derived_k = math.ceil(len(train_ds) * args.epochs / eff_batch)
        steps = min(derived_k, args.max_steps) if args.max_steps > 0 else derived_k
        print(f"Pair {pair}: {len(train_ds)} examples x {args.epochs} epochs "
              f"/ {eff_batch} per step -> K={derived_k}"
              + (f", capped to {steps}" if args.max_steps > 0 else ""), flush=True)

        # Fresh base + frozen boxes merged in; task adapter carries over.
        print(f"Merging frozen boxes {src}+{tgt} into base", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, boxes[src], adapter_name="bg0")
        model.load_adapter(boxes[tgt], adapter_name="bg1")
        model.add_weighted_adapter(["bg0", "bg1"], [1.0, 1.0],
                                   adapter_name="combo", combination_type="linear")
        model.set_adapter("combo")
        model = model.merge_and_unload()
        if pair_index == 0:
            model = get_peft_model(model, task_config)
            print("Created fresh task adapter", flush=True)
        else:
            model = PeftModel.from_pretrained(model, args.output)
            # Loaded adapters arrive frozen; only the task adapter exists
            # as LoRA here (boxes are baked into the base), so unfreezing
            # every lora_* weight revives exactly the task adapter.
            unfrozen = 0
            for pname, param in model.named_parameters():
                if "lora_" in pname:
                    param.requires_grad = True
                    unfrozen += param.numel()
            if unfrozen == 0:
                raise RuntimeError(f"No trainable LoRA weights in {args.output}")
            print(f"Continuing task adapter from {args.output} ({unfrozen} trainable params)", flush=True)
        if args.compile:
            model = torch.compile(model)
            print(f"torch.compile enabled for {pair} (inductor, default mode)", flush=True)
        model.print_trainable_parameters()

        targs = TrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_steps=steps,
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
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                          data_collator=default_data_collator)
        if args.resume_pair == pair and args.resume_from:
            print(f"Resuming {pair} from checkpoint {args.resume_from}", flush=True)
            trainer.train(resume_from_checkpoint=args.resume_from)
        else:
            trainer.train()
        model.save_pretrained(args.output)
        print(f"Saved task adapter to {args.output} after {pair}", flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"Task adapter complete: {args.output}")


if __name__ == "__main__":
    main()
