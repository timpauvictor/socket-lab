#!/usr/bin/env python3
"""LoRA post-training for translation. Used for the monolith control;
reusable later for language/task cartridges.

From the repository root (requires training dependencies and model/data access).

Monolith (all pairs mixed, both directions, the control to beat):
  python training/train_lora.py --model Qwen/Qwen3-0.6B-Base --output ./monolith \\
      --pairs ro-en,fr-en,de-en --max_pairs 150000 --epochs 2 --rank 32

Smoke test (verifies the loop in ~2 minutes before a full run):
  python training/train_lora.py --model Qwen/Qwen3-0.6B-Base --output ./smoke \\
      --pairs ro-en --max_pairs 500 --max_steps 20 --rank 8

No TRL on purpose — plain HF Trainer + PEFT + manual completion masking.
"""
import argparse
import json
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pairs", default="ro-en", help="comma-separated src-tgt pairs, e.g. ro-en,fr-en,de-en")
    ap.add_argument("--max_pairs", type=int, default=200000)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = use epochs")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume_from", default="", help="Adapter checkpoint dir for MANUAL resume: LoRA weights and step count are restored, optimizer/scheduler restart fresh (short re-warmup); empty starts fresh")
    ap.add_argument("--tune_from", default="", help="Adapter dir for a capped tune-up: loads LoRA weights like --resume_from but skips step accounting (no trainer_state.json needed) and trains exactly --max_steps; --max_steps > 0 required")
    ap.add_argument("--eos_stop", action="store_true", help="append one supervised EOS token after each completion (used in the termination tune-up)")
    ap.add_argument("--compile", action="store_true",
                    help="wrap the model with torch.compile (inductor); off by default, enable only after a smoke test")
    args = ap.parse_args()
    pair_list = [p.strip() for p in args.pairs.split(",") if p.strip()]
    if not pair_list:
        raise SystemExit("No pairs given in --pairs")
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pair_rows = []
    for pair in pair_list:
        src, tgt = pair.split("-")
        cfg = find_config(src, tgt, get_dataset_config_names("Helsinki-NLP/opus-100"))
        print(f"Loading opus-100/{cfg}", flush=True)
        ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="train")

        rows = prepare_pairs(ds, src, tgt, args.max_pairs, random, progress_every=100000)
        print(f"Training on {len(rows)} unique pairs for {pair}", flush=True)
        pair_rows.append((src, tgt, rows))
    print(f"Training on {sum(len(r) for _, _, r in pair_rows)} unique pairs total", flush=True)

    # Both directions. Prompt is masked (-100); loss only on the completion.
    examples = []
    for src, tgt, rows in pair_rows:
        examples.extend(translation_examples(src, tgt, rows))

    train_ds = Dataset.from_list(
        [{"prompt": p, "completion": c} for p, c in examples])
    train_ds = train_ds.map(partial(tokenize_completion, tokenizer=tok, eos_stop=args.eos_stop),
                            remove_columns=["prompt", "completion"])
    print(f"{len(train_ds)} training examples (both directions)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    lora = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    steps_done = 0
    if args.resume_from and args.tune_from:
        raise SystemExit("pick one: --resume_from or --tune_from, not both")
    resume_dir = args.resume_from or args.tune_from
    tune_up = bool(args.tune_from)
    if tune_up and args.max_steps <= 0:
        raise SystemExit("--tune_from needs an explicit cap: --max_steps N")
    if resume_dir:
        # Manual resume: Trainer-native resume_from_checkpoint cannot load an
        # adapter-only dir (it demands a full-model weight index), so restore
        # the LoRA weights directly and train on. Optimizer/scheduler state is
        # NOT restored; expect a short re-warmup. Tune-ups skip the step
        # accounting (and trainer_state.json): they train exactly max_steps.
        want = ("adapter_config.json", "adapter_model.safetensors")
        if not tune_up:
            want = want + ("trainer_state.json",)
        for f in want:
            if not os.path.isfile(os.path.join(resume_dir, f)):
                raise SystemExit(f"resume dir missing {f}: {resume_dir}")
        # NOTE: `model` here must be the plain base model. Feeding an
        # already-wrapped PeftModel into from_pretrained double-wraps:
        # the param count looks right but the forward routes through
        # frozen weights and backward explodes. Each arm wraps exactly once.
        model = PeftModel.from_pretrained(model, resume_dir)
        # Loaded adapters arrive frozen (version-dependent); only lora_
        # weights exist here, so unfreezing them revives exactly the adapter.
        unfrozen = 0
        for pname, param in model.named_parameters():
            if "lora_" in pname:
                param.requires_grad = True
                unfrozen += param.numel()
        if unfrozen == 0:
            raise SystemExit(f"--resume_from dir has no LoRA weights: {args.resume_from}")
        if tune_up:
            print(f"Tuning up adapter weights from {resume_dir} "
                  f"({unfrozen} trainable params; fresh optimizer/scheduler; capped at {args.max_steps} steps)", flush=True)
        else:
            with open(os.path.join(resume_dir, "trainer_state.json")) as f:
                steps_done = json.load(f)["global_step"]
            print(f"Manually resumed adapter weights from {resume_dir} "
                  f"(step {steps_done}; {unfrozen} trainable params; fresh optimizer/scheduler)", flush=True)
    else:
        model = get_peft_model(model, lora)
    if args.compile:
        model = torch.compile(model)
        print("torch.compile enabled (inductor, default mode)", flush=True)
    model.print_trainable_parameters()

    eff_batch = args.batch_size * args.grad_accum
    if resume_dir:
        # Remaining-steps accounting: a fresh Trainer always starts its
        # dataloader at step 0, so a true resume caps at total-minus-done
        # instead of retraining the full schedule. (Data order also
        # restarts; this is not an exact continuation.)
        # Tune-ups skip the accounting and train exactly max_steps.
        total_steps = math.ceil(len(train_ds) * args.epochs / eff_batch)
        if tune_up:
            max_steps = args.max_steps
            print(f"Tune-up: {max_steps} steps from {resume_dir} (of {total_steps} scheduled)", flush=True)
        else:
            remaining = total_steps - steps_done
            if remaining <= 0:
                raise SystemExit(f"Nothing left to train: step {steps_done} of {total_steps}")
            print(f"Resuming at step {steps_done}/{total_steps}; training {remaining} more steps", flush=True)
            max_steps = args.max_steps if args.max_steps > 0 else remaining
    else:
        max_steps = args.max_steps if args.max_steps > 0 else -1
    targs = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
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
    trainer.train()
    model.save_pretrained(args.output)
    print(f"Saved LoRA to {args.output}")


if __name__ == "__main__":
    main()
