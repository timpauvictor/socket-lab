#!/usr/bin/env python3
"""Shared task adapter: the translating skill on top of frozen language boxes.

For each pair (ro-en, fr-en, de-en in order): merge that pair's frozen box
adapters into the base with merge_and_unload, attach the shared task
adapter, train K steps on that pair's opus examples, save. Only the task
adapter ever trains; boxes are baked in frozen, artifacts stay separate.

K is derived, not set: K = ceil(pair_examples * epochs / effective_batch),
printed per pair (~18,750 at the full 150k-pair budget). Pass --max_steps
to cap K (smoke runs).

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
import json
from pathlib import Path
import os
import random
import time
from functools import partial

if __package__:
    from .translation_data import find_config, prepare_pairs, tokenize_completion, translation_examples
    from .task_model import allocate_steps, build_task_model, task_source
else:
    from translation_data import find_config, prepare_pairs, tokenize_completion, translation_examples
    from task_model import allocate_steps, build_task_model, task_source

import torch
from datasets import Dataset, get_dataset_config_names, load_dataset
from transformers import (AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)

def load_pair_examples(tok, pair, max_pairs):
    src, tgt = pair.split("-")
    cfg = find_config(src, tgt, get_dataset_config_names("Helsinki-NLP/opus-100"))
    print(f"Loading opus-100/{cfg}", flush=True)
    ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="train")

    rows = prepare_pairs(ds, src, tgt, max_pairs, random)
    print(f"Training on {len(rows)} unique pairs for {pair}", flush=True)
    return src, tgt, translation_examples(src, tgt, rows)


def tokenize_batch(tok, examples, *, eos_stop=False, paired_control=False):
    train_ds = Dataset.from_list(
        [{"prompt": p, "completion": c} for p, c in examples])
    return train_ds.map(partial(tokenize_completion, tokenizer=tok, eos_stop=eos_stop,
                                       paired_control=paired_control),
                        remove_columns=["prompt", "completion"])


def parameter_hashes(model):
    import hashlib
    frozen, task = hashlib.sha256(), hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest = task if parameter.requires_grad else frozen
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return frozen.hexdigest(), task.hexdigest()


def finite_callback():
    from transformers import TrainerCallback

    class FiniteTaskTraining(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError("Missing or nonfinite task gradients")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in (logs or {}).values()):
                raise RuntimeError("Nonfinite training metric")
    return FiniteTaskTraining()


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
    ap.add_argument("--tune_from", default="", help="Load existing task weights on the FIRST pair")
    ap.add_argument("--total_steps", type=int, default=0, help="Total budget divided across pairs, remainder last")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prepared_data", type=Path, help="Frozen pair JSON files, selected once for both controls")
    ap.add_argument("--eos_stop", action="store_true")
    ap.add_argument("--paired_control", action="store_true", help="Reserve the same EOS slot in both controls")
    ap.add_argument("--verify_updates", action="store_true", help="Smoke: hash frozen/task weights before and after each pair")
    args = ap.parse_args()
    if args.total_steps < 0 or (args.total_steps and (args.max_steps or args.resume_pair or args.resume_from)):
        ap.error("total_steps is incompatible with max_steps/resume; use a fresh run")
    if args.tune_from and (Path(args.output).exists() or args.resume_pair or args.resume_from):
        ap.error("tune_from requires a fresh output and does not support training resume")
    if args.paired_control and not (args.tune_from and args.prepared_data and args.revision):
        ap.error("paired_control requires tune_from, prepared_data and a pinned revision")

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

    budgets = allocate_steps(args.total_steps, len(pair_list)) if args.total_steps else None
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    start_idx = 0
    if args.resume_pair:
        if args.resume_pair not in pair_list:
            raise SystemExit(f"Unknown --resume_pair {args.resume_pair}; pairs: {pair_list}")
        start_idx = pair_list.index(args.resume_pair)

    for pair_index, pair in enumerate(pair_list):
        if pair_index < start_idx:
            print(f"Skipping {pair} (already trained; resuming from {args.resume_pair})", flush=True)
            continue
        if args.prepared_data:
            src, tgt = pair.split("-")
            rows = json.loads((args.prepared_data / f"{pair}.json").read_text(encoding="utf-8"))
            examples = translation_examples(src, tgt, rows)
        else:
            src, tgt, examples = load_pair_examples(tok, pair, args.max_pairs)
        train_ds = tokenize_batch(tok, examples, eos_stop=args.eos_stop, paired_control=args.paired_control)
        print(f"{len(train_ds)} training examples for {pair}", flush=True)

        # Derived K: full pass over this pair's examples, nothing hardcoded.
        derived_k = math.ceil(len(train_ds) * args.epochs / eff_batch)
        steps = budgets[pair_index] if budgets else (min(derived_k, args.max_steps) if args.max_steps > 0 else derived_k)
        print(f"Pair {pair}: {len(train_ds)} examples x {args.epochs} epochs "
              f"/ {eff_batch} per step -> K={derived_k}"
              + (f", capped to {steps}" if args.max_steps > 0 else ""), flush=True)

        # Fresh base + frozen boxes merged in; task adapter carries over.
        print(f"Merging frozen boxes {src}+{tgt} into base", flush=True)
        source = task_source(pair_index, args.tune_from, args.output)
        model = build_task_model(args.model, revision=args.revision,
                                 boxes=(boxes[src], boxes[tgt]), task_adapter=source,
                                 rank=args.rank, trainable=True)
        print(f"Task source: {source or 'fresh historical initialization'}; steps={steps}", flush=True)
        before = parameter_hashes(model) if args.verify_updates else None
        pair_started = time.monotonic()
        if args.verify_updates:
            torch.cuda.reset_peak_memory_stats()
        if args.compile:
            model = torch.compile(model)
            print(f"torch.compile enabled for {pair} (inductor, default mode)", flush=True)
        model.print_trainable_parameters()

        targs = TrainingArguments(
            output_dir=str(Path(args.output) / f"training-{pair}") if args.paired_control else args.output,
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
            save_total_limit=None if args.paired_control else 2,
            remove_unused_columns=False,  # compiled models hide their signature; our columns already match
            report_to="none",
            seed=args.seed,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                          data_collator=default_data_collator,
                          callbacks=[finite_callback()] if args.paired_control else None)
        if args.resume_pair == pair and args.resume_from:
            print(f"Resuming {pair} from checkpoint {args.resume_from}", flush=True)
            trainer.train(resume_from_checkpoint=args.resume_from)
        else:
            trainer.train()
        if trainer.state.global_step != steps:
            raise RuntimeError(f"Incorrect optimizer step count: {trainer.state.global_step} != {steps}")
        if before is not None:
            after = parameter_hashes(model)
            if before[0] != after[0] or before[1] == after[1]:
                raise RuntimeError("Smoke failed: frozen weights changed or task did not update")
        model.save_pretrained(args.output)
        if args.paired_control:
            model.save_pretrained(str(Path(args.output) / "pair-adapters" / pair))
            receipt = Path(args.output) / f"pair-{pair}.json"
            receipt.write_text(json.dumps({"pair": pair, "steps": trainer.state.global_step,
                "initial_source": source, "eos_stop": args.eos_stop,
                "update_verified": before is not None,
                "parameter_hashes_before": before,
                "parameter_hashes_after": after if before is not None else None,
                "metrics": trainer.state.log_history,
                "elapsed_seconds": time.monotonic() - pair_started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.verify_updates else None},
                allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved task adapter to {args.output} after {pair}", flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"Task adapter complete: {args.output}")


if __name__ == "__main__":
    main()
