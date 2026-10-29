#!/usr/bin/env python3
"""Translation eval on FLORES-200. Reports chrF.

Examples from the repository root; these require model/data access and the
legacy evaluation dependencies, not just the CPU scoring package.

Base-model check:
  python evaluation/eval_translate.py --model Qwen/Qwen3-0.6B-Base --src ro --tgt en

Legacy PEFT evaluation (equal-weight linear factor merge, not a sum of deltas):
  python evaluation/eval_translate.py --model Qwen/Qwen3-0.6B-Base \\
      --adapters ./cart-ro ./cart-en ./cart-task --src ro --tgt en

Swap test:
  python evaluation/eval_translate.py --model Qwen/Qwen3-0.6B-Base \\
      --adapters ./cart-ro ./cart-fr ./cart-task --src ro --tgt fr
"""
import argparse

from prompts import translation_prompt

import sacrebleu
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LANG_NAMES = {"ro": "Romanian", "en": "English", "fr": "French", "fi": "Finnish"}
# FLORES-200 uses ISO 639-3 codes on the Hub
FLORES_CODES = {"ro": "ron_Latn", "en": "eng_Latn", "fr": "fra_Latn", "fi": "fin_Latn"}


def build_prompt(src, tgt, sentence):
    return translation_prompt(LANG_NAMES[src], LANG_NAMES[tgt], sentence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path of the frozen base")
    ap.add_argument("--adapters", nargs="*", default=[],
                    help="LoRA adapter dirs for the legacy equal-weight linear factor merge")
    ap.add_argument("--src", default="ro")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="eval only first N (0 = all)")
    args = ap.parse_args()

    print(f"Loading base: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")

    if args.adapters:
        from peft import PeftModel
        print(f"Loading {len(args.adapters)} adapters: {args.adapters}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapters[0], adapter_name="a0")
        for i, ad in enumerate(args.adapters[1:], 1):
            model.load_adapter(ad, adapter_name=f"a{i}")
        names = [f"a{i}" for i in range(len(args.adapters))]
        weights = [1.0 / len(names)] * len(names)
        model.add_weighted_adapter(names, weights, "combo", combination_type="linear")
        model.set_adapter("combo")
    model.eval()

    src_code, tgt_code = FLORES_CODES[args.src], FLORES_CODES[args.tgt]
    print(f"Loading FLORES-200 {args.split}: {src_code} -> {tgt_code}", flush=True)
    try:
        src_ds = load_dataset("facebook/flores", src_code, split=args.split)
        tgt_ds = load_dataset("facebook/flores", tgt_code, split=args.split)
    except Exception as e:
        raise SystemExit(
            "Could not load facebook/flores. It is license-gated: visit "
            "https://huggingface.co/datasets/facebook/flores, accept the terms, then run "
            "`huggingface-cli login`.\n"
            f"Original error: {e}")

    n = len(src_ds)
    if args.limit:
        n = min(n, args.limit)
    prompts = [build_prompt(args.src, args.tgt, src_ds[i]["sentence"]) for i in range(n)]
    refs = [tgt_ds[i]["sentence"] for i in range(n)]

    hyps = []
    with torch.inference_mode():
        for s in range(0, n, args.batch_size):
            batch = prompts[s:s + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(model.device)
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            in_len = enc["input_ids"].shape[1]
            for i in range(len(batch)):
                text = tok.decode(out[i][in_len:], skip_special_tokens=True)
                hyps.append(text.split("\n")[0].strip())
            print(f"  {min(s + args.batch_size, n)}/{n}", flush=True)

    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    print(f"\nchrF ({args.src}->{args.tgt}, n={n}): {chrf.score:.2f}")
    for i in range(min(3, n)):
        print(f"\nSRC: {src_ds[i]['sentence'][:120]}")
        print(f"HYP: {hyps[i][:120]}")
        print(f"REF: {refs[i][:120]}")


if __name__ == "__main__":
    main()
