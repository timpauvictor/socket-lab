#!/usr/bin/env python3
"""Transformers termination check: greedy generation on the 6 probe prompts,
base vs tuned adapter, no llama.cpp needed. ASCII summary only.
Requires CUDA, training dependencies, and model access.

From the repository root: python evaluation/probe_terminate.py [--adapter DIR]
"""
import argparse

from prompts import PROBE_PROMPTS as PROMPTS, translation_prompt
from repetition import has_loop

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).to("cuda").eval()
        print("adapter: " + args.adapter, flush=True)
    else:
        print("adapter: none (base control)", flush=True)

    n_loop = 0
    with torch.inference_mode():
        for i, (s, t, x) in enumerate(PROMPTS):
            prompt = translation_prompt(s, t, x, trailing_space=True)
            enc = tok(prompt, return_tensors="pt").to("cuda")
            out = model.generate(**enc, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            looped = has_loop(text)
            n_loop += looped
            print(f"prompt {i} ({s}->{t}): gen_tokens={out.shape[1] - enc['input_ids'].shape[1]} chars={len(text)} looped={looped}", flush=True)
    print(f"done: {len(PROMPTS)} prompts, {n_loop} looped")


if __name__ == "__main__":
    main()
