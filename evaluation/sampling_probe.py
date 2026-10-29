#!/usr/bin/env python3
"""Sampling rescue probe v2: proper UTF-8 inputs (v1 ASCII-folded the
diacritics, mangling the input) and two conditions per prompt:
  A) greedy (temp 0.0) + repetition penalty 1.1 — isolates the penalty.
  B) temp 0.7 + repetition penalty 1.1 — full sampling rescue.
Usage: python evaluation/sampling_probe.py --out results/my-new-probe.jsonl
Existing outputs are never overwritten. Failed requests are error records,
not hypotheses, and cause a nonzero exit status.
"""
import argparse
import json
import sys

if __package__:
    from .eval_llama import request_text
    from .prompts import PROBE_PROMPTS as PROMPTS, translation_prompt
    from .repetition import has_loop
else:
    from eval_llama import request_text
    from prompts import PROBE_PROMPTS as PROMPTS, translation_prompt
    from repetition import has_loop

CONDITIONS = [
    ("greedy+penalty", 0.0, 1.1),
    ("sample07+penalty", 0.7, 1.1),
]


def complete(base_url: str, prompt: str, temperature: float, repeat_penalty: float) -> str:
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "repeat_penalty": repeat_penalty,
        "n_predict": 128,
        "stop": ["\n"],
        "stream": False,
    }
    _, body = request_text(base_url.rstrip("/") + "/completion", payload, 300, decode_errors="strict")
    response = json.loads(body)
    if not isinstance(response, dict) or not isinstance(response.get("content"), str):
        raise ValueError("Completion response must contain string content")
    return response["content"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8082")
    ap.add_argument("--out", required=True, help="new JSONL output path (must not exist)")
    args = ap.parse_args(argv)

    n_loop = 0
    n_total = 0
    n_failed = 0
    try:
        output = open(args.out, "x", encoding="utf-8")
    except OSError as exc:
        print(f"Cannot create probe output {args.out}: {exc}", file=sys.stderr)
        return 1
    with output as f:
        for i, (s, t, x) in enumerate(PROMPTS):
            prompt = translation_prompt(s, t, x, trailing_space=True)
            for cond, temp, pen in CONDITIONS:
                record = {"i": i, "cond": cond, "src_lang": s,
                          "tgt_lang": t, "prompt": prompt}
                try:
                    out = complete(args.base_url, prompt, temp, pen)
                except (OSError, ValueError, RuntimeError) as exc:
                    n_failed += 1
                    record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    print(f"prompt {i} {cond} ({s}->{t}): request failed ({type(exc).__name__})")
                else:
                    n_total += 1
                    looped = has_loop(out)
                    n_loop += looped
                    record.update(hyp=out, looped=looped)
                    print(f"prompt {i} {cond} ({s}->{t}): chars={len(out)} looped={looped}")
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                f.flush()
    print(f"done: {n_total} completions, {n_loop} looped, {n_failed} failed")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
