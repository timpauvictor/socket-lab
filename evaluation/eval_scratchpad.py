#!/usr/bin/env python3
"""Scratchpad eval: ask for the pivot-language draft plus the final
translation in ONE generation, then score only the final answer.

Prompt format:
  Translate the following {SRC} text to {TGT}.
  First write the English translation, then write the {TGT} translation.
  {SRC}: {sentence}
  English:

Parse rule: take the text after the first "{TGT}:" marker, first line.
Rows without the marker count as failures (empty hyp) — honest scoring.

Usage (from the repository root, with a running model server):
  python evaluation/eval_scratchpad.py --pairs results/pairs-de-fr-devtest.jsonl \\
      --model socket-lab --run_id base-q8-b11146 --src de --pivot en --tgt fr \\
      --base_url http://localhost:8081 --output logs/my-scratch-de-fr.jsonl
"""
import argparse
from pathlib import Path

from eval_llama import (
    append_record,
    build_meta,
    check_server,
    open_records,
    post_completion,
    report_chrf,
    load_pairs,
    validate_args,
)
from eval_llama import LANG_NAMES


def build_scratch_prompt(src: str, pivot: str, tgt: str, sentence: str) -> str:
    return (
        f"Translate the following {LANG_NAMES[src]} text to {LANG_NAMES[tgt]}.\n"
        f"First write the {LANG_NAMES[pivot]} translation, "
        f"then write the {LANG_NAMES[tgt]} translation.\n"
        f"{LANG_NAMES[src]}: {sentence}\n{LANG_NAMES[pivot]}:"
    )


def extract_final(text: str, tgt: str) -> tuple[str, bool]:
    marker = f"{LANG_NAMES[tgt]}:"
    if marker not in text:
        return "", False
    return text.split(marker, 1)[1].split("\n", 1)[0].strip(), True



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_id", required=True, help="Stable identity of weights, adapters and backend")
    ap.set_defaults(strategy="scratchpad", pivot=None)
    ap.add_argument("--src", default="de")
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--tgt", default="fr")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--request_timeout", type=float, default=300.0)
    ap.add_argument("--base_url", default="http://localhost:8081")
    ap.add_argument("--output", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    validate_args(ap, args)
    pairs = load_pairs(Path(args.pairs), src=args.src, tgt=args.tgt, split=args.split)
    meta = build_meta(args)
    output = Path(args.output)
    check_server(args.base_url.rstrip("/"), args.model, args.request_timeout)
    stream, completed = open_records(output, meta, args.overwrite, expected_pairs=pairs)

    done = 0
    followed = 0
    with stream:
        for row_id, source, reference in pairs:
            if row_id in completed:
                continue
            prompt = build_scratch_prompt(args.src, args.pivot, args.tgt, source)
            # No newline stop: the scratchpad needs multi-line output
            # (English line, then the French line). The parse takes the
            # first line after the target marker, so trailing ramble
            # does not affect scoring. Generation still halts on EOS.
            raw = post_completion(
                args.base_url.rstrip("/"), args.model, prompt,
                args.max_new_tokens, args.request_timeout, stop=[])
            hyp, ok = extract_final(raw, args.tgt)
            followed += ok
            append_record(stream, {"id": row_id, "src": source,
                                   "hyp": hyp, "ref": reference,
                                   "followed_format": ok})
            done += 1
            print(f"  {done} (format {followed}/{done})", flush=True)

    print(f"followed format (new rows this invocation): {followed}/{done}")
    report_chrf(args.src, args.tgt, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
