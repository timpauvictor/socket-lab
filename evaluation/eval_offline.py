#!/usr/bin/env python3
"""Offline FLORES eval: same prompts, decoding, and chrF scoring as
eval_llama.py, but reads (id, src, ref) pairs from an existing records
JSONL instead of streaming the gated Hub dataset.

Use local pairs you already have permission to use; no new dataset download
is needed. To compare runs, keep the same IDs, source/reference text, prompts
and generation settings.

Usage (from the repository root, with a running model server):
  python evaluation/eval_offline.py --pairs logs/eval-en-ro-devtest.jsonl \\
      --model socket-lab --run_id eos-47a09863-q8-b11146 --src en --tgt ro \\
      --base_url http://localhost:8081 \\
      --output logs/my-eostune-en-ro.jsonl
"""
import argparse
from pathlib import Path

from eval_llama import (
    append_record,
    build_meta,
    build_prompt,
    check_server,
    first_line,
    open_records,
    post_completion,
    report_chrf,
    load_pairs,
    validate_args,
)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.set_defaults(strategy="direct", pivot=None)
    ap.add_argument("--pairs", required=True, help="records JSONL to read (id/src/ref)")
    ap.add_argument("--model", required=True, help="model alias served by llama.cpp")
    ap.add_argument("--run_id", required=True, help="Stable identity of weights, adapters and backend")
    ap.add_argument("--src", default="ro")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--request_timeout", type=float, default=300.0)
    ap.add_argument("--base_url", default="http://localhost:8081")
    ap.add_argument("--output", required=True, help="fresh records JSONL to write")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    validate_args(ap, args)
    pairs = load_pairs(Path(args.pairs), src=args.src, tgt=args.tgt, split=args.split)
    meta = build_meta(args)
    output = Path(args.output)
    check_server(args.base_url.rstrip("/"), args.model, args.request_timeout)
    stream, completed = open_records(output, meta, args.overwrite, expected_pairs=pairs)

    done = 0
    with stream:
        for row_id, source, reference in pairs:
            if row_id in completed:
                continue
            prompt = build_prompt(args.src, args.tgt, source)
            hyp = first_line(post_completion(
                args.base_url.rstrip("/"), args.model, prompt,
                args.max_new_tokens, args.request_timeout))
            append_record(stream, {"id": row_id, "src": source, "hyp": hyp, "ref": reference})
            done += 1
            print(f"  {done}", flush=True)

    report_chrf(args.src, args.tgt, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
