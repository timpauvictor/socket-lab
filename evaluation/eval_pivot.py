#!/usr/bin/env python3
"""Two-leg pivot eval: src -> pivot -> tgt through one or two servers.

Each pair row supplies (id, src_text, ref_text). Leg 1 translates
src -> pivot; leg 2 translates leg-1 output pivot -> tgt. The final
hypothesis is scored against the reference with the standard chrF
signature. Same prompts, decoding, and resume mechanics as eval_llama.

Usage (from the repository root, with a running model server):
  python evaluation/eval_pivot.py --pairs results/pairs-de-fr-devtest.jsonl \\
      --model socket-lab --run_id base-q8-b11146 --src de --pivot en --tgt fr \\
      --base_url http://localhost:8081 --output logs/my-pivot-de-fr.jsonl

For language cartridges, serve src+pivot on --base_url and pivot+tgt on
--second_base_url. Both servers must expose the same --model alias. Without
--second_base_url, both legs use the original single-server behavior.
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
    read_meta,
    validate_args,
)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs JSONL to read (id/src/ref)")
    ap.add_argument("--model", required=True, help="model alias served by llama.cpp")
    ap.add_argument("--run_id", required=True, help="Stable identity of weights, adapters and backend")
    ap.set_defaults(strategy="pivot")
    ap.add_argument("--src", default="de")
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--tgt", default="fr")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--request_timeout", type=float, default=300.0)
    ap.add_argument("--base_url", default="http://localhost:8081")
    ap.add_argument("--second_base_url", help="second-leg server URL; defaults to --base_url")
    ap.add_argument("--output", required=True, help="fresh records JSONL to write")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    validate_args(ap, args)
    if args.second_base_url is not None and not args.second_base_url.strip():
        ap.error("--second_base_url must not be empty")
    first_url = args.base_url.rstrip("/")
    second_url = args.second_base_url.rstrip("/") if args.second_base_url is not None else first_url
    pairs = load_pairs(Path(args.pairs), src=args.src, tgt=args.tgt, split=args.split)
    meta = build_meta(args)
    output = Path(args.output)
    if args.second_base_url is not None:
        meta["second_base_url"] = second_url
    elif output.exists() and not args.overwrite and "second_base_url" in read_meta(output):
        raise RuntimeError("Records used --second_base_url; provide it again or use a fresh --output")
    check_server(first_url, args.model, args.request_timeout)
    if second_url != first_url:
        check_server(second_url, args.model, args.request_timeout)
    intermediate_field = "hyp_en" if args.pivot == "en" else "hyp_pivot"
    stream, completed = open_records(output, meta, args.overwrite, expected_pairs=pairs,
                                     fields=("id", "src", "hyp", "ref", intermediate_field))

    done = 0
    with stream:
        for row_id, source, reference in pairs:
            if row_id in completed:
                continue
            leg1 = build_prompt(args.src, args.pivot, source)
            mid = first_line(post_completion(
                first_url, args.model, leg1,
                args.max_new_tokens, args.request_timeout))
            leg2 = build_prompt(args.pivot, args.tgt, mid)
            hyp = first_line(post_completion(
                second_url, args.model, leg2,
                args.max_new_tokens, args.request_timeout))
            append_record(stream, {"id": row_id, "src": source,
                                   intermediate_field: mid,
                                   "hyp": hyp, "ref": reference})
            done += 1
            print(f"  {done}", flush=True)

    report_chrf(args.src, args.tgt, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
