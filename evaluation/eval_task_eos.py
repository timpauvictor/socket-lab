#!/usr/bin/env python3
"""Offline-pair BF16 Transformers evaluation for the controlled EOS experiment."""
import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud.modal.artifacts import hash_file
from evaluation.eval_llama import (PROMPT_VERSION, append_record, build_prompt, first_line,
                                   iter_records, load_pairs, open_records, read_meta)
from evaluation.repetition import has_loop
from training.task_model import build_task_model, validate_adapter


def token_evidence(ids: list[int], eos_ids: list[int], cap: int) -> dict:
    if (not isinstance(ids, list) or not ids or len(ids) > cap
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("Invalid, empty or over-cap generation")
    hits = [i for i, token in enumerate(ids) if token in eos_ids]
    if hits and hits != [len(ids) - 1]:
        raise ValueError("Unexpected tokens after EOS")
    if not hits and len(ids) != cap:
        raise ValueError("Generation stopped without EOS or token cap")
    return {"generated_token_ids": ids, "generated_tokens": len(ids),
            "eos_positions": hits, "finish_reason": "eos" if hits else "token_cap"}


def validate_rows(path: Path, meta: dict, pairs: list[tuple[str, str, str]]) -> None:
    expected = {row_id: (source, reference) for row_id, source, reference in pairs}
    for row in iter_records(path, fields=("id", "src", "ref")):
        if expected.get(row["id"]) != (row["src"], row["ref"]):
            raise RuntimeError("Evaluation resume input identity mismatch")
        if "error" in row:
            if not isinstance(row["error"], dict) or not isinstance(row["error"].get("type"), str):
                raise RuntimeError("Invalid error record")
            if any(k in row for k in ("hyp", "raw", "finish_reason", "looped")):
                raise RuntimeError("Failed generation contains success evidence")
            continue
        evidence = token_evidence(row["generated_token_ids"], meta["eos_token_ids"], meta["max_new_tokens"])
        if any(row.get(k) != v for k, v in evidence.items()):
            raise RuntimeError("Invalid generation evidence")
        if row.get("hyp") != first_line(row["raw"]) or row.get("looped") != has_loop(row["raw"]):
            raise RuntimeError("Invalid hypothesis/repetition evidence")


def summarize(path: Path) -> dict:
    import sacrebleu
    from sacrebleu.metrics import CHRF

    if sacrebleu.__version__ != "2.6.0":
        raise RuntimeError("This experiment requires sacrebleu==2.6.0")
    meta = read_meta(path)
    rows = list(iter_records(path, fields=("id", "src", "ref")))
    good = [r for r in rows if "error" not in r]
    total = meta["expected_examples"]
    failed = len(rows) - len(good)
    complete = len(rows) == total and failed == 0
    metric = CHRF()
    score = metric.corpus_score([r["hyp"] for r in good], [[r["ref"] for r in good]]) if good else None
    lengths = [r["generated_tokens"] for r in good]
    return {"arm": meta["arm"], "direction": f'{meta["src"]}-{meta["tgt"]}',
            "expected": total, "attempted": len(rows), "successful": len(good), "failed": failed,
            "missing": total - len(rows), "complete": complete,
            "chrf": score.score if complete else None,
            "partial_chrf": score.score if score and not complete else None,
            "signature": str(metric.get_signature()) if score else None,
            "repetition_fraction": sum(r["looped"] for r in good) / total if complete else None,
            "eos_stop_rate": sum(r["finish_reason"] == "eos" for r in good) / total if complete else None,
            "token_cap_rate": sum(r["finish_reason"] == "token_cap" for r in good) / total if complete else None,
            "repeating_successes": sum(r["looped"] for r in good),
            "eos_successes": sum(r["finish_reason"] == "eos" for r in good),
            "cap_successes": sum(r["finish_reason"] == "token_cap" for r in good),
            "length": {"min": min(lengths), "max": max(lengths), "mean": statistics.mean(lengths),
                       "median": statistics.median(lengths)} if lengths else None}


def evaluate(args: argparse.Namespace) -> int:
    if args.output.resolve() == args.pairs.resolve() or (args.output.exists() and args.output.samefile(args.pairs)):
        raise ValueError("Pair source and output must differ")
    pairs = load_pairs(SimpleNamespace(pairs=args.pairs, src=args.src, tgt=args.tgt, split="devtest"))
    if len(pairs) != 1012:
        raise ValueError("Expected exactly 1,012 frozen FLORES pairs")
    pairs = pairs[:args.limit] if args.limit else pairs
    boxes = None
    # Preserve TRAINING pair order even for en->ro: linear factors must match.
    if args.arm != "D":
        other = "ro" if "ro" in (args.src, args.tgt) else "de"
        boxes = (str(args.inputs / "adapters" / f"box-{other}"), str(args.inputs / "adapters/box-en"))
    adapter_files = [args.adapter / name for name in ("adapter_config.json", "adapter_model.safetensors")]
    for box in boxes or ():
        validate_adapter(box, args.model)
        adapter_files.extend(Path(box) / name for name in ("adapter_config.json", "adapter_model.safetensors"))
    validate_adapter(args.adapter, args.model, 32)
    from transformers import AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_ids = [tok.eos_token_id]
    meta = {"_meta": True, "arm": args.arm, "src": args.src, "tgt": args.tgt, "split": "devtest",
            "model": args.model, "revision": args.revision, "tokenizer_revision": args.revision,
            "backend": "transformers/peft", "dtype": "bfloat16", "prompt_version": PROMPT_VERSION,
            "stop_policy": "eos_or_cap_no_newline_stop", "eos_token_ids": eos_ids,
            "pad_token_id": tok.pad_token_id, "decode_skip_special_tokens": True,
            "max_new_tokens": 128, "do_sample": False, "num_beams": 1,
            "composition": "peft-linear-1+1" if boxes else "monolith",
            "adapter_files": {str(p.resolve()): hash_file(p)[0] for p in adapter_files},
            "pairs_sha256": hash_file(args.pairs)[0], "expected_examples": len(pairs),
            "run_id": args.run_id}
    if args.output.exists():
        if not args.resume:
            raise ValueError("Fresh output required; matching-only --resume is explicit")
        if read_meta(args.output) != meta:
            raise RuntimeError("Evaluation resume metadata mismatch")
        validate_rows(args.output, meta, pairs)
    stream, completed = open_records(args.output, meta, False)
    with stream:
        model = None
        if len(completed) < len(pairs):
            model = build_task_model(args.model, revision=args.revision, boxes=boxes,
                                     task_adapter=str(args.adapter)).to("cuda")
            model.eval()
        for row_id, source, reference in pairs:
            if row_id in completed:
                continue
            row = {"id": row_id, "src": source, "ref": reference}
            try:
                inputs = tok(build_prompt(args.src, args.tgt, source), return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=128,
                        eos_token_id=eos_ids, pad_token_id=tok.pad_token_id,
                        repetition_penalty=1.0, no_repeat_ngram_size=0,
                        forced_eos_token_id=None, min_new_tokens=0, min_length=0,
                        forced_bos_token_id=None, suppress_tokens=None, begin_suppress_tokens=None,
                        stop_strings=None, num_return_sequences=1, max_time=None)
                ids = output[0, inputs["input_ids"].shape[1]:].tolist()
                evidence = token_evidence(ids, eos_ids, 128)
                raw = tok.decode(ids, skip_special_tokens=True)
                row.update(evidence, raw=raw, hyp=first_line(raw), looped=has_loop(raw))
            except Exception as exc:
                # Failure is an attempted example, never a valid empty/nonrepeating translation.
                row["error"] = {"type": type(exc).__name__}
                append_record(stream, row)
                raise RuntimeError(f"Generation failed for id {row_id}; partial file retained") from exc
            append_record(stream, row)
            print(f"{args.arm} {args.src}-{args.tgt}: {row_id} {row['finish_reason']}", flush=True)
    result = summarize(args.output)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["complete"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=list("ABCD"))
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=0, choices=(0, 2), help="2 = disposable smoke only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (args.src, args.tgt) not in (("en", "ro"), ("ro", "en"), ("de", "en")):
        parser.error("Only the frozen three directions are supported")
    try:
        return evaluate(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
