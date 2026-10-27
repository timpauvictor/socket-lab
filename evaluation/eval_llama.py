#!/usr/bin/env python3
"""Translation eval through a user-started llama.cpp server. Reports chrF.

Quick smoke test (server must already be running):
  python evaluation/eval_llama.py --model socket-lab --run_id base-q8-b11146 --src ro --tgt en --limit 10

Full floor assay:
  python evaluation/eval_llama.py --model socket-lab --run_id base-q8-b11146 --src ro --tgt en
  python evaluation/eval_llama.py --model socket-lab --run_id base-q8-b11146 --src en --tgt ro

Score an existing records file without touching the server:
  python evaluation/eval_llama.py --score_only --output logs/eval-ro-en-devtest.jsonl

Uses raw /v1/completions for the Base model, without a chat template.
Dataset access is needed only for new streamed inference. Score-only mode
reads saved hypotheses and references; it needs neither a server nor datasets.

Completed segments are flushed to JSONL. Resume validates run metadata and
records; a partial or corrupt write requires inspection, not silent repair.
"""
import argparse
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

if __package__:
    from .prompts import translation_prompt
else:
    from prompts import translation_prompt

LANG_NAMES = {"ro": "Romanian", "en": "English", "fr": "French", "de": "German", "fi": "Finnish"}
# FLORES+ uses ISO 639-3 codes on the Hub
FLORES_CODES = {"ro": "ron_Latn", "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "fi": "fin_Latn"}
FLORES_DATASET = "openlanguagedata/flores_plus"
# Bump when the prompt format changes so stale records files stop a re-run
# instead of silently mixing scores across prompt versions.
PROMPT_VERSION = 1


def build_prompt(src: str, tgt: str, sentence: str) -> str:
    return translation_prompt(LANG_NAMES[src], LANG_NAMES[tgt], sentence)


def first_line(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def request_text(url: str, payload: dict[str, object] | None, timeout_seconds: float) -> tuple[int, str]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP request failed: {url} status {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HTTP request failed: {url}: {exc.reason}") from exc


def check_server(base_url: str, model: str, timeout_seconds: float) -> None:
    status, body = request_text(f"{base_url}/health", None, timeout_seconds)
    if status != 200:
        raise RuntimeError(f"Server health check failed: {base_url}/health status {status}")
    try:
        health = json.loads(body)
    except json.JSONDecodeError:
        health = None
    if isinstance(health, dict):
        state = health.get("status")
        if state is not None and state != "ok":
            raise RuntimeError(f"Server is not ready: {body[:200]}")

    status, body = request_text(f"{base_url}/v1/models", None, timeout_seconds)
    if status != 200:
        raise RuntimeError(f"Could not list server models: status {status}")
    try:
        models = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Server models response was not JSON") from exc
    entries = models.get("data", []) if isinstance(models, dict) else []
    available = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    if model not in available:
        raise RuntimeError(f"Model {model!r} is not served. Available: {available}")


def row_id_text(row: object, source_label: str) -> tuple[str, str]:
    if not isinstance(row, dict):
        raise RuntimeError(f"FLORES+ row from {source_label} has unexpected shape")
    row_id = row.get("id")
    text = row.get("text")
    if isinstance(row_id, int):
        row_id = str(row_id)
    if not isinstance(row_id, str) or not row_id:
        raise RuntimeError(f"FLORES+ row from {source_label} is missing id")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"FLORES+ row {row_id!r} from {source_label} is missing text")
    return row_id, text


def stream_aligned_pairs(src_code: str, tgt_code: str, split: str) -> Iterator[tuple[str, str, str]]:
    """Yield (id, source, reference) pairs, one at a time, nothing accumulated.

    The two per-language streams are consumed in lockstep: FLORES+ language
    files are parallel by construction, so the nth row of each file is the
    same sentence. A mismatched id fails fast instead of scoring garbage.
    """
    from datasets import load_dataset

    try:
        src_rows = load_dataset(FLORES_DATASET, src_code, split=split, streaming=True)
        tgt_rows = load_dataset(FLORES_DATASET, tgt_code, split=split, streaming=True)
        row_pairs = zip(src_rows, tgt_rows, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not stream openlanguagedata/flores_plus. Accept the dataset terms on "
            "https://huggingface.co/datasets/openlanguagedata/flores_plus, then run "
            "`hf auth login`.\n"
            f"Original error: {exc}"
        ) from exc

    for src_row, tgt_row in row_pairs:
        src_id, source = row_id_text(src_row, src_code)
        tgt_id, reference = row_id_text(tgt_row, tgt_code)
        if src_id != tgt_id:
            raise RuntimeError(
                f"FLORES+ streams diverged: {src_code} id {src_id!r} != "
                f"{tgt_code} id {tgt_id!r}"
            )
        yield src_id, source, reference


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("src", "tgt", "pivot"):
        value = getattr(args, name, None)
        if value is not None and value not in LANG_NAMES:
            ap.error(f"unsupported {name}: {value}")
    if args.max_new_tokens <= 0:
        ap.error("--max_new_tokens must be positive")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        ap.error("--request_timeout must be finite and positive")
    if not args.model or not args.run_id.strip():
        ap.error("inference requires --model and --run_id (identify weights, adapters and backend)")
    pairs = getattr(args, "pairs", None)
    if pairs:
        source, output = Path(pairs), Path(args.output)
        if source.resolve() == output.resolve() or (output.exists() and source.samefile(output)):
            ap.error("--pairs and --output must be different files")


def build_meta(args: argparse.Namespace) -> dict[str, object]:
    meta = {
        "_meta": True,
        "dataset": FLORES_DATASET,
        "src": args.src,
        "tgt": args.tgt,
        "split": args.split,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "strategy": getattr(args, "strategy", "direct"),
        "pivot": getattr(args, "pivot", None),
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0,
        "stop": [] if getattr(args, "strategy", "direct") == "scratchpad" else ["\n"],
        "run_id": args.run_id,
        "pairs_sha256": None,
    }
    if getattr(args, "pairs", None):
        meta["pairs_sha256"] = hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest()
    return meta


def read_meta(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        try:
            header = json.loads(stream.readline())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Records file has a corrupt header: {path}") from exc
    if not isinstance(header, dict) or header.get("_meta") is not True:
        raise RuntimeError(f"Records file has no meta header: {path}")
    return header


def open_records(path: Path, meta: dict[str, object], overwrite: bool) -> tuple[object, set[str]]:
    """Open the records file for appending; return (stream, completed ids).

    Validate stored rows while retaining only IDs. Reject incompatible metadata
    rather than mixing configurations.
    """
    if path.exists() and not overwrite:
        stored = read_meta(path)
        for key, value in meta.items():
            if key != "_meta" and (key not in stored or stored[key] != value):
                raise RuntimeError(
                    f"Records file {path} was made for {key}={stored.get(key)!r}, "
                    f"not {value!r}; use a fresh --output (legacy records remain scoreable)"
                )
        completed = {record["id"] for record in iter_records(path)}
        with path.open("rb") as raw:
            raw.seek(-1, 2)
            if raw.read(1) != b"\n":
                raise RuntimeError(f"Records file has an unfinished last line: {path}; inspect before resuming")
        stream = path.open("a", encoding="utf-8")
        return stream, completed

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("w" if overwrite else "x", encoding="utf-8")
    stream.write(json.dumps(meta, ensure_ascii=False) + "\n")
    stream.flush()
    return stream, set()


def append_record(stream: object, record: dict[str, object]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
    stream.flush()  # type: ignore[union-attr]


def iter_records(path: Path, fields: tuple[str, ...] = ("id", "src", "hyp", "ref")) -> Iterator[dict]:
    """Read records without repairing corruption or counting duplicate IDs."""
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(record, dict) and record.get("_meta") is True and line_number == 1:
                continue
            if not isinstance(record, dict) or any(not isinstance(record.get(key), str) for key in fields):
                raise RuntimeError(f"Invalid record fields at {path}:{line_number}")
            row_id = record["id"]
            if not row_id or row_id in seen:
                raise RuntimeError(f"Empty or duplicate id {row_id!r} at {path}:{line_number}")
            seen.add(row_id)
            yield record


def iter_pairs(path: Path) -> Iterator[tuple[str, str, str]]:
    for record in iter_records(path, fields=("id", "src", "ref")):
        yield record["id"], record["src"], record["ref"]


def load_pairs(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    path = Path(args.pairs)
    with path.open(encoding="utf-8") as stream:
        first = json.loads(stream.readline())
    if isinstance(first, dict) and first.get("_meta") is True:
        for key in ("src", "tgt", "split"):
            if first.get(key) != getattr(args, key):
                raise RuntimeError(f"Pair file {path}: {key} does not match the requested evaluation")
    pairs = list(iter_pairs(path))
    if not pairs:
        raise RuntimeError(f"No pairs in {path}")
    return pairs


def report_chrf(src: str, tgt: str, records_path: Path) -> int:
    from sacrebleu.metrics import CHRF

    hypotheses: list[str] = []
    references: list[str] = []
    samples: list[dict[str, str]] = []
    for record in iter_records(records_path):
        hypotheses.append(record["hyp"])
        references.append(record["ref"])
        if len(samples) < 3:
            samples.append(record)
    if not hypotheses:
        raise RuntimeError(f"No completed records in {records_path}")
    metric = CHRF()
    score = metric.corpus_score(hypotheses, [references])
    print(f"\nchrF ({src}->{tgt}, n={len(hypotheses)}): {score.score:.2f}")
    print(f"signature: {metric.get_signature()}")
    for record in samples:
        print(f"\nSRC: {record['src'][:120]}")
        print(f"HYP: {record['hyp'][:120]}")
        print(f"REF: {record['ref'][:120]}")
    return len(hypotheses)


def post_completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout_seconds: float, stop: list[str] | None = None) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stop": ["\n"] if stop is None else stop,
        "stream": False,
    }
    status, body = request_text(f"{base_url}/v1/completions", payload, timeout_seconds)
    if status != 200:
        raise RuntimeError(f"Completion failed: status {status}: {body[:500]}")
    try:
        completion = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Completion response was not JSON") from exc

    choices = completion.get("choices") if isinstance(completion, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"Completion response has no usable choices: {body[:500]}")
    text = choices[0].get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"Completion choice has no text: {body[:500]}")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:8081")
    ap.add_argument("--model", help="Model alias served by llama.cpp; not needed for --score_only")
    ap.add_argument("--run_id", default="", help="Stable identity of weights, adapters and backend; required for inference")
    ap.add_argument("--src")
    ap.add_argument("--tgt")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="eval only first N (0 = all)")
    ap.add_argument("--request_timeout", type=float, default=300.0)
    ap.add_argument("--output", default="",
                    help="records JSONL path (default logs/eval-<src>-<tgt>-<split>.jsonl)")
    ap.add_argument("--overwrite", action="store_true",
                    help="discard existing records and start fresh")
    ap.add_argument("--score_only", action="store_true",
                    help="score an existing records file without contacting the server")
    args = ap.parse_args()

    if args.limit < 0:
        ap.error("--limit must be 0 or greater")
    if args.score_only and not args.output:
        ap.error("--score_only requires --output")
    if not args.score_only:
        args.src, args.tgt = args.src or "ro", args.tgt or "en"
        validate_args(ap, args)

    base_url = args.base_url.rstrip("/")
    output = Path(args.output) if args.output else Path(f"logs/eval-{args.src}-{args.tgt}-{args.split}.jsonl")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass

    try:
        if args.score_only:
            if not output.exists():
                raise RuntimeError(f"No records file to score: {output}")
            meta = read_meta(output)
            for key in ("src", "tgt"):
                if getattr(args, key) is not None and getattr(args, key) != meta.get(key):
                    raise RuntimeError(f"--{key} does not match records metadata")
            report_chrf(str(meta["src"]), str(meta["tgt"]), output)
            return 0

        print(f"Checking server: {base_url}", flush=True)
        check_server(base_url, args.model, args.request_timeout)

        src_code, tgt_code = FLORES_CODES[args.src], FLORES_CODES[args.tgt]
        print(f"Streaming {FLORES_DATASET} {args.split}: {src_code} -> {tgt_code}", flush=True)
        stream, completed = open_records(output, build_meta(args), args.overwrite)
        print(f"Records: {output} ({len(completed)} already done)", flush=True)

        done = len(completed)
        with stream:
            for row_id, source, reference in stream_aligned_pairs(src_code, tgt_code, args.split):
                if row_id in completed:
                    continue
                if args.limit and done >= args.limit:
                    break
                prompt = build_prompt(args.src, args.tgt, source)
                hyp = first_line(post_completion(
                    base_url, args.model, prompt, args.max_new_tokens, args.request_timeout
                ))
                append_record(stream, {"id": row_id, "src": source, "hyp": hyp, "ref": reference})
                done += 1
                if args.limit:
                    print(f"  {done}/{args.limit}", flush=True)
                else:
                    print(f"  {done}", flush=True)

        report_chrf(args.src, args.tgt, output)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
