"""Shared OPUS pair preparation and completion-only tokenization.

No model/dataset imports: callers own downloads, RNG seeding and training.
"""
from collections.abc import Iterable, Mapping
from typing import Any

LANG_NAMES = {"ro": "Romanian", "en": "English", "fr": "French", "de": "German"}
MAX_TRAIN_TOKENS = 256
PAIRED_CONTENT_TOKENS = MAX_TRAIN_TOKENS - 1


def find_config(src: str, tgt: str, names: list[str]) -> str:
    for name in (f"{src}-{tgt}", f"{tgt}-{src}"):
        if name in names:
            return name
    raise SystemExit(f"No opus-100 config for {src}-{tgt}; sample of available: {names[:8]}")


def prepare_pairs(dataset: Iterable[Mapping], src: str, tgt: str, max_pairs: int,
                  rng: Any, *, progress_every: int = 0) -> list[tuple[str, str]]:
    rows = []
    for scanned, row in enumerate(dataset, 1):
        if progress_every and scanned % progress_every == 0:
            print(f"  scanned {scanned} rows for {src}-{tgt}, kept {len(rows)}", flush=True)
        translation = row["translation"]
        a, b = (translation.get(src) or "").strip(), (translation.get(tgt) or "").strip()
        if not a or not b or len(a) > 400 or len(b) > 400:
            continue
        if max(len(a), len(b)) / max(1, min(len(a), len(b))) > 3:
            continue
        rows.append((a, b))
    rows = list(dict.fromkeys(rows))
    rng.shuffle(rows)
    return rows[:max_pairs]


def training_prompt(src: str, tgt: str, sentence: str) -> str:
    # The trailing space is intentional; saved llama.cpp evaluation omits it.
    return (f"Translate the following {LANG_NAMES[src]} text to {LANG_NAMES[tgt]}.\n"
            f"{LANG_NAMES[src]}: {sentence}\n{LANG_NAMES[tgt]}: ")


def translation_examples(src: str, tgt: str, rows: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    examples = []
    for source, target in rows:
        examples.append((training_prompt(src, tgt, source), target))
        examples.append((training_prompt(tgt, src, target), source))
    return examples


def completion_boundary(offsets: list[tuple[int, int]], prompt_chars: int) -> int:
    return next((i for i, (_, end) in enumerate(offsets) if end > prompt_chars), len(offsets))


def prepare_control_pairs(src: str, tgt: str, rows: list[tuple[str, str]],
                          tokenizer: Any) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Exclude a whole selected pair if either direction cannot supervise content.

    The original selection/order is retained separately by the caller. No backfill
    or content rewriting occurs. Both controls consume this one frozen result.
    """
    kept = []
    excluded = []
    for start in range(0, len(rows), 512):
        batch = rows[start:start + 512]
        examples = translation_examples(src, tgt, batch)
        encoded = tokenizer([p + c for p, c in examples], truncation=True,
                            max_length=PAIRED_CONTENT_TOKENS, return_offsets_mapping=True)
        invalid = {}
        for index, ((prompt, _), ids, offsets) in enumerate(zip(
                examples, encoded["input_ids"], encoded["offset_mapping"], strict=True)):
            if completion_boundary(offsets, len(prompt)) >= len(ids):
                invalid.setdefault(index // 2, []).append(index % 2)
        for index, row in enumerate(batch):
            if index in invalid:
                excluded.append({"selected_pair_index": start + index, "directions": invalid[index],
                                 "reason": "no_completion_within_paired_content_budget"})
            else:
                kept.append(row)
        if start % 25600 == 0:
            print(f"  token-budget check {src}-{tgt}: {start + len(batch)}/{len(rows)} pairs; "
                  f"{len(excluded)} excluded", flush=True)
    return kept, excluded


def tokenize_completion(example: Mapping[str, str], *, tokenizer: Any,
                        eos_stop: bool = False,
                        paired_control: bool = False) -> dict[str, list[int]]:
    max_len = MAX_TRAIN_TOKENS
    # EOS is an explicit supervised target only for the opt-in tune-up.
    eos = [tokenizer.eos_token_id] if eos_stop else []
    if eos_stop and tokenizer.eos_token_id is None:
        raise SystemExit("tokenizer has no EOS token; --eos_stop needs one")
    content_budget = max_len - int(eos_stop or paired_control)
    if paired_control:
        # A boundary token can contain both the prompt's trailing space and the
        # first completion word. Standalone prompt length would mask that word.
        encoded = tokenizer(example["prompt"] + example["completion"], truncation=True,
                            max_length=content_budget, return_offsets_mapping=True)
        full_ids = encoded["input_ids"]
        plen = completion_boundary(encoded["offset_mapping"], len(example["prompt"]))
    else:
        # Preserve the historical masking recipe for existing callers.
        full_ids = tokenizer(example["prompt"] + example["completion"],
                             truncation=True, max_length=content_budget)["input_ids"]
        plen = len(tokenizer(example["prompt"], truncation=True, max_length=max_len)["input_ids"])
    full_ids = full_ids + eos
    plen = min(plen, len(full_ids))
    if plen >= len(full_ids) - len(eos):
        raise ValueError("Prompt leaves no completion tokens within the 256-token budget")
    labels = [-100] * plen + full_ids[plen:]
    pad_len = max_len - len(full_ids)
    return {"input_ids": full_ids + [tokenizer.pad_token_id] * pad_len,
            "attention_mask": [1] * len(full_ids) + [0] * pad_len,
            "labels": labels + [-100] * pad_len}
