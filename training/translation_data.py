"""Shared OPUS pair preparation and completion-only tokenization.

No model/dataset imports: callers own downloads, RNG seeding and training.
"""
from collections.abc import Iterable, Mapping
from typing import Any

LANG_NAMES = {"ro": "Romanian", "en": "English", "fr": "French", "de": "German"}


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


def tokenize_completion(example: Mapping[str, str], *, tokenizer: Any,
                        eos_stop: bool = False) -> dict[str, list[int]]:
    max_len = 256
    # EOS is an explicit supervised target only for the opt-in tune-up.
    eos = [tokenizer.eos_token_id] if eos_stop else []
    if eos_stop and tokenizer.eos_token_id is None:
        raise SystemExit("tokenizer has no EOS token; --eos_stop needs one")
    full_ids = tokenizer(example["prompt"] + example["completion"],
                         truncation=True, max_length=max_len - len(eos))["input_ids"]
    full_ids = full_ids + eos
    plen = len(tokenizer(example["prompt"], truncation=True, max_length=max_len)["input_ids"])
    plen = min(plen, len(full_ids))
    if plen >= len(full_ids) - len(eos):
        raise ValueError("Prompt leaves no completion tokens within the 256-token budget")
    labels = [-100] * plen + full_ids[plen:]
    pad_len = max_len - len(full_ids)
    return {"input_ids": full_ids + [tokenizer.pad_token_id] * pad_len,
            "attention_mask": [1] * len(full_ids) + [0] * pad_len,
            "labels": labels + [-100] * pad_len}
