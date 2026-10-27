"""Test the shared data boundary without loading models or datasets."""
from functools import partial

import pytest

from training.translation_data import (
    find_config, prepare_control_pairs, prepare_pairs, tokenize_completion, training_prompt, translation_examples,
)


class CharacterTokenizer:
    eos_token_id = 999
    pad_token_id = 999

    def __call__(self, text, *, truncation, max_length, return_offsets_mapping=False):
        if isinstance(text, list):
            rows = [self(item, truncation=truncation, max_length=max_length,
                         return_offsets_mapping=return_offsets_mapping) for item in text]
            return {key: [row[key] for row in rows] for key in rows[0]}
        result = {"input_ids": [ord(char) for char in text[:max_length]]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(min(len(text), max_length))]
        return result


@pytest.mark.parametrize("eos_stop", [True, False])
def test_completion_labels_and_long_prompt_rejection(eos_stop):
    tokenize = partial(tokenize_completion, tokenizer=CharacterTokenizer(), eos_stop=eos_stop)
    result = tokenize({"prompt": "P:", "completion": "yes"})
    expected = [ord("y"), ord("e"), ord("s")] + ([999] if eos_stop else [])
    assert [value for value in result["labels"] if value != -100] == expected
    assert len(result["input_ids"]) == len(result["labels"]) == 256
    assert result["attention_mask"][:5] == [1] * 5
    assert result["attention_mask"][5 + int(eos_stop):] == [0] * (251 - int(eos_stop))
    with pytest.raises(ValueError, match="no completion tokens"):
        tokenize({"prompt": "P" * 300, "completion": "yes"})


def test_eos_reserves_one_position_and_requires_a_token():
    tokenizer = CharacterTokenizer()
    result = tokenize_completion({"prompt": "P", "completion": "a" * 300},
                                 tokenizer=tokenizer, eos_stop=True)
    assert result["input_ids"] == [ord("P")] + [ord("a")] * 254 + [999]
    assert result["labels"] == [-100] + [ord("a")] * 254 + [999]
    tokenizer.eos_token_id = None
    with pytest.raises(SystemExit, match="no EOS token"):
        tokenize_completion({"prompt": "P", "completion": "a"}, tokenizer=tokenizer, eos_stop=True)


@pytest.mark.parametrize("completion", ["yes", "a" * 300])
def test_paired_controls_match_content_labels_and_truncation(completion):
    example = {"prompt": "P:", "completion": completion}
    off = tokenize_completion(example, tokenizer=CharacterTokenizer(), paired_control=True)
    on = tokenize_completion(example, tokenizer=CharacterTokenizer(), paired_control=True, eos_stop=True)
    length = sum(off["attention_mask"])
    assert off["input_ids"][:length] == on["input_ids"][:length]
    assert off["labels"][:length] == on["labels"][:length]
    assert on["input_ids"][length] == on["labels"][length] == 999
    assert off["attention_mask"][length] == 0 and off["labels"][length] == -100
    assert on["attention_mask"][length] == 1
    assert length <= 255
    if len(completion) > 255:
        assert length == 255


def test_paired_boundary_token_keeps_short_completion_without_changing_legacy_mask():
    class SpaceMergingTokenizer(CharacterTokenizer):
        def __call__(self, text, *, truncation, max_length, return_offsets_mapping=False):
            assert text in ("P: ", "P: yes")
            result = {"input_ids": [80, 58, 220 if text == "P: " else 1234]}
            if return_offsets_mapping:
                result["offset_mapping"] = [(0, 1), (1, 2), (2, len(text))]
            return result
    example = {"prompt": "P: ", "completion": "yes"}
    tokenizer = SpaceMergingTokenizer()
    for eos_stop in (False, True):
        result = tokenize_completion(example, tokenizer=tokenizer, paired_control=True, eos_stop=eos_stop)
        assert [v for v in result["labels"] if v != -100] == [1234] + ([999] if eos_stop else [])
    with pytest.raises(ValueError, match="no completion tokens"):
        tokenize_completion(example, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="no completion tokens"):
        tokenize_completion({"prompt": "P" * 300, "completion": "yes"},
                            tokenizer=CharacterTokenizer(), paired_control=True)


def test_control_pair_preparation_excludes_both_directions_without_backfill():
    rows = [("one", "unu"), ("x" * 200, "y" * 100), ("two", "doi"), ("a" * 100, "z" * 200)]
    original = rows.copy()
    kept, excluded = prepare_control_pairs("ro", "en", rows, CharacterTokenizer())
    assert rows == original
    assert kept == [rows[0], rows[2]]
    assert [entry["selected_pair_index"] for entry in excluded] == [1, 3]
    assert [entry["directions"] for entry in excluded] == [[0], [1]]
    for prompt, completion in translation_examples("ro", "en", kept):
        assert any(v != -100 for v in tokenize_completion({"prompt": prompt, "completion": completion},
            tokenizer=CharacterTokenizer(), paired_control=True)["labels"])


def test_pair_preparation_filters_deduplicates_then_shuffles_and_caps():
    class ReverseOrder:
        def shuffle(self, rows):
            # Deduplication and filtering must happen before shuffling.
            assert rows == [("one", "unu"), ("two", "doi"), ("ten", "zece")]
            rows.reverse()

    pairs = [(" one ", " unu "), ("one", "unu"), ("", "absent"),
             ("a" * 401, "b" * 200), ("long phrase", "x"),
             ("two", "doi"), ("ten", "zece")]
    dataset = [{"translation": {"en": a, "ro": b}} for a, b in pairs]
    assert prepare_pairs(dataset, "en", "ro", 2, ReverseOrder()) == [("ten", "zece"), ("two", "doi")]


def test_direction_order_prompt_spacing_and_config_selection():
    assert translation_examples("en", "ro", [("hello", "salut")]) == [
        ("Translate the following English text to Romanian.\nEnglish: hello\nRomanian: ", "salut"),
        ("Translate the following Romanian text to English.\nRomanian: salut\nEnglish: ", "hello"),
    ]
    assert training_prompt("en", "ro", "hello").endswith("Romanian: ")
    assert find_config("en", "ro", ["ro-en", "en-ro"]) == "en-ro"
    assert find_config("en", "ro", ["ro-en"]) == "ro-en"
    with pytest.raises(SystemExit, match="No opus-100 config"):
        find_config("en", "ro", ["de-en"])
