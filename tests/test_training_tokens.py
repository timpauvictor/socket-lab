"""Test the shared data boundary without loading models or datasets."""
from functools import partial

import pytest

from training.translation_data import (
    find_config, prepare_pairs, tokenize_completion, training_prompt, translation_examples,
)


class CharacterTokenizer:
    eos_token_id = 999
    pad_token_id = 999

    def __call__(self, text, *, truncation, max_length):
        return {"input_ids": [ord(char) for char in text[:max_length]]}


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
