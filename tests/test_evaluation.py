import argparse
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

import eval_llama as evaluation
from loop_check import has_loop, summarize


def arguments(**changes):
    values = dict(src="de", tgt="fr", split="devtest", model="test-model",
                  run_id="base-test", max_new_tokens=128, request_timeout=30,
                  strategy="direct")
    values.update(changes)
    return argparse.Namespace(**values)


def write_records(path, records, meta=None):
    rows = ([meta] if meta else []) + records
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_sampling_probe_protects_existing_output_and_separates_failures(tmp_path, monkeypatch, capsys):
    import sampling_probe as probe
    path = tmp_path / "probe.jsonl"
    path.write_bytes(b"banked bytes\n")
    monkeypatch.setattr(probe, "complete", lambda *a: pytest.fail("request before output admission"))
    assert probe.main(["--out", str(path)]) == 1
    assert path.read_bytes() == b"banked bytes\n"
    with pytest.raises(SystemExit) as error:
        probe.main([])
    assert error.value.code == 2
    monkeypatch.setattr(probe, "PROMPTS", [("ro", "en", "Buna dimineata.")])
    def complete(base_url, prompt, temperature, penalty):
        if temperature == 0:
            raise OSError("connection failed")
        return "Good morning."
    monkeypatch.setattr(probe, "complete", complete)
    fresh = tmp_path / "mixed.jsonl"
    assert probe.main(["--out", str(fresh)]) == 1
    failed, success = [json.loads(line) for line in fresh.read_text().splitlines()]
    assert failed["error"] == {"type": "OSError", "message": "connection failed"}
    assert "hyp" not in failed and "looped" not in failed
    assert success["hyp"] == "Good morning." and success["looped"] is False
    assert "1 completions, 0 looped, 1 failed" in capsys.readouterr().out
    monkeypatch.setattr(probe, "complete", lambda *a: "Good morning.")
    assert probe.main(["--out", str(tmp_path / "success.jsonl")]) == 0


@pytest.mark.parametrize("payload", [{"content": "valid"}, {"content": None}, {}, [], {"content": 123}])
def test_sampling_probe_validates_completion_response(payload, monkeypatch):
    import sampling_probe as probe
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(payload).encode("utf-8")
    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *a, **k: Response())
    if payload == {"content": "valid"}:
        assert probe.complete("http://unused", "prompt", 0, 1.1) == "valid"
    else:
        with pytest.raises(ValueError, match="string content"):
            probe.complete("http://unused", "prompt", 0, 1.1)


def test_new_records_claim_does_not_truncate_a_racing_output(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path / "racing.jsonl"
    path.write_bytes(b"another writer's data\n")
    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if p == path else original(p))
    with pytest.raises(FileExistsError):
        evaluation.open_records(path, evaluation.build_meta(arguments()), False)
    assert path.read_bytes() == b"another writer's data\n"


def test_resume_matches_identity_and_keeps_existing_records(tmp_path):
    path = tmp_path / "records.jsonl"
    args = arguments()
    meta = evaluation.build_meta(args)
    stream, completed = evaluation.open_records(path, meta, False)
    assert completed == set()
    with stream:
        evaluation.append_record(stream, {"id": "1", "src": "A", "hyp": "B", "ref": "B"})
    before = path.read_bytes()
    stream, completed = evaluation.open_records(path, meta, False)
    stream.close()
    assert completed == {"1"}
    assert path.read_bytes() == before
    for change in ({"max_new_tokens": 256}, {"strategy": "pivot", "pivot": "en"},
                   {"run_id": "different-adapter"}, {"strategy": "scratchpad"}):
        with pytest.raises(RuntimeError, match="fresh --output"):
            evaluation.open_records(path, evaluation.build_meta(arguments(**change)), False)
        assert path.read_bytes() == before


def test_legacy_records_score_without_model_or_dataset_and_do_not_resume(tmp_path, monkeypatch, capsys):
    path = tmp_path / "old.jsonl"
    write_records(path, [{"id": "0", "src": "Hallo", "hyp": "bonjour", "ref": "bonjour"}],
                  {"_meta": True, "src": "de", "tgt": "fr", "split": "devtest",
                   "dataset": evaluation.FLORES_DATASET, "model": "test-model", "prompt_version": 1})
    monkeypatch.setattr(sys, "argv", ["eval_llama.py", "--score_only", "--output", str(path)])
    assert evaluation.main() == 0
    assert "de->fr, n=1): 100.00" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="fresh --output"):
        evaluation.open_records(path, evaluation.build_meta(arguments()), False)


@pytest.mark.parametrize("bad_tail", ['{"id":', '{"id":"0","src":"A","hyp":"B","ref":"B"}\n'])
def test_resume_rejects_partial_or_duplicate_rows_without_writing(tmp_path, bad_tail):
    path = tmp_path / "bad.jsonl"
    meta = evaluation.build_meta(arguments())
    write_records(path, [{"id": "0", "src": "A", "hyp": "B", "ref": "B"}], meta)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(bad_tail)
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        evaluation.open_records(path, meta, False)
    assert path.read_bytes() == before


@pytest.mark.parametrize("module_name", ["eval_offline", "eval_pivot", "eval_scratchpad"])
def test_pair_cli_rejects_alias_before_overwrite(tmp_path, monkeypatch, module_name):
    module = importlib.import_module(module_name)
    path = tmp_path / "pairs.jsonl"
    write_records(path, [{"id": "0", "src": "Hallo", "ref": "bonjour"}])
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", [module_name, "--pairs", str(path), "--output", str(path),
                                     "--model", "test-model", "--run_id", "test", "--overwrite"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("module_name", ["eval_offline", "eval_pivot", "eval_scratchpad"])
def test_pair_cli_full_path_and_resume(tmp_path, monkeypatch, module_name):
    module = importlib.import_module(module_name)
    pairs, output = tmp_path / "pairs.jsonl", tmp_path / "output.jsonl"
    write_records(pairs, [{"id": "0", "src": "Hallo", "ref": "bonjour"}])
    monkeypatch.setattr(sys, "argv", [module_name, "--pairs", str(pairs), "--output", str(output),
                                     "--src", "de", "--tgt", "fr", "--model", "test-model", "--run_id", "test"])
    monkeypatch.setattr(module, "check_server", lambda *args: None)
    prompts = []

    def complete(url, model, prompt, *args, **kwargs):
        prompts.append(prompt)
        if module_name == "eval_scratchpad":
            assert kwargs["stop"] == []
            return "hello\nFrench: bonjour"
        return "hello" if prompt.endswith("English:") else "bonjour"

    monkeypatch.setattr(module, "post_completion", complete)
    assert module.main() == 0
    records = list(evaluation.iter_records(output))
    assert records[0]["hyp"] == "bonjour"
    assert records[0]["src"] == "Hallo"
    assert len(prompts) == (2 if module_name == "eval_pivot" else 1)
    if module_name == "eval_pivot":
        assert "English: hello\nFrench:" in prompts[1]
        assert records[0]["hyp_en"] == "hello"
    header = evaluation.read_meta(output)
    assert header["pairs_sha256"]
    assert header["strategy"] == {"eval_offline": "direct", "eval_pivot": "pivot",
                                   "eval_scratchpad": "scratchpad"}[module_name]
    before = output.read_bytes()
    prompts.clear()
    assert module.main() == 0
    assert not prompts
    assert output.read_bytes() == before
    with pairs.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"id": "1", "src": "Neu", "ref": "nouveau"}) + "\n")
    with pytest.raises(RuntimeError, match="pairs_sha256"):
        module.main()
    assert output.read_bytes() == before


def test_server_failure_does_not_truncate_output(tmp_path, monkeypatch):
    import eval_offline
    pairs, output = tmp_path / "pairs.jsonl", tmp_path / "output.jsonl"
    write_records(pairs, [{"id": "0", "src": "Hallo", "ref": "bonjour"}])
    output.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["eval_offline", "--pairs", str(pairs), "--output", str(output),
                                     "--model", "test-model", "--run_id", "test", "--overwrite"])
    def fail(*args):
        raise RuntimeError("offline")
    monkeypatch.setattr(eval_offline, "check_server", fail)
    with pytest.raises(RuntimeError, match="offline"):
        eval_offline.main()
    assert output.read_text(encoding="utf-8") == "preserve me"


def test_streams_reject_different_lengths_and_ids(monkeypatch):
    rows = {"a": [{"id": "0", "text": "A"}, {"id": "1", "text": "B"}],
            "b": [{"id": "0", "text": "C"}]}
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda dataset, code, **kwargs: rows[code]))
    with pytest.raises(ValueError, match="shorter"):
        list(evaluation.stream_aligned_pairs("a", "b", "devtest"))
    rows["b"] = [{"id": "9", "text": "C"}]
    with pytest.raises(RuntimeError, match="diverged"):
        list(evaluation.stream_aligned_pairs("a", "b", "devtest"))


def test_streamed_cli_writes_and_resumes_without_new_inference(tmp_path, monkeypatch):
    output = tmp_path / "direct.jsonl"
    monkeypatch.setattr(sys, "argv", ["eval_llama", "--model", "test-model", "--run_id", "base-test",
                                     "--src", "de", "--tgt", "fr", "--output", str(output)])
    monkeypatch.setattr(evaluation, "check_server", lambda *args: None)
    monkeypatch.setattr(evaluation, "stream_aligned_pairs", lambda *args: iter([("0", "Hallo", "bonjour")]))
    calls = []
    def complete(*args):
        calls.append(args)
        return "bonjour\nignored"
    monkeypatch.setattr(evaluation, "post_completion", complete)
    assert evaluation.main() == 0
    assert list(evaluation.iter_records(output))[0]["hyp"] == "bonjour"
    assert len(calls) == 1
    assert evaluation.main() == 0
    assert len(calls) == 1


def test_shared_prompt_format_preserves_evaluation_and_probe_spacing():
    from prompts import PROBE_PROMPTS, translation_prompt
    import sampling_probe

    expected = "Translate the following Romanian text to English.\nRomanian: Bună\nEnglish:"
    assert evaluation.build_prompt("ro", "en", "Bună") == expected
    assert translation_prompt("Romanian", "English", "Bună", trailing_space=True) == expected + " "
    assert len(PROBE_PROMPTS) == 6
    assert sampling_probe.PROMPTS[0] == (
        "German", "English",
        "Wir haben jetzt 4 Monate alte Mäuse, die Diabetes hatten und jetzt keinen mehr haben, fügte er hinzu.")
    assert sampling_probe.has_loop("x " * 24)
    assert not sampling_probe.has_loop("x " * 23)


def test_repetition_definition_preserves_length_guard_and_case(tmp_path):
    assert has_loop("one two three four five six seven eight " * 3)
    assert not has_loop("one two three four five six seven eight " * 2)
    assert not has_loop("x " * 23)
    assert has_loop("x " * 24)
    path = tmp_path / "empty.jsonl"
    write_records(path, [], {"_meta": True})
    with pytest.raises(RuntimeError, match="No hypotheses"):
        summarize(path)
