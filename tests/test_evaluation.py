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
                  strategy="direct", pivot=None, pairs=None)
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
    def request(url, body, timeout, **kwargs):
        assert url == "http://unused/completion"
        assert timeout == 300 and kwargs == {"decode_errors": "strict"}
        return 200, json.dumps(payload)
    monkeypatch.setattr(probe, "request_text", request)
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
    for package in ("requests", "datasets", "transformers"):
        monkeypatch.setitem(sys.modules, package, None)
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


@pytest.mark.parametrize("module_name,pivot", [("eval_offline", None), ("eval_pivot", "en"),
                                               ("eval_pivot", "ro"), ("eval_scratchpad", None)])
def test_pair_cli_full_path_and_resume(tmp_path, monkeypatch, module_name, pivot):
    module = importlib.import_module(module_name)
    pairs, output = tmp_path / "pairs.jsonl", tmp_path / "output.jsonl"
    write_records(pairs, [{"id": "0", "src": "Hallo", "ref": "bonjour"}])
    argv = [module_name, "--pairs", str(pairs), "--output", str(output),
            "--src", "de", "--tgt", "fr", "--model", "test-model", "--run_id", "test"]
    monkeypatch.setattr(sys, "argv", argv + (["--pivot", pivot] if pivot else []))
    monkeypatch.setattr(module, "check_server", lambda *args: None)
    prompts = []

    def complete(url, model, prompt, *args, **kwargs):
        prompts.append(prompt)
        if module_name == "eval_scratchpad":
            assert kwargs["stop"] == []
            return "hello\nFrench: bonjour"
        return "hello" if prompt.endswith(("English:", "Romanian:")) else "bonjour"

    monkeypatch.setattr(module, "post_completion", complete)
    assert module.main() == 0
    records = list(evaluation.iter_records(output))
    assert records[0]["hyp"] == "bonjour"
    assert records[0]["src"] == "Hallo"
    assert len(prompts) == (2 if module_name == "eval_pivot" else 1)
    if module_name == "eval_pivot":
        assert f"{evaluation.LANG_NAMES[pivot]}: hello\nFrench:" in prompts[1]
        intermediate = "hyp_en" if pivot == "en" else "hyp_pivot"
        assert records[0][intermediate] == "hello"
    header = evaluation.read_meta(output)
    assert header["pairs_sha256"]
    assert header["strategy"] == {"eval_offline": "direct", "eval_pivot": "pivot",
                                   "eval_scratchpad": "scratchpad"}[module_name]
    if module_name == "eval_pivot":
        assert "second_base_url" not in header
    before = output.read_bytes()
    prompts.clear()
    assert module.main() == 0
    assert not prompts
    assert output.read_bytes() == before
    corruptions = [dict(records[0], id="unknown"), dict(records[0], src="wrong source"),
                   dict(records[0], ref="wrong reference")]
    if module_name == "eval_pivot":
        missing = dict(records[0])
        del missing[intermediate]
        corruptions.extend([missing, dict(records[0], **{intermediate: None})])
    for corrupt in corruptions:
        write_records(output, [corrupt], header)
        original = output.read_bytes()
        with pytest.raises(RuntimeError, match="input identity mismatch|Invalid record fields"):
            module.main()
        assert output.read_bytes() == original and not prompts
    output.write_bytes(before)
    with pairs.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"id": "1", "src": "Neu", "ref": "nouveau"}) + "\n")
    with pytest.raises(RuntimeError, match="pairs_sha256"):
        module.main()
    assert output.read_bytes() == before


def test_pivot_two_servers_route_both_legs_and_protect_resume(tmp_path, monkeypatch):
    import eval_pivot
    pairs, output = tmp_path / "pairs.jsonl", tmp_path / "pivot.jsonl"
    write_records(pairs, [{"id": "0", "src": "Hallo", "ref": "bonjour"}])
    argv = ["eval_pivot", "--pairs", str(pairs), "--output", str(output),
            "--model", "test-model", "--run_id", "split-test", "--base_url", "http://first/"]
    monkeypatch.setattr(sys, "argv", argv + ["--second_base_url", "http://second/"])
    checks, requests = [], []
    monkeypatch.setattr(eval_pivot, "check_server", lambda *args: checks.append(args))

    def complete(url, model, prompt, tokens, timeout):
        requests.append((url, model, prompt, tokens, timeout))
        return "hello\nignored" if url == "http://first" else "bonjour\nignored"

    monkeypatch.setattr(eval_pivot, "post_completion", complete)
    assert eval_pivot.main() == 0
    assert [call[0] for call in checks] == ["http://first", "http://second"]
    assert [call[0] for call in requests] == ["http://first", "http://second"]
    assert all(call[1] == "test-model" and call[3:] == (128, 300.0) for call in requests)
    assert "German: Hallo\nEnglish:" in requests[0][2]
    assert "English: hello\nFrench:" in requests[1][2]
    assert list(evaluation.iter_records(output)) == [
        {"id": "0", "src": "Hallo", "hyp_en": "hello", "hyp": "bonjour", "ref": "bonjour"}]
    assert evaluation.read_meta(output)["second_base_url"] == "http://second"
    before = output.read_bytes()
    requests.clear()
    assert eval_pivot.main() == 0
    assert not requests and output.read_bytes() == before
    monkeypatch.setattr(sys, "argv", argv + ["--second_base_url", "http://different"])
    with pytest.raises(RuntimeError, match="second_base_url"):
        eval_pivot.main()
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="second_base_url"):
        eval_pivot.main()
    monkeypatch.setattr(sys, "argv", argv + ["--second_base_url", " "])
    with pytest.raises(SystemExit) as error:
        eval_pivot.main()
    assert error.value.code == 2
    assert not requests and output.read_bytes() == before


@pytest.mark.parametrize("existing", [False, True])
def test_pivot_second_server_failure_preserves_output(tmp_path, monkeypatch, existing):
    import eval_pivot
    pairs, output = tmp_path / "pairs.jsonl", tmp_path / "pivot.jsonl"
    write_records(pairs, [{"id": "0", "src": "Hallo", "ref": "bonjour"}])
    if existing:
        output.write_bytes(b"banked bytes\n")
    monkeypatch.setattr(sys, "argv", ["eval_pivot", "--pairs", str(pairs), "--output", str(output),
        "--model", "test-model", "--run_id", "split-test", "--base_url", "http://first",
        "--second_base_url", "http://second", "--overwrite"])

    def check(url, *args):
        if url == "http://second":
            raise RuntimeError("second server offline")

    monkeypatch.setattr(eval_pivot, "check_server", check)
    monkeypatch.setattr(eval_pivot, "post_completion", lambda *args: pytest.fail("request before health checks"))
    with pytest.raises(RuntimeError, match="second server offline"):
        eval_pivot.main()
    if existing:
        assert output.read_bytes() == b"banked bytes\n"
    else:
        assert not output.exists()


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
        "Die blaue Tasse steht neben dem kleinen Radio.")
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


@pytest.fixture
def http_server():
    """A real local transport with handwritten responses, never model inference."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    state = {"calls": [], "responses": {
        "/health": {"body": {"status": "ok"}},
        "/v1/models": {"body": {"data": [{"id": "test-model"}]}},
        "/v1/completions": {"body": {"choices": [{"text": "bonjour\nignored"}]}},
        "/completion": {"body": {"content": "Salut, țară!\nextra"}},
    }}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self):
            payload = None
            if self.command == "POST":
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append((self.command, self.path, payload))
            response = state["responses"].get(self.path, {"status": 404, "body": {}})
            body = response["body"]
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(response.get("status", 200))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in response.get("headers", {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        do_GET = respond
        do_POST = respond

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


def test_requests_transport_cli_failures_redirects_and_timeout(http_server, tmp_path, monkeypatch):
    import requests
    import eval_offline

    url, state = http_server
    pairs, output = tmp_path / "pairs.jsonl", tmp_path / "out.jsonl"
    write_records(pairs, [{"id": "0", "src": "Grüße", "ref": "bonjour"}])
    monkeypatch.setattr(sys, "argv", ["eval_offline", "--pairs", str(pairs), "--output", str(output),
        "--model", "test-model", "--run_id", "http-fixture", "--base_url", url,
        "--src", "de", "--tgt", "fr"])
    assert eval_offline.main() == 0
    assert list(evaluation.iter_records(output)) == [
        {"id": "0", "src": "Grüße", "ref": "bonjour", "hyp": "bonjour"}]
    method, path, payload = state["calls"][-1]
    assert (method, path) == ("POST", "/v1/completions")
    assert payload == {"model": "test-model", "prompt": "Translate the following German text to French.\nGerman: Grüße\nFrench:",
                       "max_tokens": 128, "temperature": 0, "stop": ["\n"], "stream": False}
    before = output.read_bytes()
    state["calls"].clear()
    assert eval_offline.main() == 0
    assert output.read_bytes() == before
    assert [call[0] for call in state["calls"]] == ["GET", "GET"]

    responses = state["responses"]
    for response, error in [({"status": 503, "body": b"unavailable"}, "status 503"),
                            ({"body": b"not-json"}, "not JSON"),
                            ({"body": {}}, "no usable choices"),
                            ({"body": {"choices": [{"text": None}]}}, "no text")]:
        responses["/v1/completions"] = response
        state["calls"].clear()
        with pytest.raises(RuntimeError, match=error):
            evaluation.post_completion(url, "test-model", "synthetic", 128, 5)
        assert len(state["calls"]) == 1
    responses["/health"] = {"body": {"status": "loading"}}
    with pytest.raises(RuntimeError, match="not ready"):
        evaluation.check_server(url, "test-model", 5)
    responses["/health"] = {"body": {"status": "ok"}}
    with pytest.raises(RuntimeError, match="not served"):
        evaluation.check_server(url, "missing-model", 5)

    responses["/redirect"] = {"status": 303, "body": {}, "headers": {"Location": "/health"}}
    state["calls"].clear()
    assert evaluation.request_text(url + "/redirect", {}, 5)[0] == 200
    assert [(method, path) for method, path, _ in state["calls"]] == [("POST", "/redirect"), ("GET", "/health")]
    for status in (307, 308):
        responses["/redirect"]["status"] = status
        state["calls"].clear()
        with pytest.raises(RuntimeError, match=f"status {status}"):
            evaluation.request_text(url + "/redirect", {}, 5)
        assert len(state["calls"]) == 1
    responses["/loop"] = {"status": 301, "body": {}, "headers": {"Location": "/loop"}}
    state["calls"].clear()
    with pytest.raises(RuntimeError, match="redirect limit"):
        evaluation.request_text(url + "/loop", None, 5)
    assert len(state["calls"]) == 5

    attempts = []
    def timeout(*args, **kwargs):
        attempts.append(kwargs)
        raise requests.Timeout("synthetic timeout")
    monkeypatch.setattr(requests, "request", timeout)
    with pytest.raises(RuntimeError, match="synthetic timeout"):
        evaluation.request_text(url + "/health", None, 0.2)
    assert len(attempts) == 1 and attempts[0]["timeout"] == 0.2


def test_sampling_probe_uses_shared_http_without_changing_native_requests(http_server, tmp_path, monkeypatch):
    import requests
    import sampling_probe as probe

    url, state = http_server
    assert probe.complete(url + "/", "probe-țară", 0.7, 1.1) == "Salut, țară!\nextra"
    assert state["calls"] == [("POST", "/completion", {
        "prompt": "probe-țară", "temperature": 0.7, "repeat_penalty": 1.1,
        "n_predict": 128, "stop": ["\n"], "stream": False})]

    monkeypatch.setattr(probe, "PROMPTS", [("Romanian", "English", "Test țară.")])
    output = tmp_path / "success.jsonl"
    state["calls"].clear()
    assert probe.main(["--base-url", url, "--out", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["hyp"] for row in rows] == ["Salut, țară!\nextra"] * 2
    assert [call[2]["temperature"] for call in state["calls"]] == [0.0, 0.7]
    assert all(call[2]["prompt"] == "Translate the following Romanian text to English.\nRomanian: Test țară.\nEnglish: "
               for call in state["calls"])

    response = state["responses"]["/completion"]
    response["body"] = b'{"content":"\xff"}'
    with pytest.raises(UnicodeDecodeError):
        probe.complete(url, "synthetic", 0, 1.1)
    assert evaluation.request_text(url + "/completion", {}, 5) == (200, '{"content":"\ufffd"}')
    response["body"] = b"not-json"
    with pytest.raises(json.JSONDecodeError):
        probe.complete(url, "synthetic", 0, 1.1)

    response.update(status=503, body=b"unavailable")
    state["calls"].clear()
    failed = tmp_path / "failed.jsonl"
    assert probe.main(["--base-url", url, "--out", str(failed)]) == 1
    rows = [json.loads(line) for line in failed.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(state["calls"]) == 2
    assert all("status 503" in row["error"]["message"] and "hyp" not in row and "looped" not in row for row in rows)

    attempts = []
    def timeout(method, endpoint, **kwargs):
        attempts.append((method, endpoint, kwargs))
        raise requests.Timeout("synthetic timeout")
    monkeypatch.setattr(requests, "request", timeout)
    timed_out = tmp_path / "timeout.jsonl"
    assert probe.main(["--base-url", url, "--out", str(timed_out)]) == 1
    assert len(attempts) == 2  # One attempt per condition, no retries.
    assert all(method == "POST" and endpoint == url + "/completion" and kw["timeout"] == 300
               and kw["allow_redirects"] is False for method, endpoint, kw in attempts)
    assert all("synthetic timeout" in json.loads(line)["error"]["message"]
               for line in timed_out.read_text(encoding="utf-8").splitlines())
