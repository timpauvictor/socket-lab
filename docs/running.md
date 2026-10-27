# Training and evaluation scripts

The reported results use the llama.cpp evaluation scripts, not the legacy direct-Transformers evaluator. The EOS tune-up is complete; the sequential task arm is retained as a historical experiment.

## Scripts

| Script | Purpose |
|---|---|
| `serve.py` | Start the local llama.cpp server in the foreground |
| `run_eval.py` | Launch evaluation through uv, including saved-record scoring |
| `run_train.py` | Launch a Modal job and tee its output to a local log |
| `training/train_lora.py` | Interleaved translation training; optional `--eos_stop` and capped `--tune_from` training |
| `training/train_task.py` | Sequential task training over merged language adapters |
| `training/train_box.py` | Monolingual Wikipedia training |
| `evaluation/eval_llama.py` | Stream FLORES+ pairs for direct inference, or rescore stored records |
| `evaluation/eval_offline.py` | Direct inference from an existing source/reference pair file |
| `evaluation/eval_pivot.py` | Two inference requests through an intermediate language |
| `evaluation/eval_scratchpad.py` | One generation with intermediate and final-language sections |
| `evaluation/loop_check.py` | Measure repeated eight-word windows in stored hypotheses |
| `evaluation/eval_translate.py` | Legacy Transformers path; different dataset/backend and PEFT factor merge |
| `vendor/llama_cpp/convert_lora_to_gguf.py` | Convert a PEFT adapter using the bundled b11146 GGUF code |

Run commands from the repository root. Dependency declarations are in [requirements.txt](../requirements.txt). Report generators live in [reporting/](../reporting/); outputs remain in [results/](../results/). Cloud jobs live in [cloud/modal/](../cloud/modal/) and [cloud/vast/](../cloud/vast/).

See the root [README](../README.md) for results and a quick saved-record scoring command. The [PyTorch guide](pytorch-guide.html) explains the training code with CPU examples. See [methods](methods.md) for the historical configurations and comparison limits.

Historical journals and review notes retain the paths used at the time: `initial/train_*.py` now lives in `training/`; evaluation and probe scripts are in `evaluation/`; the converter and its support code are in `vendor/llama_cpp/`. The old `modal/` and `vast/` code directories are now under `cloud/`. Local ignored checkpoints in `vast/staging/` are not moved.

## Shared code

- `training/translation_data.py` owns translation-pair filtering, training prompts, and completion-only tokenization. Language-box training keeps its separate full-block recipe.
- `evaluation/prompts.py` owns evaluation/probe prompt formatting and the six diagnostic inputs; `evaluation/repetition.py` owns their repeated-word check. Record validation and chrF scoring remain shared through `eval_llama.py`.
- `cloud/modal/artifacts.py` owns file hashes, manifest serialization, and verified retrieval. `cloud/modal/processes.py` owns environment capture and live subprocess logging; the original smoke job still uses buffered logging.

When copying a trainer manually, copy `translation_data.py` beside it. Task training also needs `task_model.py`. The termination probe needs `prompts.py` and `repetition.py` beside it. Modal images and the Vast training uploader include the dependencies they use; the training script names and remote training paths have not changed. Python launchers replace the old PowerShell/Bash entry points; historical journals retain their original commands.

## New inference runs

Start the server yourself in the foreground. Do not overwrite banked logs.

Start the server in one terminal, then run evaluation in another:

```text
python serve.py --server-exe llama-server --port 8083
python run_eval.py --src ro --tgt en --base-url http://localhost:8083 --run-id base-q8-b11146 --output logs/my-base-ro-en.jsonl
```

The launcher defaults to the local model file, alias `socket-lab`, and address `127.0.0.1`. Set `LLAMA_SERVER` or pass `--server-exe` for another executable location. Pass `--model` to select a different GGUF. The server launcher does not download models. The evaluation wrapper uses `uv` to supply sacreBLEU and, for inference only, Datasets.

For an adapter, restart the server with its converted file:

```text
python serve.py --server-exe llama-server --lora adapters/eos-tune-lora.gguf --port 8083
python run_eval.py --src en --tgt ro --base-url http://localhost:8083 --run-id eos-47a09863-q8-b11146 --output logs/my-tuned-en-ro.jsonl
```

Repeat `--lora` on `serve.py` for multiple adapters. The historical b11146 server receives one comma-joined `--lora` argument for those files. Do not assume the same command syntax for every llama.cpp version. A run ID is a declared identity, not proof of the server's loaded adapters; record exact model, adapter, and backend revisions yourself.

Rescore an existing file without a server or dataset package:

```bash
python -m pip install sacrebleu==2.6.0
python evaluation/eval_llama.py --score_only --output logs/eval-eostune-en-ro-devtest.jsonl
python evaluation/loop_check.py logs/eval-eostune-en-ro-devtest.jsonl
```

New inference requires `--run-id` on the wrapper, or `--run_id` on the underlying evaluation script. Resume checks strategy, pivot, decoding settings, and offline input hashes. Old headers lack these fields, so old files remain readable but cannot resume under the new guard. Choose a fresh output path. Corrupt or partial records require inspection; the scripts do not silently repair them.

## Sampling-probe outputs

The diagnostic probe requires a fresh output path and never overwrites an existing file:

```text
python evaluation/sampling_probe.py --base-url http://localhost:8083 --out results/my-new-probe.jsonl
```

Successful rows keep `hyp` and `looped`. Request or response-format failures instead contain an `error` object and no hypothesis or repetition result. The summary counts successful completions separately from failures; any failed request makes the command exit nonzero. Partial files remain available for inspection. Do not treat error rows as translations or silently include them in score denominators. Existing banked probe files are unchanged.

## Modal completion identity

Modal jobs now record a versioned request fingerprint before training and copy it into the completion manifest. It covers trainer arguments, trainer/helper and job-definition source bytes, and the input adapter configuration/weight bytes for task and EOS jobs. The job-definition snapshot also captures declared image dependencies and resource settings. EOS probe sources are included. Automatic checkpoint selection is recovery state, not a change to the original recipe.

A completed run is reused only if its run ID and request identity match. Missing legacy identities, changed parameters, changed code, or changed input adapters cause rejection before training or output writes. Monolith checkpoint recovery also requires a matching `_REQUEST.json`; other incomplete jobs still require inspection. Existing local completion receipts must match before retrieval, and matching receipts retain their original bytes.

Legacy artifacts remain readable; no manifests are backfilled or rewritten to claim a new identity. For a genuinely new run, select a new run ID and fresh local retrieval subdirectory in the job definition. Do not edit an old receipt to bypass the guard. Identity is not proof of reproducibility: unpinned remote model/dataset revisions and resolved dependency versions are not frozen by this check. Use one writer per run/output; this is not a distributed lock between concurrent Modal containers. Live provider behavior remains unverified by local stub tests.

## Adapter conversion and training

Conversion needs PyTorch, Transformers, and the converter's dependencies. It is separate from chart generation and score-only use. The adapter directory must contain its actual weights, not just the configuration committed here.

```bash
python vendor/llama_cpp/convert_lora_to_gguf.py --base-model-id Qwen/Qwen3-0.6B-Base --outtype f16 --outfile adapters/eos-tune-lora.gguf adapters/eos-tune-001
```

This can fetch base-model configuration from Hugging Face. To avoid that access, supply a prepared local configuration directory with `--base` instead. The converter uses `gguf-py-b11146`; no external `PYTHONPATH` override is needed unless you intentionally replace it.

Training requires separate model/data downloads and suitable hardware. Read `python training/train_lora.py --help` and the [Vast.ai runbooks](../cloud/vast/README.md) before starting a run. Modal jobs can incur charges; this documentation does not launch them. The Python training wrapper accepts `--modal-exe` and otherwise resolves `modal` on PATH.

```text
python run_train.py --job box --phase smoke --lang ro --dry-run
python run_train.py --job box --phase smoke --lang all
python run_train.py --job monolith --phase full
python run_train.py --job smoke --phase smoke
```

Only the first example is a preview; the others launch paid jobs. `--job smoke --phase smoke` replaces the standalone smoke launcher. Other job choices are `task` and `eos-tune`. Training logs remain under `logs/`, and child exit codes are preserved. All three root launchers support `--dry-run` without starting processes or creating logs. The server stays foreground-only; Ctrl+C cancels its owned child. No installed project CLI is required. Interrupting the training launcher cancels its local Modal client; confirm the remote job state in Modal before assuming billing has stopped.

The current scripts retain historical training defaults. `train_task.py` uses the original PEFT linear factor merge; its opt-in `--tune_from`, `--total_steps`, `--paired_control` and `--eos_stop` options support the separate controlled experiment described below. Manual monolithic resume restores adapter weights and step accounting, not optimizer state or data position. These paths must not be described as an exact continuation or a validated composition recipe.

## Controlled task-adapter EOS experiment: preparation only

The new path does not change historical recipes, banked logs or the evaluation
board. The [experiment report](controlled-task-eos-experiment.md) records the
first live preflight failure and the remaining checks; it contains no new scores. See [the complete Vast procedure](../cloud/vast/README.md#controlled-combined-adapter-eos-experiment-separate-workflow)
for staging, provenance format, approvals, budgets, execution and bundle retrieval.
The current packet authorizes local preparation ONLY, not uploads or instance use.

Read-only preview from the repository root (no dependency installation):

```text
python -B cloud/vast/task_eos_experiment.py run --repo . --output _tools/eos-preview --run-id eos-preview --destination OWNER/eos-preview --full --dry-run
python -B cloud/vast/task_eos_experiment.py stage-list --repo .
```

`training/task_model.py` is the shared train/eval box-composition owner.
`evaluation/eval_task_eos.py` uses the saved FLORES source/reference pairs only,
not their saved hypotheses; all four arms use one BF16 Transformers path.
`cloud/vast/task_eos_experiment.py` supervises freeze, disposable smoke, serial
B/C training and evaluation, complete-bundle finalization and stop. It never
uses smoke outputs to initialize full arms. Both full arms load the original
rank-32 task adapter, use 333/333/334 optimizer steps and identical 255-token
non-EOS content. Only C supervises the appended EOS. A remains untouched; D
loads the existing EOS-tuned monolith without boxes.

Focused local checks using the already available review environment:

```text
_tools/review-env/Scripts/python.exe -m pytest tests/test_training_tokens.py tests/test_task_eos.py tests/test_evaluation.py tests/test_vast.py tests/test_layout.py -q --basetemp=_tools/pytest-task-eos
```

Mocks verify local orchestration/contracts, not real GPU/PEFT correctness. Before
live execution, supply explicit authority, instance details, a new private artifact
destination, resolved model/data commits, inspected adapter provenance and a
cost/time ceiling. Real smoke updates/frozen weights, memory/time, provider stop,
all full comparisons and retrieved bundle hashes remain live checks.
