# Training and evaluation scripts

The reported results use the llama.cpp evaluation scripts, not the legacy direct-Transformers evaluator. The historical monolith EOS tune-up is complete; the EOS-only task-adapter tune-up is a separate workflow below.

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

A useful reading order is `run_train.py` -> the selected cloud job -> `training/train_task.py`. The trainer parses inputs, chooses each pair's source and step budget, then calls `train_pair` to train/save it. `translation_data.py` explains the two token-masking modes. For evaluation, start with `run_eval.py`, then follow the ordinary request/record helpers in `eval_llama.py`. Vast recipe settings are named fields in `cloud/vast/recipes.py`, not positional tuples.

The model work uses [Hugging Face Trainer](https://github.com/huggingface/transformers/blob/v4.57.1/examples/pytorch/language-modeling/run_clm.py), [TrainerCallback](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/main_classes/callback.md) and [PEFT](https://huggingface.co/docs/peft/en/quicktour). HTTP calls follow [Requests' ordinary API](https://requests.readthedocs.io/en/latest/user/quickstart/), with explicit timeouts and no automatic retries. The reference examples explain the APIs; they do not replace this experiment's settings or masking/composition rules.

- `training/translation_data.py` handles pair filtering, training prompts and completion-only tokenization. Language-box training keeps its separate full-block recipe.
- `evaluation/prompts.py` formats the evaluation/probe prompts and holds the six diagnostic inputs. `evaluation/repetition.py` checks for repeated words. Record validation and chrF scoring remain shared through `eval_llama.py`.
- `cloud/modal/artifacts.py` hashes files, writes manifests and checks downloads. `cloud/modal/processes.py` records the environment and prints child-process output as it arrives. The original smoke job still buffers its output.

When copying a trainer manually, copy `translation_data.py` beside it. Task training also needs `task_model.py`. The termination probe needs `prompts.py` and `repetition.py` beside it. Modal images and the Vast training uploader include the dependencies they use; the training script names and remote training paths have not changed. Python launchers replace the old PowerShell/Bash entry points; historical journals retain their original commands.

## New inference runs

Start the server yourself in the foreground. Do not overwrite saved logs.

Start the server in one terminal, then run evaluation in another:

```text
python serve.py --server-exe llama-server --port 8083
python run_eval.py --src ro --tgt en --base-url http://localhost:8083 --run-id base-q8-b11146 --output logs/my-base-ro-en.jsonl
```

The launcher defaults to the local model file, alias `socket-lab`, and address `127.0.0.1`. Set `LLAMA_SERVER` or pass `--server-exe` for another executable location. Pass `--model` to select a different GGUF. The server launcher does not download models. The evaluation wrapper uses `uv` to supply sacreBLEU and, for inference only, Datasets and `requests==2.34.2`. Directly invoking a server-based evaluation script requires Requests in that environment too. Saved-record scoring does not import Requests or contact a server. HTTP timeouts apply to socket operations, not a whole experiment deadline; redirects retain the previous GET/POST rules and must not replay completion POSTs through 307/308.

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

New inference requires `--run-id` on the wrapper, or `--run_id` on the underlying evaluation script. Resume checks strategy, pivot, decoding settings, and offline input hashes. The three offline evaluators also match every saved ID/source/reference tuple against the loaded pairs before appending or requesting completions. Pivot resume requires its stored intermediate (`hyp_en` for English, otherwise `hyp_pivot`). Old headers lack these fields, so old files remain readable but cannot resume under the new guard. Choose a fresh output path. Corrupt or partial records require inspection; the scripts do not silently repair them.

## EOS task-adapter benchmark on AMD (llama.cpp/Vulkan)

Use the Vulkan build of `llama-server`, not the CUDA-only cloud diagnostic
`eval_task_eos.py`. No CUDA Python environment is needed for local inference.

The finished task weights are at `adapters/task-eos/`. The derived combined
adapters are `adapters/task-eos-ro-en.gguf` and `adapters/task-eos-de-en.gguf`.
Each includes the pair's PEFT linear language-factor composition AND the EOS task
delta. Load ONE combined adapter, not the separate boxes or task again. Both
Romanian directions use the ro-en adapter; de-en uses the de-en adapter.

From the repository root, start the server in one terminal:

```text
python serve.py --server-exe llama-server --lora adapters/task-eos-ro-en.gguf --port 8083
```

If the executable is not on PATH, replace `llama-server` with the full path to
your Vulkan `llama-server.exe`. The launcher uses the existing
`models/qwen3-0.6b-base-q8_0.gguf` and requests GPU offload via `-ngl 99`.
Verify the server's startup log identifies Vulkan/your AMD GPU; a different
llama.cpp build can fall back to CPU. No server is started automatically.

In a second terminal, run these PowerShell commands. They use all 1,012 saved
source/reference pairs per direction, without another dataset download:

```powershell
$frozen = "_tools/task-eos-live/eos-only-57998302/bundle/inputs/logs"
uv run --no-project --with sacrebleu==2.6.0 --with requests==2.34.2 -- python evaluation/eval_offline.py --pairs "$frozen/eval-en-ro-devtest.jsonl" --model socket-lab --run_id task-eos-linear-ro-en-q8-vulkan --src en --tgt ro --base_url http://localhost:8083 --output logs/eval-task-eos-vulkan-en-ro-devtest.jsonl
uv run --no-project --with sacrebleu==2.6.0 --with requests==2.34.2 -- python evaluation/eval_offline.py --pairs "$frozen/eval-ro-en-devtest.jsonl" --model socket-lab --run_id task-eos-linear-ro-en-q8-vulkan --src ro --tgt en --base_url http://localhost:8083 --output logs/eval-task-eos-vulkan-ro-en-devtest.jsonl
```

Stop that server with Ctrl+C, then restart it with the German/English adapter:

```text
python serve.py --server-exe llama-server --lora adapters/task-eos-de-en.gguf --port 8083
```

Run the remaining direction in the second terminal:

```powershell
uv run --no-project --with sacrebleu==2.6.0 --with requests==2.34.2 -- python evaluation/eval_offline.py --pairs "$frozen/eval-de-en-devtest.jsonl" --model socket-lab --run_id task-eos-linear-de-en-q8-vulkan --src de --tgt en --base_url http://localhost:8083 --output logs/eval-task-eos-vulkan-de-en-devtest.jsonl
```

Each command prints chrF and saves its records. Do not overwrite saved logs.
These are the existing llama.cpp chrF settings, including newline stopping;
they do NOT measure EOS stop rates or reproduce the cloud EOS-or-cap diagnostic.
Quantized inference also differs from the BF16 cloud backend. Record the actual
llama.cpp build/GPU used rather than treating the declared run ID as proof.

### Recreate the combined exports (CPU only)

The files above are already prepared locally. For a fresh export location,
use the CPU exporter, then the existing bundled converter. This requires CPU
PyTorch, Safetensors and the converter's dependencies, not CUDA or PEFT:

```text
python training/export_task_adapter.py --task adapters/task-eos --box adapters/box-ro --box adapters/box-en --output _tools/my-task-eos-ro-en
python vendor/llama_cpp/convert_lora_to_gguf.py --base _tools/amd-task-eos/base-config --outtype f16 --outfile adapters/my-task-eos-ro-en.gguf _tools/my-task-eos-ro-en
```

Repeat with `box-de` and a fresh `task-eos-de-en` output for German/English.
The local base configuration was fetched from model revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; no base weights are needed for export.
The exporter rejects existing output directories. The vendor converter can
overwrite GGUF files, so choose a fresh filename when recreating an export.
Combined exports use rank 48 with alpha 48 (unit scaling), retain source hashes
in `export-info.json`, and are inference artifacts, not task tune-from inputs.
They preserve float32 factor algebra, not BF16 base-merge rounding or GGUF
quantization rounding. Training inputs and original task weights stay unchanged.

## Sampling-probe outputs

The diagnostic probe requires a fresh output path and never overwrites an existing file:

```text
python evaluation/sampling_probe.py --base-url http://localhost:8083 --out results/my-new-probe.jsonl
```

Successful rows keep `hyp` and `looped`. Request or response-format failures instead contain an `error` object and no hypothesis or repetition result. The summary counts successful completions separately from failures; any failed request makes the command exit nonzero. Partial files remain available for inspection. Do not treat error rows as translations or silently include them in score denominators. Existing banked probe files are unchanged.

## Modal completion identity

Modal jobs now record a versioned request fingerprint before training and copy it into the completion manifest. It covers trainer arguments, trainer/helper and job-definition source bytes, and the input adapter configuration/weight bytes for task and EOS jobs. The job-definition snapshot also captures declared image dependencies and resource settings. EOS probe sources are included. Automatic checkpoint selection is recovery state, not a change to the original recipe.

A completed run is reused only if its run ID and request identity match. Missing legacy identities, changed parameters, changed code, or changed input adapters cause rejection before training or output writes. Monolith checkpoint recovery also requires a matching `_REQUEST.json`; other incomplete jobs still require inspection. Existing local completion receipts must match before retrieval, and matching receipts retain their original bytes.

Retrieval accepts only portable flat artifact filenames and checks all selected destinations before downloading. Existing matching regular files are reused; conflicting files and destination links are rejected. Each transfer uses a fresh, exclusively owned sibling `.part` file. A failed transfer prints its retained partial path for inspection; a retry creates a different partial and preserves earlier evidence, including pre-existing `<name>.part` files. Only verified bytes reach the final filename; completion is published last. Inspect retained partials before any manual cleanup. Use one writer per retrieval directory; these checks are not a sandbox against concurrent directory replacement.

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

Only the first example is a preview; the others launch paid jobs. `--job smoke --phase smoke` replaces the standalone smoke launcher. Other job choices are `task` and `eos-tune`. Training logs default to `logs/`, and child exit codes are preserved. The launcher exclusively creates its console log before starting the Modal client: an existing path, including a link, is rejected without truncation or launch. For another invocation, select a fresh path with `--log-file logs/my-task-smoke-console.log`. Relative log paths use `--repo-root`; absolute paths are used as supplied. `--dry-run` previews the selected path without creating it. All three root launchers support `--dry-run` without starting processes or creating logs. The server stays foreground-only; Ctrl+C cancels its owned child. No installed project CLI is required. Interrupting the training launcher cancels its local Modal client; confirm the remote job state in Modal before assuming billing has stopped.

Task training now accepts repeated language bindings, for example `--box ro=./box-ro --box en=./box-en` for `--pairs ro-en`. Supply a binding for every language in the selected pairs; duplicates, missing directories and malformed bindings fail before loading models/data. Quote each `LANG=PATH` argument if its path has spaces. The former `--box_ro`, `--box_en`, `--box_fr`, `--box_de` flags are retired; current Modal/Vast callers use the new spelling. Historical command logs retain their original arguments and are not edited or automatically replayed. A changed command/source identity requires a fresh approved run, not editing an old request receipt. The CPU exporter's separate `--box PATH` option is unchanged: pass exactly two paths in training-pair order, without language labels.

The current scripts retain historical training defaults. `train_task.py` uses the original PEFT linear factor merge; its opt-in `--tune_from`, `--total_steps`, `--paired_control` and `--eos_stop` options support the EOS-only tune-up described below. Manual monolithic resume restores adapter weights and step accounting, not optimizer state or data position. These paths must not be described as an exact continuation or a validated composition recipe.

## EOS-only task-adapter tune-up: local implementation

The new path does not change historical recipes, banked logs or the evaluation
board. See the [workflow summary](controlled-task-eos-experiment.md) and
[the complete Vast procedure](../cloud/vast/README.md#eos-only-task-adapter-tune-up-separate-workflow)
for staging, provenance format, approvals, budgets, execution and bundle retrieval.
Start with the local preview. Get explicit approval before uploading files or using a paid instance.

Read-only preview from the repository root (no dependency installation):

```text
python -B cloud/vast/task_eos_experiment.py run --repo . --output _tools/eos-preview --run-id eos-preview --destination OWNER/eos-preview --full --dry-run
python -B cloud/vast/task_eos_experiment.py stage-list --repo .
```

Training and evaluation both use `training/task_model.py` to combine the language adapters.
`evaluation/eval_task_eos.py` uses the saved FLORES source/reference pairs only,
not their saved hypotheses; original and EOS-tuned use the same BF16 Transformers
path and frozen cartridge composition. `cloud/vast/task_eos_experiment.py` now
supervises ONE EOS-only copy: freeze, disposable 3/3/3-step smoke, optional full
333/333/334-step tune-up, basic diagnostic, complete-bundle finalization and stop.
Full training restarts from the original rank-32 task adapter, never smoke weights.
Only the task LoRA changes; all cartridges/base weights remain frozen. The internal
`--paired_control` flag selects safe 255-token content/joint-offset masking; it
does not train another control. EOS is supervised at the end of every target.

The diagnostic uses only two saved examples per direction (en-ro, ro-en, de-en),
six inputs per adapter. It records EOS/cap/repetition/errors, not a full benchmark
or causal comparison. Explicit matching `eval_task_eos.py --resume` retains failed IDs as attempted and continues only unattempted pairs; it does not retry failed rows. Any retained failure keeps the diagnostic incomplete, its final benchmark metrics absent, and its exit status nonzero even when every remaining pair has been attempted. Malformed success/error evidence is rejected without modifying the file. No monolith EOS adapter is required. The final product is
`OUTPUT/bundle/full/task-eos/`; the unchanged source remains the baseline.

Focused local checks using the already available review environment:

```text
_tools/review-env/Scripts/python.exe -m pytest tests/test_training_tokens.py tests/test_task_eos.py tests/test_evaluation.py tests/test_vast.py tests/test_layout.py -q --basetemp=_tools/pytest-task-eos
```

These local tests use fake cloud and model calls. They don't show that real
GPU/PEFT training works. Before a paid run, get approval, check the instance and
adapter sources, choose a new private output destination, pin the model/data
commits, and set time and spending limits. On the real host, check the smoke
weight changes, frozen weights, memory/time and stop mechanism. Also verify the
1,000-step run, small diagnostic and downloaded file hashes.
