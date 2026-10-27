# Vast training operations

These commands operate on paid instances. They do not buy, start, or destroy
an instance. Uploads, provisioning, training, and instance stops occur only
when you run the corresponding command without `--dry-run`.

Use a compatible Linux/PyTorch host and read its `/etc/vast-agents-guide.md`
before acting. Confirm the Python environment, GPU, storage, and container
credentials on that specific host. The historical environment was
`/venv/main`; do not assume the image's versions or current rental prices.

## Local upload

From the repository root, preview the upload first:

```text
python cloud/vast/upload.py --host HOST --port PORT --dry-run
python cloud/vast/upload.py --host HOST --port PORT
```

`SSH_HOST` and `SSH_PORT` can supply the values instead. The uploader invokes
`ssh` and `scp` with argument lists. The sole remote setup command creates
fixed directories. It sends the Python operations and training scripts,
`translation_data.py`, the checkpoint at `vast/staging/ckpt-9000`, and the
four language adapters under `adapters/box-{ro,en,fr,de}`. Override the
checkpoint location with `--checkpoint`. All sources must exist before the
first connection. No tokens or `.env` files are included.

The remote layout stays `/workspace/bh`, with trainers under `scripts/` and
language adapters under `boxes/`. The uploader does not stage EOS tune-up
weights: prepare `/workspace/bh/tune-src` separately with the finished
monolith's configuration and weights, as for the historical run.

An upload failure returns a nonzero status; it does not stop a rented
instance from your local machine. Inspect the destination and billing yourself.

## Provision on the instance

Set `HF_TOKEN` privately in your SSH environment, then run:

```text
/venv/main/bin/python /workspace/bh/provision.py --dry-run
/venv/main/bin/python /workspace/bh/provision.py
```

Provisioning checks CUDA, installs the same non-Torch dependency specifications
as the old script into the interpreter you selected, caches the token with
owner-only permissions, checks base-model read access, and checks Triton,
`g++`, and GPU information. It does not reinstall Torch. The token must also
have upload access to the target private repositories; the model download
checks read access only. Provisioning failure does not automatically stop
the instance: inspect the error and stop it manually if you are not continuing.

## Smoke checks and full jobs

Run commands in the foreground so you can see failures. The runner writes
training output to `/workspace/bh/logs/`; it also writes a separate watchdog
log. If you use your own terminal-session manager, you still own monitoring
and billing. Run one job at a time. Do not overlap training and inference,
or reuse a job tag/log while its previous runner or watchdog is active.

```text
/venv/main/bin/python /workspace/bh/run.py compile-smoke
/venv/main/bin/python /workspace/bh/run.py task-smoke
/venv/main/bin/python /workspace/bh/run.py task
/venv/main/bin/python /workspace/bh/run.py monolith
/venv/main/bin/python /workspace/bh/run.py eos-tune
```

These are separate choices, not a batch to execute blindly:

- `compile-smoke`: plain then compiled, each 500 pairs / 20 steps / rank 8.
  Successful runs stay local and leave the instance running. Compare total
  time and loss; the recorded short trials did not demonstrate a speedup.
- `task-smoke`: the historical full-box, rank-8 task adapter smoke, capped at
  20 steps per pair. A successful run uploads/verifies its artifacts and
  leaves the instance running for the next check. Failures take the stop path.
- `task`: three sequential pair cycles over the language adapters, 150,000
  parallel sentence pairs per language pair, two epochs, task rank 32.
- `monolith`: resume from `/workspace/bh/ckpt-9000`, with the historical
  150,000-pair, two-epoch, rank-32 recipe.
- `eos-tune`: load `/workspace/bh/tune-src`, supervise EOS, and cap fresh
  optimizer/scheduler training at 1,000 steps with learning rate 5e-5.

Task, monolith, and task-smoke retain their historical `--compile` default;
EOS tune-up omits it. Use `--no-compile` to make an explicit override for a new
run. The runner exposes `--workspace`, `--python`, and `--dry-run`; its default
training interpreter is `/venv/main/bin/python`. Full jobs upload to the same
historical private Hugging Face repository IDs, then request an instance stop.

For recovery, pass `--resume-from PATH` to monolith, or both
`--resume-from PATH --resume-pair ro-en` (or the appropriate pair) to task.
Inspect checkpoint contents before selecting one. Monolith restores adapter
weights and step accounting, not optimizer, scheduler, or data-order state;
do not describe it as exact checkpoint recovery.

## Failure and shutdown behavior

Before training, the runner requires the instance-provided numeric
`CONTAINER_ID` and `CONTAINER_API_KEY`. It never falls back to an account key.
An independent watchdog remains alive throughout training and finalization:

- More than 60 minutes without training-log writes, or without any log,
  triggers a stop request.
- More than 15 minutes with an unfinished epilogue triggers a stop request.
- A completion message with no epilogue start after 15 minutes also triggers
  a stop request.
- Failed stop transports are retried at the five-minute check interval.

On success, finalization uploads adapter files, training logs, and run-info,
then verifies uploaded names and adapter weight size. On training failure it
uploads logs/run-info only. Missing artifacts or upload failures still attempt
to stop the instance and leave its disk intact. Failed uploads retain their
staging directories beside the log for recovery. The watchdog is not disabled
before uploads finish. Failed finalization leaves it available for recovery.

Ctrl+C or SIGTERM asks the runner to cancel its own training process group,
then finalize the failed run. A hard process/host failure may prevent that
immediate path; the independent watchdog is the fallback, not a guarantee
against infrastructure failure.

Stop uses the Vast CLI first, then the instance-scoped REST endpoint. No
command destroys an instance. A stop acknowledgment is a request, not proof
that billing ceased: check the dashboard. Stopped disks can still incur
storage charges. Destroying an instance remains a separate user action.

For manual finalization or diagnostics:

```text
/venv/main/bin/python /workspace/bh/lifecycle.py finish TAG OUTPUT_DIR OWNER/REPO EXIT_CODE LOG_FILE --dry-run
/venv/main/bin/python /workspace/bh/lifecycle.py watchdog LOG_FILE COMPLETION_TEXT TAG --dry-run
```

Dry runs perform no uploads, downloads, installs, marker/log writes, process
launches, or instance operations. `finish --keep-running` is reserved for
successful, verified smoke workflows; failure still requests a stop.

## Replaced entry points

| Previous script | Python replacement |
|---|---|
| `upload.sh` | `upload.py` |
| `provision.sh` | `provision.py` |
| `run-task.sh` | `run.py task` |
| `run-monolith.sh` | `run.py monolith` |
| `run-eos-tune.sh` | `run.py eos-tune` |
| `smoke-task.sh` | `run.py task-smoke` |
| `smoke-compile.sh` | `run.py compile-smoke` |
| `epilogue.sh` | `lifecycle.py finish` |
| `watchdog.sh` | `lifecycle.py watchdog` |

The [historical journal](../../docs/session-2026-10-20-vast.md) preserves its
original commands and measurements. Python replacements have local stub and
failure-path tests; live cloud acceptance still requires an explicitly
approved run. No refactor check constitutes a training result.

## Controlled combined-adapter EOS experiment (separate workflow)

Local preparation is not authority to use a paid host. Do not use `run.py eos-tune`,
`upload.py`, or the adapter-only finalizer for this experiment. Its package keeps
repository-relative paths, not the historical flat `scripts/` layout.

### Read-only local preview

From the repository root, with an existing Python environment (no installation):

```text
python -B cloud/vast/task_eos_experiment.py run --repo . --output _tools/eos-preview --run-id eos-preview --destination OWNER/eos-preview --full --dry-run
python -B cloud/vast/task_eos_experiment.py stage-list --repo .
```

`-B` also prevents Python import-cache writes. The first command only prints
argv: no output directories, processes, network, model/data downloads, packages,
training or instance operations. Revision placeholders are deliberate in preview;
live admission rejects them. The second prints an explicit staging allowlist.

The supervisor freezes input copies and selected OPUS rows once, smokes B/C at
3 steps per pair, and evaluates smoke A/B/C/D on two saved examples per direction.
With `--full`, it then starts B/C independently from the ORIGINAL task weights,
not smoke weights, at 333/333/334 steps. It evaluates all four arms afresh on all
1,012 saved pairs in each of en-ro, ro-en and de-en (12 files / 12,144 attempts).
A/B/C always use the same linear-factor box composition as training. D has no
boxes. Decoding is BF16 Transformers/PEFT, greedy, EOS-or-128-token-cap, with no
newline stop. Reports score first-line chrF but repetition uses full continuations.
Errors/missing rows make final scores/rates incomplete, not successful empty outputs.

### Required before any live operation

Obtain explicit upload/install/run/stop approval, SSH host/port/user and instance
ID, a NEW private Hugging Face `OWNER/RUN_ID` destination, and maximum elapsed seconds,
GPU price/hour and maximum spend in the same currency. Supply credentials through
private environment/tooling only. Confirm dataset/model access and private output
write permission. No credentials belong in notes, source, argv or archives.

Resolve ONE base/tokenizer commit and ONE OPUS-100 commit to 40-hex revisions.
Inspect all six adapter origins, including source revisions/run receipts. Compare
both config and weight SHA-256 values against those origins. The known monolith
revision is `47a09863e61d03e0ef5bbae12cd36516ac770446`; a different origin requires
an explicit approved alternate-source decision. Local presence and the recovered
monolith `run-info.json` do NOT establish source equality.

Prepare a private `provenance.json` object with keys `box-ro`, `box-en`, `box-fr`,
`box-de`, `task-full`, `eos-tune-001`. Each entry has this shape (replace placeholders):

```json
{"source":"OWNER/REPOSITORY or recorded immutable artifact source",
 "revision":"40-hex-source-commit",
 "evidence":"where/how these source bytes and run metadata were inspected",
 "sha256":{"adapter_config.json":"64-hex", "adapter_model.safetensors":"64-hex"}}
```

For the original Modal language boxes, use `source_type: "modal_manifest"`
instead of a fabricated Hub revision. Set `source` to
`modal-volume:VOLUME:VOLUME_PATH`, and include `run_id`, `manifest_sha256`,
`evidence` and the same two-file `sha256` object. The freezer validates the
original `_COMPLETE.json` bytes, completed rank-16 run/base/language/source identity,
and both recorded file hashes. State whether this is local historical receipt
inspection or independent remote verification; do not conflate them.

The file has no secrets. Hash/provenance disagreement aborts before training.
If an alternate monolith was approved, its entry also needs
`approved_alternate_source` with that decision. This is recorded evidence, not an
independent source audit. No unverifiable entry should be marked as inspected.

Read `/etc/vast-agents-guide.md` on the approved host. Check the selected Python,
CUDA/PyTorch/BF16, available RAM/disk, package versions, model/tokenizer access,
instance-only stop credentials, Vast stop transport and private artifact access.
For THIS experiment, after installation approval, use the scoped historical stack:

```text
uv pip install --python /venv/main/bin/python -r cloud/vast/task-eos-requirements.txt
uv pip check --python /venv/main/bin/python
/venv/main/bin/python -B -c "import torch, transformers, peft, datasets, accelerate, huggingface_hub, sacrebleu; print('Training and evaluation imports passed')"
```

Inspect the installation preview first; do not replace the working CUDA/Torch
stack. Datasets 5.1 requires a Hub API absent from Hub 0.36, while Transformers
4.57 requires Hub below 1.0. Therefore pin Datasets 5.0.1 with the historical
Transformers/PEFT stack rather than leaving the image's newer Datasets installed.
Validate ALL actual imports, not package-presence metadata alone, before launch.
The local preparation environment does not prove GPU/PEFT compatibility. Suggested starting class is 24 GB VRAM / 32 GB RAM / 80 GB free disk,
not a measured minimum. Stop-mechanism acceptance is an explicit user-owned live
check; do not test it by stopping an active run unexpectedly.

### Staging after approval only

Use the printed `stage-list` as an archive allowlist, keeping its relative paths.
For example, in a shell with `tar`, redirect that list to a private local file,
then `tar -cf PACKAGE.tar -T LIST.txt`. Add the inspected `provenance.json`
separately. Preview the list before doing this; it includes the six adapters,
three read-only pair logs, code and any existing adapter run receipts, NOT tokens,
`.env`, a model cache, unrelated results or this repository's Git data.
Upload only that package to an explicitly approved FRESH instance directory and
extract there (for example `/workspace/eos-code-RUN_ID`). Do not overlay an older
experiment. Compare staged config/weight hashes with the provenance file before
running. No staging/SSH command has been executed during local preparation.

### Instance command preview and approved execution

Replace the paths, revisions, repository owner, RUN_ID and budget placeholders.
First preview with the selected instance interpreter:

```text
/venv/main/bin/python -B /workspace/eos-code-RUN_ID/cloud/vast/task_eos_experiment.py run --repo /workspace/eos-code-RUN_ID --output /workspace/eos-RUN_ID --run-id RUN_ID --destination OWNER/RUN_ID --model-revision MODEL_COMMIT --dataset-revision OPUS_COMMIT --provenance /workspace/provenance.json --max-seconds SECONDS --max-cost COST --hourly-rate RATE --full --dry-run
```

Only after explicit execution approval: replace `--dry-run` with `--approved-live`.
`--full` authorizes the serial full stage after successful smoke. Without it,
only smoke runs, uploads and stops; it does NOT leave a billable host running.
A separately requested smoke-only session and any later full session need distinct
run IDs/destinations and an explicit budget decision (full repeats its own smoke).
If memory needs a different microbatch/accumulation, STOP and settle a matched
change before either full arm; the current runner deliberately fixes 8 x 4.
Measure smoke time/memory and review losses/update receipts before authorizing
full execution when those measurements are not already available.

The supervisor preserves every pair's adapter, checkpoints, logs, selected rows,
input copies, source hashes, runtime/package/hardware metadata, all evaluation
files, score JSON/report and completion/failure record under `OUTPUT/bundle/`.
It does not resume training automatically or publish banked scores. It rejects
existing outputs and existing destination repositories. Standalone evaluator
`--resume` is matching-only, validates source text/token evidence and keeps errors
incomplete; it is not an automatic supervisor recovery path.

An independent watchdog enforces the earlier of the elapsed-time ceiling and
`max_cost / hourly_rate`, through finalization. The supervisor reserves the final
60 seconds for failure finalization; this is not a guarantee that a large upload
will fit. It logs active stages every ten seconds; the guard also catches a dead
supervisor after one hour inactivity. Failed stages/cancellation/upload still
attempt instance-scoped stop. Failed stop leaves the guard retrying. No destroy
operation exists. The spend calculation is an estimated GPU-rate bound, not proof
of total provider/storage/network billing; approve those separately as needed.

### Bundle retrieval and remaining live checks

Finalization seals every nested bundle file with size/SHA-256, creates a NEW
private repository, uploads, then downloads ALL files at the returned immutable
commit and compares hashes (not just weight sizes). The local bundle is never
removed, including on upload failure. Frozen inputs have their own nested manifest.
After approved retrieval, run this read-only check with the staged code:

```text
python -B cloud/vast/task_eos_artifacts.py verify RETRIEVED_BUNDLE
```

Also retrieve `OUTPUT/finalize.stdout.log`, `finalize.stderr.log`, `closeout.json`
and the live `progress.log`/`watchdog.log` from disk. These necessarily postdate the
sealed bundle and are retained beside it, not falsely claimed inside its hash
manifest. `finalize.stdout.log` gives the verified remote commit and manifest hash;
`closeout.json` reports stop/upload acknowledgments separately from the bundle's
pre-finalization completion snapshot. An interrupted finalizer can leave the remote
upload present but unverified; inspect/retrieve rather than reusing the destination
or automatically retraining. A failed upload can have NO remote recovery copy.

Remaining live evidence: source provenance, resolved revisions/data selection;
real task-weight load, finite gradients/loss, task updates and unchanged frozen
weights in both three-pair smokes; memory/time; stop-mechanism acceptance; all twelve
full evaluations and their IDs; full bundle retrieval hashes; final stop receipt;
and the user's dashboard confirmation that GPU billing ceased. Stopped storage
may still bill. French is not evaluated, D has different training history, and
one seed does not establish significance or lack of forgetting. Report mixed or
negative directions honestly; local tests are not experiment results.
