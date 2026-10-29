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

The setup command checks CUDA and installs the old script's non-Torch package
requirements into the Python environment you selected. It saves the token so
only its owner can read it, checks access to the base model, and checks Triton,
`g++` and the GPU. It does not reinstall Torch. The token must also
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
original commands and measurements. Python replacements have local tests with fake cloud calls,
including failure cases. A real cloud check still needs an explicitly approved
run. Passing code tests is not a training result.

## EOS-only task-adapter tune-up (separate workflow)

Preparing files locally does not give permission to use a paid host. Do not use `run.py eos-tune`,
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
checks before a real run reject them. The second prints an explicit staging allowlist.

This workflow trains ONE EOS-enabled copy of the finished task adapter.
See the [workflow summary](../../docs/controlled-task-eos-experiment.md). It does
not rebuild cartridges or train a no-EOS control. Original weights stay untouched.

The supervisor freezes inputs/selected OPUS rows, then runs a disposable EOS smoke
at 3 steps per pair. With `--full`, it starts again from ORIGINAL task weights,
not smoke weights, and performs 333/333/334 steps across ro-en/fr-en/de-en: 1,000
updates total, both directions, with only the task LoRA trainable. The existing
modular trainer restarts its optimizer/scheduler per pair; this is not the old
monolith's mixed-dataset schedule.

Each phase checks original versus EOS-tuned on the first TWO saved examples in
each of en-ro, ro-en and de-en: six inputs per adapter, twelve attempts per phase.
Internal evaluator IDs A/C mean original/EOS-tuned, not additional training arms.
Both use the same frozen linear-factor cartridge composition as training.
Decoding is BF16 Transformers/PEFT, greedy, EOS-or-128-token-cap, no newline stop.
Raw/token evidence, repetition and failures are preserved. Tiny diagnostic chrF
is not a general quality benchmark or a causal EOS claim. Missing/error rows make
the diagnostic incomplete. French is trained but not generation-checked.
The final adapter is `OUTPUT/bundle/full/task-eos/`, with product identity in
`request.json`. Smoke-only success is not a finished EOS adapter.

Opt-in training labels use the fast tokenizer's character offsets on the JOINT
prompt/completion text. A token containing the prompt's trailing space and the
first completion word is supervised, not mistaken for prompt-only content.
This retains completion content and does not rewrite source text. Historical non-paired masking is unchanged. Frozen preparation
preserves each raw selection as `PAIR.selected.json`, then excludes an ENTIRE
pair if either direction has no completion token within the
unchanged 255-token content budget. `PAIR.excluded.json` records original indices,
directions and reasons; `PAIR.json` keeps the remaining order without backfill.
Raw/usable/excluded counts and all file hashes remain in the bundle.

### Required before any live operation

Get explicit approval to upload, install, run and stop. Record the SSH host,
port, user and instance ID. Choose a NEW private Hugging Face `OWNER/RUN_ID`
destination. Set the time limit in seconds, GPU price per hour and maximum
spend, using the same currency for both prices. Supply credentials through
private environment/tooling only. Confirm dataset/model access and private output
write permission. No credentials belong in notes, source, argv or archives.

Resolve ONE base/tokenizer commit and ONE OPUS-100 commit to 40-hex revisions.
Inspect the five required adapter origins (four cartridges and task-full),
including source revisions/run receipts. Compare both config and weight SHA-256
values against those origins. The unrelated monolith EOS adapter is not required
or loaded. Local file presence alone does not prove source identity.

Prepare a private `provenance.json` object with keys `box-ro`, `box-en`, `box-fr`,
`box-de`, `task-full`. Each entry has this shape (replace placeholders):

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
and both recorded file hashes. Say whether you checked an old local receipt or independently checked the
remote files. Those are different checks.

Do not put secrets in this file. If hashes or source records disagree, stop
before training. Recording the checks does not independently prove the sources
are authentic. Do not mark an entry as checked if you could not verify it.

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
not a measured minimum. The user must approve a real check of the stop mechanism. Do not test it by
unexpectedly stopping an active run.

### Staging after approval only

Use the printed `stage-list` as an archive allowlist, keeping its relative paths.
For example, in a shell with `tar`, redirect that list to a private local file,
then `tar -cf PACKAGE.tar -T LIST.txt`. Add the inspected `provenance.json`
separately. Preview the list before doing this; it includes the five adapters,
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
change before full training; the current runner deliberately fixes 8 x 4.
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
weights in the three-pair EOS smoke; memory/time; stop-mechanism acceptance;
exactly 1,000 full optimizer steps; the small original/EOS diagnostic and matching
IDs; bundle and final-adapter retrieval hashes; final stop receipt; and the user's
dashboard confirmation that GPU billing ceased. Stopped storage may still bill.
Do not claim universal EOS termination or improved translation quality from six
inputs. Report regressions honestly; local tests are not live training results.
