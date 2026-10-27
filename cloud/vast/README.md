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
