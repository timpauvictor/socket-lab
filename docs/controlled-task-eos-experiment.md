# Controlled task-adapter EOS experiment

## Status: incomplete before training

Local preparation is implemented. The first approved Vast attempt on 2026-10-27
failed during input preparation, before any smoke or full optimizer step and
before any new evaluation. There are **no new experiment scores** to report.
Historical results and adapters remain unchanged.

The experiment compares:

| Arm | Starting weights | Additional training |
|---|---|---|
| A | Original task adapter over frozen language boxes | None |
| B | Independent load of the same original task adapter and boxes | 1,000 total steps, no EOS target |
| C | Independent load of the same original task adapter and boxes | 1,000 total steps, EOS target |
| D | Existing EOS-tuned monolith, without boxes | None; fresh reference evaluation |

B/C use ro-en, fr-en and de-en in order, with 333/333/334 steps and both directions
per pair. The selected examples, order, non-EOS tokens, truncation and settings
match. Full arms never initialize from smoke or from one another. Training and
A/B/C evaluation share the historical PEFT linear-factor box-composition owner.

The planned fresh evaluation is all four arms on en-ro, ro-en and de-en: 1,012
saved source/reference pairs per direction, 12 files and 12,144 attempts. One
BF16 Transformers/PEFT path uses greedy decoding until EOS or 128 new tokens,
without newline stopping. chrF scores first-line output; repetition uses the full
continuation. Token evidence, failures and completeness remain explicit.

## Preflight evidence

- The host had a 24 GB Blackwell GPU, CUDA/BF16 support and working PyTorch.
- The original task and EOS monolith config/weight hashes matched their inspected
  Hugging Face revisions. The EOS monolith matched
  `47a09863e61d03e0ef5bbae12cd36516ac770446`.
- Each language box matched its original completed rank-16 Modal run manifest.
  This was local historical-manifest inspection, not independent remote Modal
  reinspection. The freezer now validates this source type without inventing a
  Hub revision for artifacts that were not published there.
- Base/tokenizer revision resolved to
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; OPUS-100 resolved to
  `805090dc28bf78897da9641cdf08b61287580df9`.
- A separate new private artifact repository passed an actual upload, immutable
  revision download and byte comparison before execution.
- An independent deadline watchdog guarded setup and transferred responsibility
  to the experiment's independent watchdog after supervised launch.

## Observed failure and scoped repair

The image supplied Datasets 5.1.0 and Hugging Face Hub 2.2.0. Installing the
historical Transformers 4.57.6 stack required Hub below 1.0, and the resolver
selected Hub 0.36.2. The image's existing Datasets remained installed.

The first real Datasets import then failed:

```text
ImportError: cannot import name 'BucketNotFoundError' from 'huggingface_hub.errors'
```

Package-presence checks and successful imports of Transformers/PEFT did not
catch this cross-package incompatibility. Input preparation failed before
training. The runner entered its failure-finalization/stop path. A private
recovery bundle was retrieved at an immutable Hub revision; all 34 bundle files
passed SHA-256/size verification. It includes original input copies, preflight
records, stdout/stderr and an honest incomplete completion record. Private
credentials were not included.

SSH subsequently refused connections. The final instance stop acknowledgment
was not retrieved, so provider state and cessation of GPU billing still require
confirmation. The instance was not destroyed or automatically restarted.

The scoped repair adds `cloud/vast/task-eos-requirements.txt`, pinning Datasets
5.0.1 with the historical Transformers/PEFT stack and Hub 0.36.2. Installation
must be followed by `uv pip check` and actual imports of the whole training/eval
stack before launch. This repair has local checks; a renewed live dependency
check and all smoke/training/evaluation acceptance remain pending.

## Checks and next execution

The affected focused suite passed **100 tests** after the preparation and
provenance/dependency changes. Backend stand-ins test ownership, token matching,
resume/error handling, allocation and lifecycle failures; they do not prove real
GPU training. See [running instructions](running.md#controlled-task-adapter-eos-experiment-preparation-only)
and the [Vast procedure](../cloud/vast/README.md#controlled-combined-adapter-eos-experiment-separate-workflow).

A retry requires explicit authorization to restart the existing instance, current
SSH details, fresh run/output/artifact identities and a remaining-budget check.
The original approved deadline/spend ceiling must not reset automatically. No
training checkpoint exists to resume. Real smokes must verify finite gradients,
task updates, frozen weights and memory/time before full execution proceeds.

When results exist, compare C-B for EOS supervision, B-A for extra training and
C-A for the practical tune-up. D has different training history. Report every
direction, including negative results; one seed does not establish significance
or lack of forgetting, and French is not evaluated here.
