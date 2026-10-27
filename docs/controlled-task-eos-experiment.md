# Controlled task-adapter EOS experiment

## Status: incomplete before training

Local preparation is implemented. Two approved Vast attempts on 2026-10-27
failed before any smoke or full optimizer step and before any new evaluation.
The first failed on dependency imports; the second exposed a token-boundary bug.
There are **no new experiment scores** to report.
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

SSH subsequently refused connections. After an explicitly approved user restart,
the first attempt's closeout receipt was recovered: upload and instance stop were
both acknowledged. Dashboard billing confirmation remains user-owned. The
instance was not destroyed or automatically restarted.

The scoped repair adds `cloud/vast/task-eos-requirements.txt`, pinning Datasets
5.0.1 with the historical Transformers/PEFT stack and Hub 0.36.2. Installation
must be followed by `uv pip check` and actual imports of the whole training/eval
stack before launch. On the approved retry, `uv pip check` passed for all 108 installed packages and
actual imports of the full training/evaluation stack passed. No Torch replacement
was needed. This resolved the dependency failure, but did not prove training.

## Second attempt: completion-boundary masking

The retry froze all three raw selections of 150,000 pairs, then failed while
tokenizing smoke B, before model loading or an optimizer step. Its complete
43-file recovery bundle was retrieved at an immutable revision and hash-verified.
The failure-stop path ran again; that retry's post-finalization stop receipt
remains pending retrieval.

The first offending row was short, not over budget. Its standalone prompt ended
with a space token; joint prompt/completion tokenization merged that space with
the first completion word. Counting standalone prompt tokens masked the whole
completion. Paired controls now use joint-token character offsets to find the
supervised boundary. Historical non-paired masking remains unchanged.

A local real-tokenizer scan inspected all 900,000 frozen translation directions.
After the boundary fix, eight entire pairs still genuinely could not provide a
completion token in one or both directions: six fr-en and two de-en. The user
approved excluding such an entire pair from BOTH controls during once-only frozen
preparation, without rewriting text or backfilling replacements. The raw selection,
excluded original indices/directions/reasons, usable order, counts and hashes are
preserved. Both translation directions remain for each retained pair.

The local scan used the exact tokenizer source revision but newer local library
versions; the pinned remote stack must verify its own counts. Local predictions
are 150,000 ro-en, 149,994 fr-en and 149,998 de-en usable pairs. No training used
these corrected inputs yet. The original budget/deadline remains in effect.

## Checks and next execution

The affected focused suite passed **102 tests** after the boundary and whole-pair
filtering fixes. The actual shared filtering owner also processed the complete
local frozen corpus and confirmed 150,000 / 149,994 / 149,998 usable pairs while
preserving the raw selections. Exact remote-library acceptance remains pending. Backend stand-ins test ownership, token matching,
resume/error handling, allocation and lifecycle failures; they do not prove real
GPU training. See [running instructions](running.md#controlled-task-adapter-eos-experiment-preparation-only)
and the [Vast procedure](../cloud/vast/README.md#controlled-combined-adapter-eos-experiment-separate-workflow).

The user approved the matched exclusion rule and another retry on the existing
instance. Before launch, confirm access, fresh run/output/artifact identities and
the remaining original budget/deadline.
The original approved deadline/spend ceiling must not reset automatically. No
training checkpoint exists to resume. Real smokes must verify finite gradients,
task updates, frozen weights and memory/time before full execution proceeds.

When results exist, compare C-B for EOS supervision, B-A for extra training and
C-A for the practical tune-up. D has different training history. Report every
direction, including negative results; one seed does not establish significance
or lack of forgetting, and French is not evaluated here.
