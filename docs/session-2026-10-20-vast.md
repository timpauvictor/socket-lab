# Vast launch session — 2026-10-20

Operational journal for the evening the program moved from Modal to Vast.ai:
buying hosts, proving the self-closing loop, launching both full runs.
Live numbers in §§1–12 are "as of last check" while runs were still in
flight. §13 records the local-eval era; §14 corrects §13 where later
evidence overturned it and states the current plan.

## 1. Starting state

- Qwen3-0.6B-Base floor (local llama.cpp, `/v1/completions`, greedy temp 0):
  ro->en **57.16**, en->ro **38.98**, de->en **60.43**. fr floors still
  blocked on the local server being started.
- Four full box adapters local and verified (Modal fulls): ro 2.395,
  fr 2.455, de 2.515, en 2.692.
- Monolith Modal run killed twice on budget; `ckpt-9000` staged in
  `vast/staging/ckpt-9000/` for resume.
- Modal halted (~$8 left, zero burn). Vast funded with **$10**.
- Plan entering the session: task-first on Vast (validates merge path),
  monolith second.

## 2. Pre-flight work (all landed in-repo)

- **Logging/best-practices audit of `train_task.py`** → fix bundle planned
  (loader heartbeat, loss summary, timestamps, empty-data guard, Vast
  manifest, scheduler comment). Plan approved in principle; **NOT
  implemented** — still pending.
- **Self-closing runs** (`vast/epilogue.sh`, `vast/watchdog.sh`, wired into
  `run-task.sh` / `run-monolith.sh`, README section 7). Design: success →
  HF upload + verify → STOP; failure → logs to HF → STOP; hang (60+ min
  stale log, or orphaned epilogue) → STOP. Never destroys: the container
  credentials lack destroy scope by design, so **no account-level key ever
  lives on a rented box** (uses Vast-set `CONTAINER_ID`/`CONTAINER_API_KEY`).
  Morning destroy is one dashboard click per box.
- **HF repos created** (private, model): `vick1211/bespoke-qwen06b-task-full-001`,
  `vick1211/bespoke-qwen06b-monolith-full-001`,
  `vick1211/bespoke-qwen06b-smoke-test` (epilogue dry-run target).
- **Secrets flow**: single `vast/.env` holding only `HF_TOKEN`
  (gitignored via new root `.gitignore`). Token sourced from the local HF
  CLI cache without ever appearing on screen; scp'd root-only to the box,
  sourced remotely, box-side copy deleted after login caches. Value never
  in chat, commands, or logs — verified by length/format only.

## 3. Host buying decisions

- **Utah $0.240 rejected** (lowest reliability 98.7%, slowest disk/CPU) for
  **Slovakia 1x PRO 4000** ($0.275/hr, 99.90%, ID 56880004,
  51.83.197.242:43042) — saving $0.32 to take 13x the interruption risk
  on an overnight run was the wrong trade.
- **2x offers rejected** (no discount, shared fate, our stack is
  single-GPU). Same reasoning later rejected a hypothetical 4-GPU box:
  ~2x the hourly cost for zero usable gain.
- **Template: PyTorch (Vast).** Plain CUDA = slower from-scratch install;
  vLLM/SGLang/Ollama = inference servers, wrong tool. Jupyter-cert dialog
  dismissed (we never touch Jupyter). Direct SSH over proxy.
- Accidentally bought a second 1x (Austria, stopped, later **destroyed** —
  clean, it never did any work), then bought the monolith replacement:
  **Alberta 1x PRO 4000** ($0.261/hr, 99.55%, m:32499, container 56912404,
  199.126.134.31:45626).

## 4. Environment findings (from reading the full instance agent guide)

- Everything runs in **`/venv/main`** — every script sources it, since
  non-interactive SSH does not. This was a live bug (provision used system
  python); fixed everywhere.
- Image already ships **torch 2.14.1+cu130/cu132** (Blackwell-ready, exceeds
  the 12.8 wheel floor) — provision only adds
  transformers/peft/datasets/accelerate/huggingface_hub. Confirmed via
  `vast-capabilities`, not assumed.
- `/workspace` is **not volume-backed** (`workspace_is_volume: False`) —
  the private HF repos are the persistence, by design.
- 16G container overlay is tight but sufficient (monolith peaked 16% used).
- `huggingface_hub.login()` crashes on fine-grained tokens
  (`KeyError: 'accessToken'`) — provision writes the token cache file
  directly; `snapshot_download` is the real gated-access test.

## 5. Bugs found and fixed (all in-repo)

1. `remove_unused_columns` crash under `torch.compile` (compiled forward
   hides its signature; Trainer stripped every column) — flag added in
   **all three** training scripts.
2. `smoke-compile.sh` exit-code masking (`run_one` always returned 0 via
   trailing `echo`) — rc captured and propagated.
3. Native `resume_from_checkpoint` cannot load adapter-only dirs (demands a
   full-model weight index) — replaced with **manual resume** in
   `train_lora.py`: fail-fast file checks *before* tokenizing, weights via
   `PeftModel.from_pretrained`, step count from `trainer_state.json`,
   remaining-steps cap (47250 = 56250 - 9000).
4. Double-wrap in the first manual-resume attempt (stray `get_peft_model`
   before the branch → forward through frozen weights, backward explosion
   on step 0) — each arm wraps exactly once, plus the proven unfreeze loop
   (`lora_` params forced trainable, fail-loud on zero).
5. `pkill -f` self-match footgun (pattern appears in the killer's own
   cmdline; killed my own session, exit 255) — bracket patterns
   (`[t]rain_task`) from then on.

## 6. Smoke results (all green)

| Smoke | Result |
|---|---|
| compile A/B, Slovakia | plain 37s / 2.0434, compiled 109s / 2.0413, rc=0 both. Loss parity → **compile ON** for full runs |
| compile A/B, Alberta (cu132) | plain 80s / 2.0454, compiled 214s / 2.0408, rc=0. Verdict holds on the new env (slower CPU explains the wall times) |
| task pipeline (full boxes, 3 pairs x 20 steps, compiled) | exit 0, adapter carry-over correct (2,293,760 params), completion marker, zero tracebacks |
| epilogue dry-run | 9.2MB smoke adapter uploaded + size-verified, self-identified instance 56880004, "would stop" printed, nothing mutated |

## 7. Incidents: the two "mystery stops," both explained

Both were the epilogue failure path working as designed — the
understanding lagged, not the system:
- **Slovakia stop #1**: my `pkill` gave the first full run a nonzero exit
  → epilogue uploaded logs → stopped the box. (This run was unwanted
  anyway — only smokes were approved scope. Lesson recorded, not repeated.)
- **Alberta stop #1**: the native-resume crash above → epilogue stopped
  the box with forensics on disk and failure logs verified in the HF
  monolith repo.
- Alberta stop #2 was the double-wrap crash, same loop, same clean stop.

## 8. Live runs (as of last check — both in steady training)

- **Slovakia / task** (`train_task.py`, rank 32, `--compile`): pair 3/3
  (de-en) at 46%, steady **2.18-2.19 it/s**, GPU 92% / 68C. ~1.3h to the
  self-closing finish (upload → verify → stop).
- **Alberta / monolith** (manual resume, step 9000, 47,250 remaining):
  step ~7,250/47,250 (15%), flat **1.99 it/s** post-ramp, loss 1.41
  descending on the fresh scheduler, GPU 86%. ~5.6h left.
- Alberta runs warm (80-82C vs Slovakia's 68C at higher util) — host
  cooling, within spec, watching it each pull; high-80s would change the
  verdict on renting there again.

## 9. Costs (Vast balance $10.00 → $8.84 at last report)

- Spent ~$1.16: smokes, one killed run, idle minutes, ~3.6h of task training.
- Projected: task remainder ~$0.96 (lands ~$7.88), monolith ~$1.90.
  Both jobs ≈ $4 total, ~$6 headroom. Compile running at 2.18 vs the
  budgeted 1.8 is the tailwind.

## 10. Monitoring pattern (agreed)

No sessions held open, no sleeps. Every check is one fast bounded probe
per box — process, log tail, GPU. Nothing needs the agent between checks;
watchdog + epilogue own the endings. Status only on request.

## 11. Pending / next (first-wave snapshot; current plan in §14)

1. Task + monolith completions (self-closing; morning destroys).
2. Logging bundle from section 2 (approved, unimplemented).
3. Full eval board: fr/de base floors (need local server up), then chrF
   all finished adapters all directions vs 57.16/38.98. Deliberately
   deferred mid-run snapshot evals — one clean final comparison.
4. Speedup package for future runs (researched, read-only): verify
   attention backend + dataloader workers (free), eff-batch 64 with scaled
   LR (the halver, chrF-validated), Liger / max-autotune as optional A/Bs.
5. `train_task.py` carries the same native-resume crash risk in its
   `--resume_from` flag (unused so far) — same manual treatment if ever needed.

## 12. Host inventory (last confirmed this session)

| Role | SSH endpoint | Instance | Job & process | Log on box | Output on box | HF repo |
|---|---|---|---|---|---|---|
| Task (Slovakia) | `ssh -p 43042 root@51.83.197.242` ($0.275/hr) | container 56880004 | **COMPLETE exit 0** (2026-10-20T19:02:32Z) — all 3 pairs at ~2.18 it/s; box stopped itself | (was `/workspace/bh/logs/task.log`) | artifacts in HF repo + local `adapters/task-full/` | `vick1211/bespoke-qwen06b-task-full-001` |
| Monolith (Alberta) | `ssh -p 45626 root@199.126.134.31` ($0.261/hr) | container 56912404, m:32499 | **COMPLETE exit 0** (2026-10-20T23:27:47Z) — resumed step 9000, 47,250 more steps at ~1.99 it/s; box stopped itself | (was `/workspace/bh/logs/monolith.log`) | artifacts in HF repo + local `adapters/monolith/` | `vick1211/bespoke-qwen06b-monolith-full-001` |
| Dead | — | Austria box, was stopped | never did any work | — | — | — |
| Local driver | this machine, key `~/.ssh/id_ed25519` | — | issues the probes, holds `vast/.env` (gitignored, HF token only) | repo `logs/` (smoke logs) | repo `adapters/`, `vast/staging/` | — |

Epilogue/watchdog run on each box under its own container credentials;
`monolith-run.log` / `task-run.log` hold the launcher + epilogue echoes.
On completion the epilogue uploads the output dir + log to the box's HF
repo, verifies, and stops the instance — nothing needs fetching by hand.
Fast re-pull pattern (bounded, no sessions held open): process line via
`ps -o pid,etime,cmd -C python3 | grep [t]rain_`, then `tail -c 350` on
the box log, then `nvidia-smi --query-gpu=utilization.gpu,temperature.gpu`.

## 13. Local eval era (same evening)

### GGUF LoRA conversion pipeline

llama.cpp ships no LoRA converter in the Windows bin dir, so the pipeline
was built in-repo: `initial/convert_lora_to_gguf.py` pinned at the b11146
tag (matches the server binary), plus the `conversion/` helper package and
a tag-matched `gguf-py-b11146` (pip's gguf release lacks the DFLASH arch
the b11146 script needs — used via PYTHONPATH). `--base` needs an HF
model dir (`config.json`), not a GGUF file — fetched Qwen3-0.6B-Base
config with the cached HF token, deleted after. Scratch API-listing files
removed; helper scripts kept for re-conversion. Outputs: task/monolith
18.4MB (224 tensors = 28 layers x 4 proj x A/B, f16), box-{ro,en} 9.18MB
each (rank 16 = half). Verified structurally (tensor counts/names) before
first load.

### Launcher + server lessons

- `start-local-qwen06b.ps1` gained `-Lora` (string array). b11146 takes
  ONE `--lora` with comma-joined paths — repeated flags silently keep
  only the last, caught before a misleading eval could run.
- Port 8081 was squatted by the Codex CLI (listening on 127.0.0.1:8081
  since the prior day). The server bound 0.0.0.0:8081, looked healthy,
  got zero traffic; evals died with connection resets. Moved to 8082.
  Rule earned: when a live server is unreachable but the process looks
  fine, identify the actual listener (netstat + process lookup) before
  blaming the model stack.
- Invoking the ps1 wrappers via ssh-exec powershell leaves
  `$PSScriptRoot` empty — pass `-RepoRoot` explicitly.
- `$RepoRoot` quirk aside, the eval wrapper resolves everything else
  itself; `-Output` keeps adapter runs off the floor files, and
  incremental resume makes every run restart-safe.

### Eval results (canonical: results/eval-board.csv + .xlsx via build_board.py)

| Pair | Setup | chrF | Delta vs floor |
|---|---|---|---|
| ro-en | base floor | 57.16 | — |
| en-ro | base floor | 38.98 | — |
| de-en | base floor | 60.43 | — |
| ro-en | task-only, n=10 smoke | 39.99 | n/a (noise; engagement proof only — outputs fluent) |
| en-ro | task-only | 29.36 | -9.62 |
| en-ro | task+box-ro+box-en | 33.11 | -5.87 |
| de-en | task+box-ro+box-en | 37.88 | -22.55 |
| en-ro | monolith | 38.94 | -0.04 (retracted; 59.0% looped — score coincidence, not preserved skill) |
| ro-en | monolith | 38.60 | -18.56 (91.9% looped) |
| de-en | monolith | 38.62 | -21.81 (94.1% looped) |

Floors and adapter runs all n=1012 devtest except the ro-en smoke. Loop
rates come from `initial/loop_check.py`: an 8-word window repeated 3+ times. The
task stack is base + one GGUF LoRA (no boxes = skill alone) or base +
three LoRAs via the single comma-joined flag (full training stack); the
monolith stack is base + one LoRA, no boxes involved. fr pairs untouched
(no base floors, no adapter runs).

### Hypothesis arc (corrected by §14)

- Forgetting (sequential overwrite of one shared adapter): ROSE on the
en-ro pattern (task trails floor), DIED on de-en — the last-trained pair
shows the worst gap, retaining 63% of its floor vs 85% for first-trained
en-ro. Recency explains nothing.
- The monolith en-ro 38.94 was then misread as vindication for
interleaved training. Retracted: output inspection plus
`initial/loop_check.py` showed the monolith also degenerates. Same
number, different behavior — not preserved skill.
- Current lead: termination failure, not only translation quality. The
adapters translate the first clause reasonably, then fail to stop: greedy
decoding falls into repetition attractors, while sampling wanders past
where the answer should have ended. See §14 for the loop table, probe
history, base control, training-loss evidence, and the EOS tune-up plan.

### Decisions taken

- Control board first (monolith ro-en + de-en locally, free). Completed:
38.60 and 38.62, both heavily looped. The planned interleave retrain is
superseded: the next test is an EOS termination tune-up, not a full
retrain. Chosen via explicit user approval of the control-board-first
sequence.
- Full ro-en adapter run deprioritized (sequential adapter a dead end;
the monolith control settles it).
- Slovakia task box approved for user destroy-click; Alberta monolith box
stopped itself (exit 0 verified), needs destroy click.
- Standing rule added as vast/README.md section 8: read the full instance
agent guide before the first substantive command on any new box.

### Costs

Balance $8.84 mid-task-evening; both jobs have since run to completion
(final accounting pending — record from the Vast dashboard).

Section 11 status: item 1 done (both completions, self-closed); item 2
logging bundle still unapproved; item 3 eval board now complete for
task/monolith on ro/en/de floors (fr untouched), with loop-rate warnings;
item 4 speedup research still open.

## 14. Termination failure, probes, and current plan (correction to §13)

### What the monolith numbers really mean

The control-board completion overturned the §13 vindication story:
monolith ro-en scored 38.60 and de-en 38.62, far below floors 57.16 and
60.43. More importantly, all adapter outputs under the greedy harness
are dominated by repetition loops, while base outputs are not.

| Setup | Loop rate under greedy temp 0 |
|---|---|
| base ro-en / en-ro / de-en | 0.1% / 6.5% / 0.0% |
| task en-ro (alone → +boxes) | 74.3% → 58.3% |
| task+boxes de-en | 95.0% |
| monolith en-ro / ro-en / de-en | 59.0% / 91.9% / 94.1% |

Method: `initial/loop_check.py` flags an output when an 8-word window
repeats 3+ times. The identical 38.60/38.62 pair is therefore not a
translation measurement converging; it is looped output converging to
the same chrF band. Boxes reduce looping (74.3% → 58.3% on task en-ro),
so the full stack still matters, but no tested adapter terminates like
the base.

### Probe history

- v1 sampling probe used ASCII-folded inputs (diacritics stripped to dodge
a console bug) and therefore mangled the test inputs; its “skill present”
read was premature.
- v2 (`initial/sampling_probe.py`) uses proper UTF-8 inputs and two
conditions: greedy + repetition penalty 1.1, and temp 0.7 + penalty 1.1.
Result on monolith LoRA: penalty breaks exact repetition into-English,
but outputs still run 250–600 chars and drift; en-ro still loops 2/2
even with the penalty.
- Base control (`results/sampling-probe-base.jsonl`, same script, plain
base on the same port): 12/12 clean terminated translations, 13–102 chars
(e.g. “We have now four-month-old mice that had diabetes and no longer
do.”). Same prompts, same decoding — only the LoRA differs. The
termination breakage is adapter-induced.

### Why EOS is now the lead

Monolith training itself was healthy: final train loss 1.4067, declining
to ~1.3, stable gradients, clean LR decay, 47,250 steps in ~6.8h. Healthy
teacher-forced loss plus degenerate generation is the classic signature
of missing termination training. The recipe gap is concrete:
`initial/train_lora.py::tokenize_fn` masks the prompt, supervises only
completion tokens, then pads with ignored `-100` labels — the model never
trains on what to emit when finished (no EOS/newline target). Teacher
forcing never tests stopping, so low loss coexists with broken endings.

### Current plan (discussion-only until the new node exists)

- Buy one single 1x RTX PRO 4000-class 24GB Vast node. No 2x: our stack
is single-GPU, so the second GPU adds cost and shared fate without speed.
- Run the EOS termination tune-up first: same monolith adapter resume
path, `--max_steps 1000` (~8 minutes), EOS/newline target with real loss.
Measure loop rate on a small local probe subset before deciding more.
- Only escalate step count on evidence (e.g. 3,000 if loops persist);
more steps are not automatically better because every extra step reshapes
the translation weights the tune-up is trying to preserve.
- Benchmark on the same Vast machine only after training exits and the
GPU is idle. Training and inference will not overlap on one GPU: overlap
distorts timing/stability and breaks comparability.
- Local floors remain the primary comparison; the Vast benchmark is a
secondary parity check because CUDA/Vulkan and machine setup can differ.
- Stop the container only after artifacts are uploaded and verified; the
Vast dashboard destroy remains a separate user click.
- Destroys still pending: Slovakia task box (approved) and Alberta
monolith box (stopped itself at exit 0, needs destroy click). Final Vast
accounting still pending from the dashboard.

### Postscript: the 401, not preemption (2026-10-22)

The resurrected tune-up box (38.87.238.254:44251) settled it. Forensics
from its preserved disk: training completed cleanly — 1000/1000 steps at
1.33 it/s, final train loss 1.3059 (below the monolith's 1.41), "Saved
LoRA" marker, adapter + checkpoints intact. The killer was the epilogue:
`HfHubHTTPError: 401 Unauthorized` on `repos/create` ("Invalid username
or password"), so no repo was ever made and the box stopped with the
artifact stranded on disk. The preemption theory is retracted for this
box — and suspect for the previous one, which died the same silent way.
Provision only ever verified *read* access (`snapshot_download` of the
gated base); write access was assumed and absent. Lesson recorded: future
provisioning must verify *write* auth up front (a harmless `create_repo
exist_ok` on the target repo) and fail fast, or the finish funnel dies
after the GPU money is spent. Standing user-side fix: refresh `vast/.env`
`HF_TOKEN` with a write-capable token.
- Recovered by hand: adapter + train.log pulled via scp to
`adapters/eos-tune-001/`, converted to `adapters/eos-tune-lora.gguf`
(18.4MB), uploaded to `vick1211/bespoke-qwen06b-monolith-eos-001` from
the local machine (local CLI token has write; upload verified by commit).
- Redundant replacement box (199.126.134.31:37152, launched while the
outcome was unknown) stopped via container-key self-stop; disk kept.
- Vast-side termination probe (greedy, transformers, 6 prompts): base
1/6 looped, tuned 0/6 — en-ro terminates in ~21-25 tokens where the
untuned monolith looped 59-92%. Two into-English probes still run to the
128-token cap without exact repetition; quality verdict awaits the local
chrF board.

### Postscript: the cartridge wins (2026-10-22)

Full chrF board for the tuned adapter, run on the same Vast box under
llama.cpp b11146-CUDA with the identical harness settings (temp 0,
max 128, stop newline) via `initial/eval_offline.py` against the same
FLORES rows — then reproduced locally with `--score_only` to the digit:

| Pair | Tuned chrF | Floor | Delta | Looped |
|---|---|---|---|---|
| en-ro | 44.36 | 38.98 | +5.38 | 0.8% |
| ro-en | 59.34 | 57.16 | +2.18 | 0.0% |
| de-en | 59.90 | 60.43 | -0.53 (tie) | 0.0% |

The cartridge (interleaved monolith + 1,000-step EOS tune-up) meets or
beats the naked base on every tested pair while terminating cleanly.
This is the vindication the untuned monolith's 38.94 falsely claimed:
score and termination together. Records banked as
`logs/eval-eostune-*-devtest.jsonl`; board rebuilt at 13 rows. Tune box
stopped via container-key self-stop after banking; dashboard destroy
remains a user click. Task arm and interleave retrain are superseded —
sequential training without EOS targets is the confirmed-dead path.

### Postscript: de-fr does not compose (2026-10-22)

Zero-shot pairing test, same box, same harness (`initial/eval_offline.py`
against identical FLORES rows, reproduced locally via `--score_only`):
base de-fr floor 47.57 (0.9% looped) vs tuned cartridge 39.55 (-8.02,
0.8% looped). Termination is clean; quality is not. The cartridge's
skill is English-anchored: X<->en training improved X<->en but damaged
the unseen de->fr direction relative to the naked base. Boundary of the
v0 claim established: cartridges transfer across trained pairings, not
across untrained ones — at least under this recipe. Records banked as
`logs/eval-base-de-fr-devtest.jsonl` +
`logs/eval-eostune-de-fr-devtest.jsonl` (pairs input:
`results/pairs-de-fr-devtest.jsonl`); board rebuilt at 15 rows. Bench box
stopped via container-key self-stop after banking; dashboard destroy
remains a user click.

### Postscript: the pivot composes (2026-10-23)

Two-leg de-en-fr evals on a fresh PRO 4000 box (`initial/eval_pivot.py`,
leg 1 de->en, leg 2 en->fr, same greedy harness, same 1,012 rows):
base pivot 47.70 (matches its 47.57 direct floor; pivoting costs a
healthy model nothing) and tuned pivot **49.11** (+1.54 over floor,
+1.41 over the base pivot, +9.56 over the tuned direct run; 0.2%
English-looking, 0.3% looped). The explicit English intermediate
routes around the target-language control failure: both legs are
trained directions and both are strong, so chaining them composes
where the direct mapping cannot. This confirms the de-fr diagnosis
mechanistically — the knowledge is present in both legs, only the
direct routing is broken. Records banked as
`logs/eval-pivot-base-de-fr-devtest.jsonl` +
`logs/eval-pivot-tuned-de-fr-devtest.jsonl` (each row also keeps the
`hyp_en` intermediate); board rebuilt at 17 rows. Bench box stopped
via container-key self-stop after banking; dashboard destroy remains
a user click.

### Postscript: the scratchpad fails on the tuned adapter (2026-10-23)

Think-in-English test: one generation per row (English draft, then
French final, `initial/eval_scratchpad.py`), parsed on the French
marker. Base follows the format on 94.6% of rows and scores 45.90
(below its 47.57 direct floor — the scratchpad buys a healthy model
nothing). The tuned adapter follows it on 0% — not from incompetence
but from the EOS cure itself: it emits EOS the moment the English
clause looks complete and never reaches the French section (verified
on raw completions, including a one-line `//` format variant). The
termination training overgeneralized: "stop when the answer looks
complete" cannot distinguish an intermediate answer from a final one.
Outcome 3 of the three predicted. Internalizing the pivot would need
a distinct continue-vs-finish signal (e.g. EOS only after final
answers); the pipeline pivot at 49.11 stands as the working answer.
Records banked as `logs/eval-scratch-base-de-fr-devtest.jsonl`;
board rebuilt at 18 rows. Bench box stopped via container-key
self-stop after banking; dashboard destroy remains a user click.
