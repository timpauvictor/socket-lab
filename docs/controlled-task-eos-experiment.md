# Task-adapter EOS tune-up

Fine-tune one copy of the existing translation task adapter to emit EOS after
each answer. Keep the base model and language cartridges frozen. Preserve the
original adapter and save the EOS-tuned version separately.

## Workflow

1. Save fixed copies of the original adapter, cartridges and selected training data.
2. Run a short EOS smoke test: three steps per language pair, nine total. Check
   that gradients are finite, task weights changed and frozen weights did not.
3. With `--full`, start again from the original adapter and train
   333/333/334 steps on ro-en/fr-en/de-en, including both directions: 1,000 total.
4. Check original versus EOS-tuned on two saved examples in each of en-ro,
   ro-en and de-en. Retain generated tokens, EOS/cap reason, repetition and errors.
5. Upload and hash-verify the private bundle, retain all artifacts and stop Vast.

The final adapter is `OUTPUT/bundle/full/task-eos/`. Smoke outputs are disposable
and never initialize full training. The existing modular trainer uses the
appropriate frozen cartridges and a fresh optimizer/scheduler for each pair.

This is a small termination diagnostic, not a full translation benchmark or a
causal EOS comparison. French is trained but not generation-checked. EOS
termination and improved translations are not guaranteed.

See [running instructions](running.md#eos-only-task-adapter-tune-up-local-implementation)
and the [Vast procedure](../cloud/vast/README.md#eos-only-task-adapter-tune-up-separate-workflow)
for commands, source records, approval before a paid run, remaining budgets
and how to download the files.
Local checks do not establish real GPU training success. Confirm final stop and
billing separately; stopped storage may still bill.
