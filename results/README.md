# Results and reproduction

`eval-board.csv` is the banked comparison table. Its 18 rows identify direction, configuration, sample count, chrF, evaluation file, adapter files, and notes. Seventeen rows have 1,012 examples; the task-only Romanian-English smoke row has 10. Historical notes contain interpretations that later review limits; the measurements and records are preserved.

## Generate presentation files

```bash
python -m pip install openpyxl matplotlib
python reporting/build_board.py
python reporting/plot_board.py
```

These commands read the CSV. They do not recompute translation scores, run inference, train a model, or access a dataset. Use `--output` for the workbook or `--output_dir` for figures to write verification copies elsewhere.

`figure-data.csv` contains the plotted values. The repetition figure selects the highest recorded untuned rate for each direction from full non-base runs. It compares different configurations with the EOS-tuned monolithic adapter, not one controlled before-and-after experiment.

## Recompute metrics from records

```bash
python -m pip install sacrebleu==2.6.0
python evaluation/eval_llama.py --score_only --output logs/eval-eostune-en-ro-devtest.jsonl
python evaluation/loop_check.py
```

chrF signature: `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`.

The repetition diagnostic requires at least 24 whitespace-separated words. It flags a hypothesis if an exact eight-word window occurs at least three times, counting overlapping windows. Matching is case-sensitive and retains punctuation. It is not a measurement of EOS termination or a token-limit hit. The scripts also report the fraction of hypotheses longer than 400 characters; that is a separate diagnostic.

Direct and pivot requests use `temperature=0`, a 128-token limit per request, and a newline stop. Scratchpad evaluation uses a different prompt, 256 tokens, and no newline stop. Historical records omit finish reasons and token counts.

## Small probes

`sampling-probe.jsonl` and `sampling-probe-base.jsonl` contain six hand-written prompts under two sampling settings. They are diagnostic material, not full FLORES comparisons. Preserve them and the journal as historical evidence rather than replacing them with new measurements.
