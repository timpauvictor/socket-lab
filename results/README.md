# Results and reproduction

## Matched German–French check — 2026-10-28

[Exact-score summary](german-french-matched-2026-10-28.json): six runs on the same 1,012 saved FLORES+ devtest pairs and the same RTX PRO 4000 Blackwell machine. All used the same Qwen3-0.6B-Base Q8 model, llama.cpp b11146 (`7fe450e19`), CUDA 12.8, temperature 0, a newline stop, and a 128-token limit per leg. Full runs used one worker, direct then pivot sequentially for each setup, with fresh servers. Cached greedy outputs varied in startup checks; this is not a bit-identical reproducibility claim.

| Setup | Direct German → French chrF | Manual English-pivot chrF |
|---|---:|---:|
| Base | 47.54 | 47.84 |
| EOS monolith | 39.17 | 48.95 |
| EOS cartridges | 26.41 | 47.46 |

The cartridges use the same EOS-tuned task adapter with German+French directly, or German+English then English+French. Each leg uses one combined adapter, not separate competing language servers. The pivot is explicit manual routing, not learned routing or demonstrated direct composability. The monolith had 28 empty final outputs directly and 3 through the pivot; the other four runs had none. No empty examples were filtered. These chrF scores do not measure EOS termination, semantic correctness, statistical significance, or AMD performance.

The JSON includes exact scores, counts, runtime/input/source identities, and the filenames and SHA256 hashes of six **private** record files. It contains no source/reference text, generated translations, private paths, or server endpoints. Public readers cannot recompute these six scores from hashes alone. The summary keeps the code IDs recorded for the runs, including the pivot change that wasn't committed yet. Later code fixes did not produce these scores.

### Attribution and access

- Dataset: [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus), maintained by the [Open Language Data Initiative](https://oldi.org), declares [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Its current access conditions include agreement not to rehost its text where web crawlers can pick it up. The six new raw logs and frozen source/reference pairs are therefore not redistributed in this update. Dataset access remains subject to its own approval and terms; this project's Apache-2.0 license does not replace them.
- Base model: [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), Qwen team, declares Apache-2.0.
- Inference runtime: [llama.cpp b11146](https://github.com/ggml-org/llama.cpp/releases/tag/b11146); upstream and bundled vendor attribution remain separate from first-party code. No weights or runtime binaries are added here.

## Historical comparison table

`eval-board.csv` is the historical banked comparison table, not a combined table of all later runs. Its 18 rows identify direction, configuration, sample count, chrF, evaluation file, adapter files, and notes. Seventeen rows have 1,012 examples; the task-only Romanian-English smoke row has 10. Historical notes contain interpretations that later review limits; the saved scores are kept. The CSV, workbook, figure data and charts are unchanged. FLORES pairs, raw translation logs and old probe outputs aren't included; I've kept the originals locally.

## Generate historical presentation files

```bash
python -m pip install openpyxl matplotlib
python reporting/build_board.py
python reporting/plot_board.py
```

These commands read the CSV. They do not recompute translation scores, run inference, train a model, or access a dataset. Use `--output` for the workbook or `--output_dir` for figures to write verification copies elsewhere.

`figure-data.csv` contains the plotted values. The repetition figure selects the highest recorded untuned rate for each direction from full non-base runs. It compares different configurations with the EOS-tuned monolithic adapter, not one controlled before-and-after experiment.

## Recompute metrics from records

You need your own local records and permission to use them for these examples. The old files named below aren't included.

```bash
python -m pip install sacrebleu==2.6.0
python evaluation/eval_llama.py --score_only --output logs/eval-eostune-en-ro-devtest.jsonl
python evaluation/loop_check.py
```

chrF signature: `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`.

The repetition diagnostic requires at least 24 whitespace-separated words. It flags a hypothesis if an exact eight-word window occurs at least three times, counting overlapping windows. Matching is case-sensitive and retains punctuation. It is not a measurement of EOS termination or a token-limit hit. The scripts also report the fraction of hypotheses longer than 400 characters; that is a separate diagnostic.

Direct and pivot requests use `temperature=0`, a 128-token limit per request, and a newline stop. Scratchpad evaluation uses a different prompt, 256 tokens, and no newline stop. Historical records omit finish reasons and token counts.

## Small probes

The old probe outputs included text from the dataset, so I've kept them locally rather than sharing them here. The probe now starts with six made-up sentences, not the old FLORES prompts. That means new probe outputs aren't directly comparable to the old ones. I haven't rerun them to replace the old results.
