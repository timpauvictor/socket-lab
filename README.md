# socket-lab

## The idea

I'm an amateur, not a researcher. I had an idea: could a language model work like a console, with small "cartridges" that add different abilities?

The existing model would be the socket. The cartridges would be **adapters**: extra trained weights used with the model, rather than replacements for the whole model.

I tried a small translation experiment to explore this. This repository contains the scripts, scores, and notes. It is a record of the attempt, not proof that the cartridge idea works. I used LLM assistance with the code, checks, and writing.

## What I tried

I used [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), an existing small language model. The training changed the adapters, not the original model's weights.

There were two main approaches:

- Train separate language adapters and a translation adapter, then try using them together.
- Train one adapter on several translation directions instead of combining separate language adapters.

The languages were English, Romanian, French, and German. Translation training paired Romanian, French, and German with English, in both directions. German-to-French was not a direct training pair.

Early outputs often repeated text. The single adapter later received extra training that included an explicit end-of-text marker. The results below use that later version, called `eos-tune` in the saved files.

My honest takeaway: **I didn't get the separate cartridges working better, but I did get a single adapter to improve some translation results.**

## What happened

The translations were scored with **chrF**. It compares short sequences of characters in a generated translation with a reference translation. Higher scores mean more overlap. It is not a percentage of correct translations, and it cannot tell us by itself whether a translation gets the meaning right.

### Latest German–French check

I later compared all three setups on the same machine, using the same 1,012 FLORES devtest examples for each run:

| Setup | Direct German → French | German → English → French |
|---|---:|---:|
| Original model | 47.54 | 47.84 |
| Single EOS-tuned adapter | 39.17 | 48.95 |
| EOS-tuned cartridges | 26.41 | 47.46 |

The cartridges did much worse directly and, even through English, didn't beat the original model's score. This didn't demonstrate the cartridge idea working, but it doesn't prove that every version of the idea must fail. I haven't established whether the small score differences are reliable.

The task adapter used with the cartridges also received extra EOS training. Direct cartridges combined German and French; the two-step route combined German and English, then English and French. I chose that route manually; the model didn't learn to choose it.

All six runs used the same RTX PRO 4000 Blackwell GPU, Q8 base model, and llama.cpp b11146/CUDA 12.8 runtime. Each request used temperature 0, a 128-token limit **per leg**, and a newline stop. These are translation scores, not measurements of EOS termination. The single-adapter runs had 28 empty direct outputs and 3 empty two-step outputs; they stayed in the scores. The other four runs had none.

The [matched summary](results/german-french-matched-2026-10-28.json) has exact scores, settings, and file hashes. I'm keeping these six raw logs private because FLORES+ access terms restrict rehosting its text where web crawlers can pick it up. Hashes identify the saved files; they don't let readers independently recompute the scores.

### Earlier results (historical)

These older comparisons were not fully controlled and are separate from the matched check above. Each used 1,012 examples from the FLORES translation test set.

| Translation direction | Original model's chrF | Model with the later adapter |
|---|---:|---:|
| English → Romanian | 38.98 | 44.36 |
| Romanian → English | 57.16 | 59.34 |
| German → English | 60.43 | 59.90 |

English–Romanian scored higher in both directions. German–English scored slightly lower. This is a mixed result, not an improvement everywhere.

German–French was another warning against a broad success claim. Direct translation scored **47.57** with the original model and **39.55** with the later adapter. Translating into English first, then translating that result into French in a second request, scored **47.70** and **49.11**, respectively. That is a manually chosen two-step process, not the model learning to choose cartridges.

Repeated text was much less common after the extra training. However, the runs do not separate the effect of the end-of-text marker from the effect of doing more training.

The [historical results table](results/eval-board.csv) includes unsuccessful attempts too. It has 18 rows; one is a short 10-example check, not a full comparison. Recalculating chrF from the saved translations reproduced all 18 recorded scores to two decimal places. That checks the scoring, not the whole experiment.

## What this doesn't prove

- Separate cartridges can reliably work together. That was the idea, but it has not been demonstrated here.
- The model keeps all its previous abilities after adapter training. These tests do not establish that.
- The translations are good just because a score is higher. chrF is not a human review of their meaning.
- The same results will hold on other text, tasks, or repeated training runs. Those checks have not been done.

There are also known problems with the comparisons. The separate-adapter approach combined adapters differently during training and evaluation. One German–English run used Romanian and English language adapters. Some older records do not capture enough settings to repeat the run exactly.

Those are reasons to be cautious, not details to hide. The [methods](docs/methods.md), [limitations](docs/limits.md), and [experiment journal](docs/session-2026-10-20-vast.md) contain more detail. The journal includes earlier interpretations that were later qualified or corrected.

## Check a saved result

FLORES pairs, raw translation logs and old probe outputs aren't included here. I've kept the original files locally. You need your own local records for this example, and permission to use the data.

With Python and `uv` installed, run this from the repository root:

```text
python run_eval.py --score-only --output logs/eval-eostune-en-ro-devtest.jsonl
```

This recalculates chrF on the CPU from saved translations without changing the file. It needs no model, server, or dataset download. `uv` supplies the pinned scoring package and may download that package on first use. See the [running instructions](docs/running.md) for using an existing Python environment, starting inference, or training.

## Files, instructions, and licenses

- [Results](results/): score table, charts, and saved comparison data.
- [Logs](logs/): selected training logs; raw translation records are not distributed.
- [Training](training/), [evaluation](evaluation/), and [reporting](reporting/): first-party Python scripts. See [running instructions](docs/running.md).
- [PyTorch guide](docs/pytorch-guide.html): how this project's training code works, with small CPU examples and limits on what the experience demonstrates.
- [Bundled conversion tools](vendor/llama_cpp/): third-party GGUF support, kept separate from the experiment code.
- [Adapters](adapters/README.md): configurations and links to weights. The weights are not included here, and some linked sources may require access.
- [Remote training scripts](cloud/modal/): running these jobs can incur charges. There are also [Vast scripts](cloud/vast/).

Reading the saved results does not require training a model. Running the models yourself needs additional software and model files; see the linked instructions rather than treating this README as a complete setup guide.

The project's v0 code uses [Apache-2.0](LICENSE). Models, datasets, and third-party code have their own terms. The Qwen base model declares Apache-2.0; FLORES+ declares CC BY-SA 4.0 and requires access approval. Wikipedia and OPUS training sources have their own license and attribution requirements. This repository's license does not replace those terms or grant rights to separately hosted weights.

A proposed future model that generates adapters is not implemented here. Its separate licensing plan is described in [license scope](docs/license-scope.md).

Copyright (C) 2026 Victor Timpau.
