# socket-lab

## The idea

I'm an amateur, not a researcher. I had an idea: could a language model work like a console, with small "cartridges" that add different abilities?

The existing model would be the socket. The cartridges would be **adapters**: extra trained weights used with the model, rather than replacements for the whole model.

I tried a small translation experiment to explore this. This repository contains the scripts, scores, and notes. It is a record of the attempt, not proof that the cartridge idea works.

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

Each comparison below used 1,012 examples from the FLORES translation test set.

| Translation direction | Original model's chrF | Model with the later adapter |
|---|---:|---:|
| English → Romanian | 38.98 | 44.36 |
| Romanian → English | 57.16 | 59.34 |
| German → English | 60.43 | 59.90 |

English–Romanian scored higher in both directions. German–English scored slightly lower. This is a mixed result, not an improvement everywhere.

German–French was another warning against a broad success claim. Direct translation scored **47.57** with the original model and **39.55** with the later adapter. Translating into English first, then translating that result into French in a second request, scored **47.70** and **49.11**, respectively. That is a manually chosen two-step process, not the model learning to choose cartridges.

Repeated text was much less common after the extra training. However, the runs do not separate the effect of the end-of-text marker from the effect of doing more training.

The [full results table](results/eval-board.csv) includes unsuccessful attempts too. It has 18 rows; one is a short 10-example check, not a full comparison. Recalculating chrF from the saved translations reproduced all 18 recorded scores to two decimal places. That checks the scoring, not the whole experiment.

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
