# Limits

- The EOS-tuned monolithic adapter improves English-Romanian in both directions. German-English is 0.53 chrF below the base; no significance test establishes a tie.
- Direct German-French translation drops from 47.57 to 39.55 chrF. Two-step translation through English reaches 49.11 with the tuned adapter and 47.70 with the base. This tests routing through one adapter, not composition of independently trained adapters.
- Forgetting is not ruled out. A poor score on the last-trained direction cannot isolate forgetting from repetition, merge differences, data order, or other training effects.
- Repetition is a text-window diagnostic, not a count of EOS failures or output-limit hits. Historical records contain neither finish reasons nor generated token counts.
- The EOS tune-up used additional training and a changed target format. The records do not isolate which change caused the observed improvement.
- The tuned scratchpad probes failed to produce the requested final-language section. The journal's EOS-overgeneralization explanation is a hypothesis, not a controlled causal result.
- There is one seed per configuration, no uncertainty estimate, and no human translation assessment. The reported metric is chrF on FLORES devtest, not performance on other domains or long text.
- Historical task training uses a PEFT linear factor merge, while llama.cpp evaluation stacks adapter deltas. The German-English task-plus-language row uses Romanian and English language adapters. These are not matched composition tests.
- The historical tokenizers could mask all labels when a prompt filled the 256-token budget. Current validation rejects these examples instead of silently training on them. Their frequency in the original corpus was not measured. Training settings and banked artifacts remain unchanged.
- Model and dataset revisions and all environment details are not pinned across every recorded run. Some weight references require access that could not be verified anonymously.

Code, style transfer, automatic adapter selection, and a hypernetwork remain proposed work, not demonstrated features of this repository.
