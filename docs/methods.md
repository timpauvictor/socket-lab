# Methods

Training uses `Qwen/Qwen3-0.6B-Base`. The optimizer updates LoRA parameters, not the original model parameters. Adapters target `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

## Recorded training configurations

- Language adapters: one rank-16 adapter per language, trained on `wikimedia/wikipedia` text. The full recipe samples up to 100,000 blocks of 256 tokens. The sampler only scans an initial part of the dump. Articles after that part cannot be selected, so this is not a uniform sample of all Wikipedia.
- Sequential task adapter: rank 32, trained on Romanian-English, French-English, and German-English pairs in that order. Each cycle loads a fresh base, merges the two language adapters, then continues the shared task adapter.
- Monolithic adapter: rank 32, trained on interleaved OPUS-100 pairs in both directions. The full budget is 150,000 filtered pairs per language pair and two epochs.
- EOS-tuned monolithic adapter: the completed monolithic adapter received 1,000 additional steps at learning rate `5e-5`, with explicit end-of-sequence labels. The original task and monolithic runs did not use these labels.

The sequential task code uses PEFT `combination_type="linear"`. This combines the low-rank factors; it is not an exact sum of adapter deltas. The historical setting remains unchanged. The recorded German-English task-plus-language evaluation also loaded Romanian and English language adapters, not German and English adapters. These differences limit conclusions about independent adapter composition.

## Evaluation

The direct harness sends raw `/v1/completions` requests with `temperature=0`, `max_new_tokens=128`, and `stop=["\n"]`. It scores the first output line with sacreBLEU chrF 2.6.0.

The pivot harness makes two direct requests: German to English, then English to French. The scratchpad harness uses a different prompt, a 256-token limit, and no newline stop. It extracts the final-language section from one generation. These are different evaluation strategies, not identical decoding conditions.

The full runs in the table use the same 1,012 source/reference pairs for each direction. One Romanian-English task-only run has just 10 examples. Local and rented-server runs used different hardware and software backends. Old file headers are missing the adapter ID, generation limits, strategy and reason generation stopped. The journal and table fill in some of those details, but not all of them.

New runs require a caller-supplied `--run_id` and record strategy, token limit, stop strings, pivot language, and the offline pair-file hash. This prevents accidental resume across declared configurations. It does not verify that the server actually loaded the declared weights. Use a fresh output file for legacy records, which remain scoreable.

## Interpretation

The EOS-tuned single adapter scored higher on two tested training directions and repeated text less often. German-English was still 0.53 chrF below the base. The adapter got more training and new end-of-answer labels at the same time. I didn't run a matching extra-training control, so I can't tell which change helped.

Low training loss does not establish reliable generation. The recorded scores do not rule out forgetting, establish general skill composition, or validate the historical merge method. See [limits](limits.md) and the preserved [journal](session-2026-10-20-vast.md).
