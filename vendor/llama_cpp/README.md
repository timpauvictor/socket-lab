# Bundled llama.cpp conversion support

This directory holds the converter and support modules previously under
`initial/`. They are bundled third-party infrastructure, not an original
model or training framework developed for this experiment.

- `convert_lora_to_gguf.py`: PEFT adapter conversion entry point.
- `conversion/`: model-specific conversion support.
- `gguf-py-b11146/`: bundled GGUF reader/writer and tensor utilities.

This cleanup only relocates these files. Their bytes, existing notices,
relative imports, and local GGUF lookup remain unchanged. The directory
name identifies the bundled b11146 GGUF code; it does not establish an
independently verified upstream commit for every conversion file.

The existing local-path adjustment in the converter and `conversion/base.py`
selects the adjacent `gguf-py-b11146` directory. Keep that layout intact.
Third-party code retains its applicable terms; the project's root license
does not replace them.

See [running instructions](../../docs/running.md) for the conversion command.
Conversion requires its ML dependencies and actual adapter weights; it is
separate from CPU-only rescoring and report generation.
