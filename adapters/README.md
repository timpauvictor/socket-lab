# Adapters

Weights are not vendored in git. Use these Hugging Face sources:

- Task cartridge: `vick1211/bespoke-qwen06b-task-full-001`
- Monolith cartridge: `vick1211/bespoke-qwen06b-monolith-full-001`
- Termination-tuned monolith: `vick1211/bespoke-qwen06b-monolith-eos-001`
  - Verified commit used locally: `47a09863`

Local `adapters/*/` folders keep small manifests and run metadata only.
`*.safetensors`, `*.gguf`, optimizer states, checkpoints, and `vast/staging/`
are intentionally ignored.
