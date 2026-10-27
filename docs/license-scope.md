# License scope: v0 vs v1

Decision: keep v0 and v1 in separate repositories with different licenses.

- v0 publication repo: permissive Apache-2.0.
  Contents: frozen-socket cartridge recipe, eval harness, results board,
  session journal, methods/limits.
- v1 hypernetwork repo: AGPL-3.0-or-later.
  Contents: the mint/hypernetwork code that issues cartridges.

Rules:

- Dependency may flow v0 -> v1, not v1 -> v0.
- Do not copy AGPL v1 code back into the permissive v0 repo.
- Copyright holder: Victor Timpau
  (Hugging Face: `vick1211`; GitHub: `timpauvictor`).
