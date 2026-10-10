# Lodestone

Lodestone is the model layer of a local-first AI stack. It looks after the language, vision, speech and embedding
models a household or a small site runs on its own hardware, and gives everything above it one steady place to ask
"which models do I have, where are they, and may I use that GPU right now?". Pure Python 3.10+ standard library: no
dependencies.

## Where it fits

Lodestone is layer 0 of a three-layer pyramid. Dependencies point down only: a layer never imports the one above it.

| Layer | Repository | Job |
| --- | --- | --- |
| 2 | [Hearth](https://github.com/lottage/hearth) | Apps, starting with a Home Assistant house assistant |
| 1 | [StoneSage](https://github.com/lottage/stonesage) | Agent platform: souls, tools, approvals, memory |
| 0 | **Lodestone** (this repository) | Serves the models |

## What is in it

- **Engines and options** (`engine_controller`, `engine_options`): control llama.cpp servers and read every option the
  installed build accepts from the build itself, instead of a hand-kept list.
- **Model loader and downloader** (`model_loader`, `model_downloader`): choose a model for an engine, size it against
  the real GPU or unified memory, review the exact change, apply it with a backup, a health check and automatic
  rollback. Downloads GGUF files directly or from Hugging Face with atomic writes.
- **Engine profiles** (`engine_profiles`): named bundles of "which model on which engine", applied one target at a time.
- **Engine lease and idle gate** (`engine_lease`, `idle_gate`): lend a GPU to a heavier job for a limited time with a
  dead man's switch that hands it back, and only when the machine has actually been idle.
- **Provenance** (`provenance`, `provenance_scan`): what every model file is (hash, lineage, licence) and which
  llama.cpp build can load it.
- **System profile** (`system_profile`): the live hardware, engines and models, so nothing has to be written down by hand.
- **Boost** (`boost`): a pool of free cloud inference behind one router, with per-source quota accounting and an egress
  rule for what may leave the site. The local engine stays the default and the last fallback.
- **Endpoints and site** (`endpoints`, `site`, `config_mask`): every service address comes from configuration, an
  environment variable, or a localhost default; the time zone likewise; secrets are masked before configuration is shown.

## Install

```
pip install git+https://github.com/lottage/lodestone
```

From a checkout:

```
pip install -e .
```

## Tests

```
python tests/run_tests.py
```

The runner is offline by design: it refuses every connection that is not to localhost, and refuses to run network
programs such as `ssh`, so a test run can never touch a real machine.

## Licence

[The Unlicense](LICENSE): public domain, do what you like.
