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

## Gateway

`python -m lodestone.gateway` puts one OpenAI-compatible address in front of your engines, so an app needs one base URL
and one key instead of one port per model. It serves `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/audio/transcriptions` and `/v1/audio/speech`, plus `/v1/models` and `/health`. A request is routed by its `model`
(a role alias such as `coordinator`, `worker` or `vision`, or a model id), and streamed replies are relayed chunk by
chunk. Configuration lives under `gateway` in `config.json`: `gateway.host` (default `127.0.0.1`), `gateway.port`
(default `8000`) and `gateway.api_key`, or the `LODESTONE_API_KEY` environment variable, which turns on
`Authorization: Bearer` checking. Binding beyond loopback without a key logs a warning.

## Install with Docker

Docker is the way in that works the same on Linux, Windows and macOS. Python 3.10+ writes the Compose files, so nothing
else has to be installed besides Docker itself:

```
pip install git+https://github.com/lottage/lodestone
python -m lodestone.installer --docker          # looks at the machine, writes docker-compose.yml and .env here
# fill LODESTONE_API_KEY in .env (the gateway), then
docker compose up -d
```

(`--docker` also takes `--profile auto|gpu-large|gpu-small|cpu`, `--gpu auto|vulkan|cuda|cpu`, `--out DIR`, `--vision` and
`--no-stonesage`.) What comes up:

- one engine container per role (coordinator, and for the bigger profiles a worker and an embedder), each running
  llama.cpp's own published server image (`ghcr.io/ggml-org/llama.cpp`, pinned to one build; change the tag in `.env`),
  with the same flags the bare-metal install uses: `--device` and `GGML_VK_VISIBLE_DEVICES` pin a card, `--flash-attn`
  always has a value, `--jinja` on the chat models, `--metrics` on;
- `models`: a one-shot container that downloads the profile's GGUF files from Hugging Face into a shared volume
  (resumable, SHA-256 recorded; `LODESTONE_VERIFY_MODELS=1` re-checks them), then exits. The engines wait for it and mount
  the volume read-only. The first start downloads 5 to 12 GB depending on the profile;
- `gateway`: the one address to use, an OpenAI-compatible API on `http://localhost:8000/v1` (bound to this computer
  only until you change `LODESTONE_BIND`), with the engines reached by service name;
- `stonesage`: the StoneSage cockpit with the Hearth app on `http://localhost:8888`; leave it out with `--no-stonesage`.

Engines are not published on any port, and no secret is written into the generated files: `.env` holds the empty
`LODESTONE_API_KEY=` and `HASS_TOKEN=` for you to fill. Re-running the generator replaces the Compose file (keeping a
`.bak-<time>` copy) and never touches an existing `.env`.

| Docker on | CPU | NVIDIA GPU | AMD or Intel GPU |
| --- | --- | --- | --- |
| Linux (x86-64 or arm64) | yes | yes, with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) | yes, through `/dev/dri` (Vulkan) |
| Windows (Docker Desktop, WSL2) | yes | yes, with a current NVIDIA driver | no |
| macOS (Docker Desktop, Intel or Apple Silicon) | yes | no | no |

The CPU profile is the default whenever no usable GPU is visible, so the same command works everywhere; the images are
multi-architecture, so Apple Silicon runs them natively. Give Docker Desktop at least 8 GB of memory for the CPU
profile. `--gpu auto` picks `cuda` for NVIDIA, `vulkan` for AMD or Intel on Linux, and `cpu` otherwise. This path is
tested offline (the generated files are checked against the rules above for every profile and GPU kind); it has not yet been run on real GPUs.

**What to expect on a CPU.** The CPU profile runs a small model (granite-4.0-h-tiny) on the processor. Roughly: a modern
desktop CPU decodes 20-40 tokens per second, which is fine for chat; an older server CPU manages only a few tokens per
second, so answers take tens of seconds and the first reply of a conversation (the model reads a long prompt) can take
minutes. Waits for the model are lengthened automatically on a CPU stack (`COORDINATOR_CPU=1`, set by the installer; or
`cluster.coordinator_cpu` and `courage.llm_timeout_s` in StoneSage's `config.json`). A GPU is strongly recommended, above
all for Computer's voice use, which needs answers in a few seconds. The installer prints this note whenever it picks the
CPU profile.

`docker/` in this repository holds a ready CPU example (`docker/docker-compose.yml`, `docker/.env.example`), and the
`Dockerfile` at the root builds the gateway image (`docker build -t ghcr.io/lottage/lodestone:latest .`) that the Compose
file uses for the gateway and the downloader.

## Install on a fresh machine without Docker (bare metal, Ubuntu 24.04)

`python -m lodestone.installer` takes a fresh Ubuntu 24.04 box to running engines plus the gateway. Run it without
arguments first: it looks at the machine (OS, CPU, RAM, GPUs through `vulkaninfo`, sysfs and `nvidia-smi`), picks a
profile, and prints every step with the reason for it and the exact commands, changing nothing. When the plan looks
right:

```
sudo python -m lodestone.installer --apply
```

Options: `--profile auto|gpu-large|gpu-small|cpu` (`auto` chooses by the VRAM of the largest GPU: 11.5 GB or more,
7.5 GB or more, else CPU), `--prefix /opt/lodestone`, `--config PATH` and `--vision` (a vision engine, if a GPU has
about 6.5 GB to spare). What `--apply` does, in order, stopping at the first failure:

1. checks Linux, root, Ubuntu or Debian, and free disk space;
2. installs the build packages with apt (the Vulkan headers, drivers and shader compiler only for the GPU profiles);
3. creates a `lodestone` system user in the `render` and `video` groups;
4. clones llama.cpp at one pinned commit and builds `llama-server` with cmake (Vulkan for the GPU profiles, CPU
   otherwise);
5. downloads the profile's models from Hugging Face (atomic, resumable, SHA-256 recorded in `models/SHA256SUMS`):
   `gpu-large` Qwen3-14B Q4_K_M, Qwen2.5-Coder-3B Q5_K_M and bge-large-en-v1.5 f16; `gpu-small` Qwen3-8B Q4_K_M and
   bge; `cpu` granite-4.0-h-tiny Q4_K_M and bge (20-40 tokens/s on a modern desktop CPU, a few on an older server CPU:
   the installer says so when it picks this profile; a GPU is strongly recommended for voice use). The table is at the
   top of `installer.py`, easy to edit;
6. writes one systemd unit per engine and one for the gateway. Engines listen on 127.0.0.1, each GPU engine is pinned
   with `GGML_VK_VISIBLE_DEVICES` and waits at start until Vulkan sees every GPU (an engine that starts before the
   driver is ready silently runs on the CPU);
7. merges `cluster.*_url` and `gateway.*` into the config file: nothing is dropped, a key you already set is kept, and
   the old file is copied to `<file>.bak-<time>` first;
8. starts everything and waits for each `/health`.

Re-running is safe: a finished build, a downloaded model and an unchanged file are skipped. Set `gateway.api_key`
before exposing the gateway beyond the machine. Needs Linux with `apt`, root, and about 3 GB plus the models of free
disk space. The installer itself is standard library only. It has been tested offline against fixture output; a run
on real hardware is still to come.

## Install the Python package

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
