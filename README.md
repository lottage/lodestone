# Lodestone

Lodestone is the model layer of a local-first AI stack. It looks after the language, vision, speech and embedding
models a household or a small site runs on its own hardware, and gives everything above it one steady place to ask
"which models do I have, where are they, and may I use that GPU right now?". Pure Python 3.10+ standard library: no
dependencies.

## Where it fits

Lodestone is layer 0 of a three-layer pyramid. Dependencies point down only: a layer never imports the one above it.

| Layer | Repository | Job |
| --- | --- | --- |
| 2 | [Hearth](https://github.com/lottage/hearth), [Grove](https://github.com/lottage/grove) | Apps: a Home Assistant house assistant; an optional agents add-on |
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
`Authorization: Bearer` checking. Without a key the gateway listens on loopback only: asking it to bind anywhere wider
without one is refused, and it will not start.

### What you need first

A clean machine needs these once, before the commands below.

**Python 3.10 or newer with `venv`, and git.** pip fetches the layers from GitHub, so even a pip install needs git. Modern
Linux (Ubuntu 24.04 among them) refuses `pip install` outside a virtual environment, so make one:

- Ubuntu / Debian: `sudo apt install git python3-venv python3-pip`, then
  `python3 -m venv ~/stonesage && . ~/stonesage/bin/activate`
- Windows (PowerShell): install Python 3.10+ and Git for Windows, then
  `py -m venv $HOME\stonesage; & $HOME\stonesage\Scripts\Activate.ps1`
  (if PowerShell refuses the script: `Set-ExecutionPolicy -Scope Process RemoteSigned`, then repeat)
- macOS: `brew install python git` (or the python.org installer), then
  `python3 -m venv ~/stonesage && . ~/stonesage/bin/activate`

**For the Docker install: Docker Engine, the Compose plugin and Buildx.** Docker Desktop (Windows, macOS) has all three. On
Ubuntu or Debian use Docker's own apt repository (`docker-ce docker-ce-cli containerd.io docker-buildx-plugin
docker-compose-plugin`), or Ubuntu's `docker.io` plus `docker-buildx` and `docker-compose-v2`: `docker.io` alone has no Buildx,
and the legacy builder it falls back to is deprecated and can fail. Compose 2.20 or newer (`docker compose version`). For an
NVIDIA GPU also the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/); for an AMD or Intel GPU,
Linux with `/dev/dri`.

## Install with Docker

Docker is the way in that works the same on Linux, Windows and macOS. Python 3.10+ writes the Compose files, Docker runs
them (with the tools above in place):

```
pip install "git+https://github.com/lottage/lodestone@v0.1.0-rc.7"
python -m lodestone.installer --docker          # looks at the machine, writes docker-compose.yml and .env here
# the installer wrote your API keys into .env; start the stack (the installer prints this line with the right folder)
docker compose up -d --build
```

**Images.** While a release has no published images (every pre-release, such as v0.1.0-rc.7) the installer writes `build:`
sections: Compose builds the layers' images itself from GitHub at that tag (no clone needed; the first start takes a few
minutes longer), then downloads the models. For a release with published images the file names them by version instead
(`ghcr.io/lottage/<layer>:0.1.0-rc.7`, never `latest`). `--build` / `--no-build` override the choice; the one command,
`docker compose up -d --build`, works either way.

First start, read this once: stock Ubuntu needs `docker-buildx` (Docker's own repository has it as `docker-buildx-plugin`)
before the build works, and the installer needs `python3-venv` and `python3-pip` (the step above). Leave `HASS_TOKEN=` empty
in `.env` if you do not have the token yet: the setup wizard in the cockpit asks for it. The first start downloads several GB
of base images, then the models. `models` and `stonesage-image` are one-shot containers, so `docker compose ps` shows them
as "Exited (0)" once they are done, by design; the other services should show "healthy" or "running".

(`--docker` also takes `--profile auto|<rung>` (below), `--gpu auto|vulkan|cuda|cpu`, `--out DIR`, `--vision`,
`--no-stonesage`, `--build` / `--no-build` and `--with-garden`, which adds [Grove](https://github.com/lottage/grove), an optional add-on
app, with its own walled networks and a firewall script that is shown and never applied without your yes.) What comes up:

- one engine container per role (coordinator, and for the bigger profiles a worker and an embedder), each running
  llama.cpp's own published server image (`ghcr.io/ggml-org/llama.cpp`, pinned to one build; change the tag in `.env`),
  with the same flags the bare-metal install uses: `--device` and `GGML_VK_VISIBLE_DEVICES` pin a card, `--flash-attn`
  always has a value, `--jinja` on the chat models, `--metrics` on;
- `models`: a one-shot container that downloads the profile's GGUF files from Hugging Face into a shared volume
  (resumable, SHA-256 recorded; `LODESTONE_VERIFY_MODELS=1` re-checks them), then exits. The engines wait for it and mount
  the volume read-only. The first start downloads 2 to 21 GB depending on the profile;
- `gateway`: the one address to use, an OpenAI-compatible API on `http://localhost:8000/v1` (bound to this computer
  only until you change `LODESTONE_BIND`), with the engines reached by service name;
- `stonesage`: the StoneSage cockpit with the Hearth app on `http://localhost:8888`; leave it out with `--no-stonesage`.

Engines are not published on any port, and no secret is written into the Compose file: `.env` (owner-only) holds the
`LODESTONE_API_KEY` (the gateway's bearer) and `STONESAGE_API_KEY` (the cockpit's login) the installer generated, plus an
empty `HASS_TOKEN=` for you to fill; `docker compose up` stops if a key is missing. Re-running the generator replaces the
Compose file (keeping a `.bak-<time>` copy) and never touches an existing `.env`, whose keys it reuses.

| Docker on | CPU | NVIDIA GPU | AMD or Intel GPU |
| --- | --- | --- | --- |
| Linux (x86-64 or arm64) | yes | yes, with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) | yes, through `/dev/dri` (Vulkan) |
| Windows (Docker Desktop, WSL2) | yes | yes, with a current NVIDIA driver | no |
| macOS (Docker Desktop, Intel or Apple Silicon) | yes | no | no |

The CPU profile is the default whenever no usable GPU is visible, so the same command works everywhere; the images are
multi-architecture, so Apple Silicon runs them natively. Give Docker Desktop at least 8 GB of memory for the CPU
profile. `--gpu auto` picks `cuda` for NVIDIA, `vulkan` for AMD or Intel on Linux, and `cpu` otherwise.

The Docker path has been run end to end on an NVIDIA GPU (CUDA, a Quadro M2000 with 4 GB, 2026-10-11) and on a CPU-only
stack (2026-10-12). The AMD / Intel path through `/dev/dri` (Vulkan), and most rungs of the GPU ladder
(`python -m lodestone.installer --list-profiles`), are tested offline only: the generated files are checked against the
rules the engines need, not run.

### GPU profiles

The installer names every card it finds, with its memory and which probe reported it (`nvidia-smi`, sysfs or Vulkan), and
picks the rung of this ladder that the largest card reaches: `found Quadro M2000, 4.0 GB -> gpu-4gb`. A card under 2 GB, an
integrated GPU, or none means the CPU profile, and the installer says which. `--profile gpu-4gb` (any rung) forces a rung,
also on a card the probes cannot size; `python -m lodestone.installer --list-profiles` prints this table with the reasoning
behind every row. The old names `gpu-large` and `gpu-small` are aliases of `gpu-12gb` and `gpu-8gb`. A second card takes the
vision engine and the embedder when they do not fit beside the main model.

| Profile | VRAM | Typical cards | Models | Speed |
| --- | --- | --- | --- | --- |
| `gpu-24gb` | 23 GB and up | RTX 3090 / 4090, RX 7900 XTX, RTX A5000 | coordinator: Qwen3-32B-Q4_K_M.gguf, 19.8 GB (GPU)<br>embedder: bge-large-en-v1.5-f16.gguf, 0.7 GB (GPU) | not measured |
| `gpu-16gb` | 15.5 to 23 GB | RTX 4060 Ti 16 GB, 4070 Ti Super, 4080, RTX A4000, RX 6800 / 6900 / 7800 XT | coordinator: Qwen3-14B-Q4_K_M.gguf + Qwen3-0.6B-Q8_0.gguf (draft), 9.6 GB (GPU)<br>vision: Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf + mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf, 6.0 GB (GPU)<br>embedder: bge-large-en-v1.5-f16.gguf, 0.7 GB (GPU) | ~41 tok/s for the 14B on an RX 6750 XT (same engine as gpu-12gb) |
| `gpu-12gb` | 11.5 to 15.5 GB | RTX 3060 12 GB, RTX 4070 / 4070 Ti, RX 6700 XT / 6750 XT, Arc B580 | coordinator: Qwen3-14B-Q4_K_M.gguf + Qwen3-0.6B-Q8_0.gguf (draft), 9.6 GB (GPU)<br>worker: qwen2.5-coder-3b-instruct-q5_k_m.gguf, 2.4 GB (GPU)<br>embedder: bge-large-en-v1.5-f16.gguf, 0.7 GB (GPU) | ~41 tok/s (RX 6750 XT, measured); Computer's tool eval 45/46 |
| `gpu-8gb` | 7.5 to 11.5 GB | RTX 3060 Ti / 3070 / 4060, RX 6600 / 7600, Arc A750 | coordinator: Qwen3-8B-Q4_K_M.gguf, 5.0 GB (GPU)<br>embedder: bge-large-en-v1.5-f16.gguf, 0.7 GB (GPU) | not measured |
| `gpu-6gb` | 5.5 to 7.5 GB | GTX 1060 6 GB, RTX 2060, RTX 3060 laptop 6 GB, RX 5600 XT | coordinator: granite-4.0-h-tiny-Q4_K_M.gguf, 4.2 GB (GPU)<br>embedder: bge-large-en-v1.5-f16.gguf, 0.7 GB (GPU) | 75 tok/s (RX 6600, measured) |
| `gpu-4gb` | 3.5 to 5.5 GB | GTX 1650, RTX 3050 4 GB laptop, Quadro M2000 / P1000 / T1000, RX 6400 / 6500 XT | coordinator: Qwen3-4B-Q4_K_M.gguf, 2.5 GB (GPU)<br>embedder: bge-large-en-v1.5-f16.gguf, 0.7 GB (GPU) | 13.1 tok/s through the gateway (Quadro M2000, measured 2026-10-11) |
| `gpu-2gb` | 2 to 3.5 GB | GT 1030, GTX 1050 2 GB, GTX 1060 3 GB, MX150 / MX250 laptop GPUs | coordinator: granite-4.0-h-tiny-Q4_K_M.gguf, 4.2 GB (12 layers on the GPU, the rest on the CPU)<br>embedder: bge-large-en-v1.5-f16.gguf, 0.7 GB (CPU) | not measured (partial offload: the CPU's speed, a little better) |
| `cpu` | no GPU, or under 2 GB | no GPU, integrated graphics, a card under 2 GB | coordinator: granite-4.0-h-tiny-Q4_K_M.gguf, 4.2 GB (CPU)<br>embedder: bge-large-en-v1.5-f16.gguf, 0.7 GB (CPU) | 20-40 tok/s on a modern desktop CPU (Granite: 35 on an i7-12700K, measured); a few on an old server CPU |


**What to expect on a CPU.** The CPU profile runs a small model (granite-4.0-h-tiny) on the processor. Roughly: a modern
desktop CPU decodes 20-40 tokens per second, which is fine for chat; an older server CPU manages only a few tokens per
second, so answers take tens of seconds and the first reply of a conversation (the model reads a long prompt) can take
minutes. Waits for the model are lengthened automatically on a CPU stack (`COORDINATOR_CPU=1`, set by the installer; or
`cluster.coordinator_cpu` and `courage.llm_timeout_s` in StoneSage's `config.json`). A GPU is strongly recommended, above
all for Computer's voice use, which needs answers in a few seconds. The installer prints this note whenever it picks the
CPU profile.

`docker/` in this repository holds a ready CPU example (`docker/docker-compose.yml`, `docker/.env.example`), and the
`Dockerfile` at the root builds the gateway image (`docker build -t ghcr.io/lottage/lodestone:0.1.0-rc.7 .`) that the Compose
file uses for the gateway and the downloader.

## Install on a fresh machine without Docker (bare metal, Ubuntu 24.04)

`python -m lodestone.installer` takes a fresh Ubuntu 24.04 box to running engines plus the gateway. Run it without
arguments first: it looks at the machine (OS, CPU, RAM, GPUs through `vulkaninfo`, sysfs and `nvidia-smi`), picks a
profile, and prints every step with the reason for it and the exact commands, changing nothing. When the plan looks
right:

```
sudo python -m lodestone.installer --apply
```

Options: `--profile auto|gpu-24gb|gpu-16gb|gpu-12gb|gpu-8gb|gpu-6gb|gpu-4gb|gpu-2gb|cpu` (`auto` picks the rung of the
ladder above that the largest GPU's VRAM reaches), `--list-profiles`, `--prefix /opt/lodestone`, `--config PATH` and
`--vision` (a vision engine, if a GPU has about 6.5 GB to spare). What `--apply` does, in order, stopping at the first
failure:

1. checks Linux, root, Ubuntu or Debian, and free disk space;
2. installs the build packages with apt (the Vulkan headers, drivers and shader compiler only for the GPU profiles);
3. creates a `lodestone` system user in the `render` and `video` groups;
4. clones llama.cpp at one pinned commit and builds `llama-server` with cmake (Vulkan for the GPU profiles, CPU
   otherwise);
5. downloads the profile's models from Hugging Face (atomic, resumable, SHA-256 recorded in `models/SHA256SUMS`):
   the ladder's models for the rung (from Qwen3-32B on a 24 GB card to Granite 4.0 h-tiny on a small card or the CPU, plus
   the bge-large-en-v1.5 f16 embedder; 20-40 tokens/s on a modern desktop CPU, a few on an older server CPU: the installer
   says so when it picks the CPU profile; a GPU is strongly recommended for voice use). The table is at the top of
   `installer.py`, easy to edit;
6. writes one systemd unit per engine and one for the gateway. Engines listen on 127.0.0.1, each GPU engine is pinned
   with `GGML_VK_VISIBLE_DEVICES` and waits at start until Vulkan sees every GPU (an engine that starts before the
   driver is ready silently runs on the CPU);
7. merges `cluster.*_url` and `gateway.*` into the config file: nothing is dropped, a key you already set is kept, and
   the old file is copied to `<file>.bak-<time>` first;
8. starts everything and waits for each `/health`.

Re-running is safe: a finished build, a downloaded model and an unchanged file are skipped. The gateway listens on the
machine only; set `gateway.api_key` before pointing it at your network (it refuses to start otherwise). Needs Linux with `apt`, root, and about 3 GB plus the models of free
disk space. The installer itself is standard library only. It has been tested offline against fixture output; a run
on real hardware is still to come.

## Install the Python package

```
pip install "git+https://github.com/lottage/lodestone@v0.1.0-rc.7"
pip install --upgrade "git+https://github.com/lottage/lodestone@v0.1.0-rc.7"   # an existing install moves to this release
```

From a checkout:

```
pip install -e .
```

## Security

The cockpit and its API need a login. There is one server key; nothing is served without it except the login page, the web
UI's static files and a bare `/api/health` (`ok` and a version).

- **Where the key is.** `STONESAGE_API_KEY` in the environment, else `api.key` in `config.json`. If neither exists when the
  server starts, it generates one, saves it into `config.json` (owner-only) and logs where, never the key. Read it with
  `python -m stonesage.auth show-key`; with Docker it is the `STONESAGE_API_KEY` line of the `.env` the installer wrote.
  Delete `api.key` and restart for a new one (every login ends).
- **Browsers** enter the key once on the login page and get an HttpOnly, SameSite=Strict session cookie (Secure over HTTPS).
  A cookie request that changes something must also come from the same origin. **Scripts** send `Authorization: Bearer <key>`
  (also for `/v1`, the Boost proxy and `/metrics`). Wrong keys are counted per address and locked out for a few minutes.
- **No cross-origin access.** There is no CORS: a web page on another site gets nothing, and the `Host` header must be an IP
  address, `localhost` or a name in `api.allowed_hosts` (defence against DNS rebinding).
- **Machines that cannot send a header** (Home Assistant's Ollama integration, a Prometheus server) can be listed by address in
  `api.trusted_hosts`; they then reach only the Ollama-style routes and `/metrics`, nothing else.
- **Dangerous features are off.** Four things stay behind a second switch in `config.json` (`features`), each also behind the
  login: `web_terminal` (the shell route and the PTY WebSocket on 127.0.0.1:8088), `code_tools` (tools whose code the server
  runs), `file_editor` (read and write inside the configured workspace folders only) and `raw_ha_call` (a raw Home Assistant
  service call, in Hearth). A route whose feature is off answers 404.
- **The settings API cannot grant itself powers.** `api`, `features`, `apps` and `workspace_roots` are not writable through it,
  and changing the address of something that has a stored token or key (Home Assistant, a Boost provider, Proxmox) needs that
  secret entered again in the same request. `GET /api/config` masks every key, token, password and cookie.
- **Approvals belong to the person who was asked.** A pending "shall I unlock the door?" is stored under the caller plus its own
  conversation id; ids that look like the server's own (`ha:`, `alexa:`, `api:`...) are ignored.

An upgrade from a version without a login: start it once (the key is generated), open the cockpit, enter the key, and add
`api.trusted_hosts` / `api.allowed_hosts` for the machines and names that need them. See docs/LAYERS.md in the Hearth repository
if you use the Home app; the protections are described there.

## Tests

```
python tests/run_tests.py
```

The runner is offline by design: it refuses every connection that is not to localhost, and refuses to run network
programs such as `ssh`, so a test run can never touch a real machine.

## Licence

[The Unlicense](LICENSE): public domain, do what you like.
