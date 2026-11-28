# vastai-llmrunner

Boots a llama.cpp stack on a Vast.ai (or any CUDA) container: `llama-swap` for
hot-swappable chat models, a small always-on embedding model, a config UI, and a
Cloudflare tunnel — all behind one URL, which is what SillyTavern wants.

## Endpoints (port 5000 / your tunnel URL)

| Path | Served by | Notes |
| --- | --- | --- |
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | guard → llama-swap | chat model, swapped on demand |
| `/v1/chat/completions` with `model: media-dossier` | guard → RTR → llama-swap | media pass; see [Media router (RTR)](#media-router-rtr) |
| `/v1/embeddings` | embed.py sidecar (:8090) | OpenAI-compatible embeddings |
| `/embedding` | embed.py sidecar (:8090) | llama.cpp's native embedding endpoint |
| `/editor` | cfgedit | config UI, logs, terminal |
| `/ui`, `/logs` | llama-swap | llama-swap's own UI |

The embedding model runs in its own `llama-server` process **outside**
llama-swap, so it is never swapped out: text generation and embeddings work at
the same time, from the same base URL.

## Using it from SillyTavern

**Text generation** — Chat Completion → Custom (OpenAI-compatible):

- Custom Endpoint (Base URL): `https://<your-tunnel>/v1`
- API key: anything non-empty
- Pick the model from the list. The number of parallel slots is chosen per
  model (see [Slots](#slots)).

**Vector storage (embeddings)** — Extensions → Vector Storage:

- Source **vLLM** (or any OpenAI-compatible source): URL `https://<your-tunnel>`,
  model name anything (the sidecar serves whichever model it loaded).
- or Source **llama.cpp**: URL `https://<your-tunnel>` — this hits `/embedding`.

Re-vectorize your chats after changing the embedding model: different models
produce different vector spaces and dimensions.

## Model IDs in the editor

Each row in the editor's **Models** table becomes one entry in `config.yaml`
named `<name>` (plus `<name>-nomtp` and `<name>-text` when it has an mmproj),
where `<name>` comes from the GGUF file name in the model URL, minus the
`.gguf` extension and any `?download=true` style query string. Ids used to
carry a slot count (`<name>-p1` … `-p8`); a default model saved under one of
those is pointed at the new id on the next config regen, but clients that
saved an old id need to pick the model again, and a `DEFAULT_MODEL` container
env var needs the `-pN` dropped.

Two different models can share a file name (the same quant published by two
repos, for example). Those names are made unique: the first model to claim a
name keeps it, and a later one is prefixed with its repo owner
(`unsloth-Model-Q4_K_M`), falling back to a counter. Adding a second row
with an identical model URL is dropped on save, since it would only ever
produce the same entry.

After **Save & Regenerate**, the **Default model** dropdown refreshes in
place — no page reload needed.

## Exposing only the active model

`/editor` has an **Expose only active model to /v1/models** checkbox. With it on,
`/v1/models` lists just the model that is actually loaded (or, if nothing is
loaded, the configured default model) instead of every entry in
`config.yaml`. Clients like SillyTavern then can't pick a model that
would trigger a swap, and their model dropdown stays readable.

It is on by default; the container env var `EXPOSE_ACTIVE_ONLY=0` turns it off
at boot, and `/app/expose_active_only` (written by the UI, `1` or `0`) wins
when present.
The filter never hides everything: if neither a running nor a default model
matches the listing, the full list is served as before.

## Downloads

While a model is being fetched, `/editor` shows a progress bar per file —
percent, MB downloaded, speed and ETA — above the status bar, and the same bar
appears for the embedding model in the **Embeddings** section. A file whose size
the server won't report gets an indeterminate (striped) bar and a running MB
count instead. Retries and failures are shown on the bar itself.

Two knobs control download throughput:

| Setting | Default | Notes |
| --- | --- | --- |
| `DOWNLOAD_CONNECTIONS` | `16` | connections aria2c opens per file (`-x`/`-s`), 1-16 |
| `DOWNLOAD_PARALLEL` | `1` | how many files download at once, 1-8 |

A model with an mmproj and a draft model is three separate files, so
`DOWNLOAD_PARALLEL=3` fetches them together instead of one after another. Disk
eviction accounts for every download in flight, so raising it can't overfill
`/models`. `DOWNLOAD_CONNECTIONS` is also used by the embeddings sidecar.

Both are in `/editor` twice: the toolbar selects (**Connections/file**,
**Parallel downloads**) write `/app/download_connections` and
`/app/download_parallel` and apply to the next download with no config regen,
while the **Settings** entries of the same name are written into `config.yaml`
by **Save & Regenerate**. As with `DOWNLOADER`, the `/app/...` file wins over the
env var when both are set.

### Download integrity

aria2c writes to `<file>.part` and the file is renamed into place only after
aria2c exits cleanly and the size matches the server's (`X-Linked-Size` from
HuggingFace, else the final `Content-Length`; an unknown size skips only the
size check). A model is reused as cached only if it exists, has no `.aria2`
control file and matches that size, so a truncated file from an interrupted run
is deleted and fetched again. On startup, any model with an `.aria2` sibling is
removed. A per-file lock (`<file>.lock`) makes a second `serve.py` wait for the
first and then find the file cached. The editor refuses to unload models or
save/regenerate the config (HTTP 409) while a download or its sha256 check is
running, since that would kill it.

### sha256 verification

Every model file the box downloads (chat model, mmproj, draft, image/video
model files, and the embeddings model) is hashed with sha256 after the download
completes and compared with an expected hash (`dlverify.py`):

- **HuggingFace URLs** (`https://huggingface.co/<repo>/resolve/<rev>/<file>`):
  the file's LFS sha256 from the Hub's `paths-info` API for that
  repo/revision/file. Each successful lookup is cached in
  `/models/.sha256-cache.json`, and the cached value is used when a later
  lookup fails (Hub down, no network).
- **Other URLs**: the optional `sha256` field of the `models.json` entry
  (64 hex characters; it is the hash of the entry's `url` file and is ignored
  for HuggingFace URLs).

While hashing, the status shows **verifying** with a percentage. On a
mismatch the file is deleted and downloaded once more; a second mismatch stops
with an error that names the file, the hash it got and the hash it expected.

**"Unverified"** means there was nothing to compare against: a non-HF URL
without a `sha256` field, a file the Hub lists without a sha256 (a small
non-LFS file), or a failed Hub lookup with no cached hash. The file is used
anyway, the log prints `WARNING: <file> is UNVERIFIED: <reason>`, and the
status file carries `"unverified": [<files>]`, which the editor shows as
"⚠ UNVERIFIED" next to the status. A verified file is hashed once, right
after its download; a cached verified file is not re-hashed on later boots. A
`<file>.verify` marker exists from the start of a download until its check
passes, so a run killed mid-check checks the file again next time instead of
trusting it. An unverified file keeps its marker, so every boot asks the Hub
again (hashing the file once it answers) and repeats the UNVERIFIED warning
until the check passes.

Boot cost: hashing reads the whole file once, so a fresh download takes longer
to become ready by one full read of each file (for a 32 GB model that read is
not small; the actual time has not been measured). A cached, verified file
costs nothing on later boots. An unverified file costs one Hub request per
boot, plus a full hash on the boot where the Hub first answers.

The aria2c transfer limits default to aria2's own values and can be tightened
with env vars when a host's link stalls:

| Setting | Default | Notes |
| --- | --- | --- |
| `DOWNLOAD_CONNECT_TIMEOUT` | `60` | `--connect-timeout`, seconds |
| `DOWNLOAD_TIMEOUT` | `60` | `--timeout`, seconds |
| `DOWNLOAD_LOWEST_SPEED` | `0` | `--lowest-speed-limit`, bytes/s; `0` is off |

aria2c also runs with `--allow-overwrite=true --auto-file-renaming=false`
(safe because the `.part` name is private; it still resumes from its control
file). The `hf` downloader is unchanged; `huggingface_hub` honours
`HF_HUB_DOWNLOAD_TIMEOUT` (default 10 s) itself. `HF_BACKEND=hf_transfer` is
deprecated upstream in favour of `hf_xet`, the default.

## Context length

The status bar in `/editor` leads with the loaded model's context per slot out
of the trained maximum (e.g. `context per slot: 255,744 / 262,144 trained max
(98%)`, amber when past it), followed by slot count and total context, with a
**copy context length** button that copies the per-slot number, which is the
value to put in a client's context size setting.

## VRAM detection

The context budget starts from the GPUs' total memory. `serve.py` asks
`nvidia-smi` first; some hosts ship a broken one (a wrapper whose
`nvidia-smi.real` is missing), so it falls back to NVML (`libnvidia-ml`, part of
the driver) and then to `llama-server --list-devices`. The log line
`[serve] detected N GPU(s) via <source>` says which one answered.

If none of them work, the budget drops to the 512-token floor and the context
line says `no GPU memory detected`. Set `VRAM_MB` in the container env — one
number for the total (`131072`) or one per GPU (`65536,65536`) — to skip
detection, or `CTX_SIZE` to fix the total context outright.

## Slots

`serve.py` sizes the total context from the VRAM left after the weights, then
splits it into parallel slots. By default (`PARALLEL=auto`) it uses as many
slots as can each hold the model's full trained context — e.g. ~600k tokens of
budget on a 262k model gives 2 slots × 262,144 — and 1 slot when even one
full-length slot doesn't fit. At most 8 slots (`MAX_SLOTS` env var), and never
under 2048 tokens each.

The **slots for <model>** slider under the status bar overrides that per
model. It shows, as you drag it, what each slot count gives:

- left, toward **◀ slots**: more slots, each with less context (any mode);
- right, toward **context ▶**: fewer slots, each past the trained length, extended
  with YaRN RoPE scaling — only with the `yarn`
  [Past trained context](#going-past-the-trained-context) mode. Otherwise the
  slider stops at the auto count.

Outside `yarn`, budget beyond the auto slots' trained length goes unused. With
`yarn` every slot count uses the whole budget, so at auto the remainder is
spread across the slots and extended too (1 slot × 511,488 at 1.95× yarn on a
262k model with ~511k of budget).

**Apply** saves the count in `/app/slots/<model id>` and reloads the model if
it is the one running; **Auto** removes it. The options come from the budget
the model recorded on its last load (`/app/ctx_budget/<model id>.json`), so a
model has to load once before the slider appears. A count the budget no longer
allows (e.g. after a KV cache quant change) is clamped into range on load.

### Going past the trained context

The context is sized from whatever VRAM is left after the weights, so a slot can
end up with more context than the model was actually trained for — which makes
llama-server refuse to load, or produce gibberish past the trained length. The
**Past trained context** dropdown in `/editor` picks what happens when the
per-slot context (budget / slots) exceeds the GGUF's `context_length`:

| mode | behaviour |
| --- | --- |
| `clamp` (default) | ceiling the context at the trained length — safest, no quality loss; any budget past the auto slots' trained length goes unused |
| `yarn` | keep the larger context and extend it with YaRN RoPE scaling (`--rope-scaling yarn --rope-scale <factor> --yarn-orig-ctx <trained>`) |
| `linear` | treated as `clamp` — see below |
| `none` | treated as `clamp` — see below |

llama-server caps every slot at the model's trained context
(`n_ctx_slot()` in `tools/server/server-context.cpp`) and only raises that
cap when YaRN scaling is set (`llama-context.cpp` re-derives `n_ctx_train`
from `--yarn-orig-ctx / rope-freq-scale`). Linear or no scaling would only
spend VRAM on context llama-server never lets a slot use, so `serve.py` plans
those modes like `clamp`, and the context line notes it.

`clamp` also applies to an explicit `CTX_SIZE`, so the ceiling is never
exceeded. When the GGUF does not declare a `context_length`, nothing is changed.

The dropdown writes `/app/ctx_overflow` and unloads the model, so the next load
uses it; the container env var `CTX_OVERFLOW` sets the boot default, and the
per-model `CTX_OVERFLOW` field in **Settings** covers models that need a
different policy. The override file wins over both. The active decision is shown
in the `/editor` context line (e.g. `— yarn rope scaling x3.00 past the trained
32768`) and in the server command under **Active server command**.

## Speculative decoding / MTP loads

llama.cpp's automatic memory fitting (`--fit`, on by default) probes context
creation before the real load, and some architectures reject that probe: a
Gemma-4 MTP assistant aborts the whole load with

```
llama_init_from_model: failed to initialize the context: Gemma4Assistant requires ctx_other to be set
common_init_: failed to create context with model ...
```

`serve.py` already picks the context size, batch, ubatch and GPU split itself,
so fitting has nothing left to adjust. `FIT` controls it:

| value | behaviour |
| --- | --- |
| `auto` (default) | pass `--fit off` when MTP speculation is on, otherwise leave llama-server's default alone |
| `on` / `off` | always pass that value |

The flag is only passed when the `llama-server` binary advertises `--fit` in its
help, so an older build is never handed an argument it would reject. Set it per
model in **Settings** or with the `FIT` container env var.

Note that llama.cpp expects the MTP assistant GGUF as the *draft* model next to
the base model — `--model <base>` plus `--spec-draft-model <mtp-assistant>`.
Putting the same file in both the **Model URL** and **Draft/MTP URL** fields
logs a warning.

## llama-swap version

`init.sh` installs a pinned llama-swap release, `LLAMA_SWAP_TAG` (default
`v277`), because the generated config depends on llama-swap's routing keys.
Set `LLAMA_SWAP_TAG=vNNN` in the container env to use another release, or
`LLAMA_SWAP_TAG=latest` to take the newest one at boot. The installed version
is logged at boot (`llama-swap version: ...`). A binary already on `PATH` is
kept as is.

## Media router (RTR)

`rtr.py` is a small router that `init.sh` starts on `127.0.0.1:$RTR_PORT`
(default `8082`, loopback only; Caddy is unchanged). guard forwards every chat
request whose `model` is `media-dossier` to it. RTR runs a media model against
llama-swap and answers with one non-streaming chat completion: the message
content (and a top-level `rtr` field) is the result, holding the dossier, a
`status` (`ok`, `partial` or `failed`), per-step `timings`, the media
projector's sha256 and the llama.cpp build string (both informational).

The request carries an `rtr` object: `media_model` (a llama-swap model id,
required), and optionally `rp_model`, `kind` (`dossier` or `delta`),
`dry_run` (return only the plan: coexist or swap, and what would be evicted),
`flush` (queued follow-up questions answered while the media model is
loaded) and `rp_prefill` (`{"messages": [...]}`, the role-play chat to re-read
right after the role-play model is reloaded, so that re-read is timed and the
cache is warm before guard lets chat through). Without a `response_format`,
RTR asks for the dossier JSON schema; a reply that does not validate is
repaired once, then returned as `partial`. Each flushed item gets the same
repair pass against the delta schema (`answers`, `timeline_add`, `uncertain`).

Besides the dossier, the `rtr` result (and a `dry_run` plan) carries these
fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `rp_vision` | bool | the role-play model's llama-swap entry has a non-empty `MMPROJ_URL`, so `serve.py` starts it with `--mmproj`; `false` when there is no RP or the id is not in the config |
| `co_set` | string or null | the llama-swap matrix set that holds both the RP and the media model, whether or not this pass used it |
| `gpu` | object or null | what the planner used: `{"name", "total_mb", "free_mb", "need_mb", "margin_mb"}`; `name` and `total_mb` come from one `nvidia-smi` query at RTR start (names joined with ` + `, totals summed), `free_mb` is the free VRAM read for this plan; any of these is `null` when unknown, and the whole field is `null` when nothing was read |
| `timings.prefill_rp_s` | number or null | seconds to re-read `rp_prefill.messages` after the RP reload; `null` in coexist mode, when the RP was kept, or without `rp_prefill` |
| `rp_prefill_error` | string | only present when that re-read failed |
| `flushed[].repaired` | bool | the flushed answer went through the repair pass |

A running media model counts as the role-play model. If the RP is the media
model itself (`rp_model` equals `media_model`), RTR loads it once and keeps it
loaded (`rp_reload` is `{"ok": true, "attempts": 0, "kept": true}`); if it is
already running beside nothing else, the pass runs without a swap.

RTR keeps the role-play model loaded and runs the media model beside it only
when llama-swap's config has a coexist set for the pair and `nvidia-smi`
shows enough free VRAM (the media model's need plus 1 GiB). Otherwise it
**swaps**: it unloads the role-play model, loads the media model, runs the
pass, unloads the media model and reloads the role-play model with a
one-token request. The reload runs even when a step fails.

During a swap guard holds other generation requests until the swap ends, for
at most `SWAP_HOLD_S` seconds (default `600`). After that the request gets
`503` with `{"error":{"message":"media swap in progress",...}}` and a
`Retry-After` header. If RTR is down, guard does not hold anything, and a
`media-dossier` request gets `503` with
`{"error":{"message":"media router unavailable","type":"service_unavailable"}}`.
RTR logs to `/tmp/rtr.log`; a failure to start is logged but does not stop the
boot. The boot watchdog restarts RTR whenever it exits, with no limit, logging
each restart. The wait before a restart doubles from 5 s (5, 10, 20, ...) up
to a maximum of 300 s, and goes back to 5 s once RTR has stayed up for 120 s.

For a browser client on another origin, guard answers CORS preflights
(`OPTIONS` with `Access-Control-Request-Method`, `204`) on the generation paths
and adds `Access-Control-Allow-Origin` to `media-dossier` responses. By default
any origin is allowed (the request's origin is echoed; no credentials). This is
open on purpose for now and is to be secured later (see
[docs/known-issues.md](docs/known-issues.md)). Setting `RTR_CORS_ORIGINS`
(exact origins, comma-separated, or `*`) restricts it to that list; loopback
origins (`localhost`, `127.0.0.1`, `[::1]`, `*.localhost`, over `http`,
`https` or `tauri`) stay allowed then. This adds headers on the existing path only; no port
is opened. Real browser behaviour is untested.

After a swap the role-play model starts with an empty cache, so its next reply
re-reads the whole chat, unless the request sent `rp_prefill`. RTR has been tested only against the fake llama-swap
in `tests/`; see [docs/known-issues.md](docs/known-issues.md) for this and the
other open limits.

| Env | Default | Meaning |
| --- | --- | --- |
| `RTR_PORT` | `8082` | RTR's loopback port (guard and RTR both read it) |
| `SWAP_HOLD_S` | `600` | how long guard holds a request during a swap |
| `RTR_TIMEOUT_S` | `1800` | guard's timeout for a media request |
| `RTR_LOAD_TIMEOUT_S` | `900` | RTR's timeout for each load or unload |
| `RTR_GEN_TIMEOUT_S` | `600` | RTR's timeout for each media generation |
| `RTR_CORS_ORIGINS` | *(any origin)* | when set, only these origins (plus loopback) may call `media-dossier` from a browser; comma-separated, or `*` |

## Where the model list comes from

The editor header says which source is in effect — `models from: saved params
(N models)` or `models from: container env defaults`.

`/app/params.json`, written by **Save & Regenerate**, wins over the container's
`MODEL_URL`, `MODEL_URL_2`, ... env vars. The env vars are used only when no
usable params file exists (first boot, after **Reset to env defaults**, or if
the file is unreadable). A saved file with an empty model list is honoured as
written — it is not treated as "unset", which would silently bring the env
models back over what was just saved. Saving an empty table asks for
confirmation first.

## Repo model registry (`models.json`)

`models.json` at the repo root lists known models. `init.sh` fetches it with the
scripts on every boot (and **Update Scripts** refreshes it). It is used only when
the container env var `MODELS=id1,id2` is set; ids are the GGUF file stem, or a
name from `presets`. `DEFAULT_MODEL` works as before.

**Precedence for the model list.** `MODELS` set and registry valid: `models.json`
> `/app/params.json` > `MODEL_URL*` env. `MODELS` unset: unchanged (params.json >
env). Registry models are *not* re-masked by params.json: their per-model knobs
(`cache_type[_k/_v]`, `ctx_size`, `gpu_layers`, `fit`, `ctx_overflow`, `no_mtp`,
merged over the top-level `defaults`) override the global **Settings** for that
model; other settings still come from params.json/env. While `MODELS` is active
the editor shows the registry as authoritative; edits to the model rows are
ignored on save (Settings and Embeddings still save).

Entry fields: `url` (required), `mmproj`, `draft`, the knobs above, `note`,
`sha256` (expected hash of a non-HuggingFace `url`; see
[sha256 verification](#sha256-verification)).
`defaults` also accepts `image_min_tokens`, `image_max_tokens`,
`mtmd_batch_max_tokens`, `mlock`. Only https URLs; unknown keys and any
token/secret-like field (the container env keeps `HF_TOKEN`) are rejected.

**Failure behaviour.** The registry is validated at boot. If it is missing or
invalid, cfginit logs the problems loudly, then uses the last good copy
(`/app/models.json.good`, written after each successful validation), then
params.json/env. It never exits, so llama-swap still starts.

**Pinning.** `REPO_REF` (default `main`) selects the branch, tag or commit the
scripts and `models.json` are fetched from, e.g. `REPO_REF=v1.2`. `init.sh`
itself must be fetched by the template's onstart; add `models.json` to that
fetch list too, or the first boot after enabling `MODELS` relies on the
`init.sh` fetch.

## Image and video generation (sd.cpp)

Optional. llama-swap can also run [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
`sd-server`, so SillyTavern can generate images from the same tunnel. Set the
container env var `IMAGE_MODELS=z-image-turbo` (ids or preset names from the
`images` section of `models.json`, comma separated; `all-images` is a preset).
Unset or empty means disabled: nothing is installed, downloaded or written to
`config.yaml`. This is independent of `MODELS`.

On boot `init.sh` installs the `sd-server` binaries to `/opt/sd`, `cfginit.py`
adds one llama-swap entry per selected model, and the first request for that
model downloads its files with the same downloader as chat models (`.part`
files, size check, `/models`, `HF_TOKEN`, `DOWNLOAD_*` settings) through
`imgserve.py`. Any failure in the install or in the registry is logged loudly
and image generation stays off; chat and llama-swap still start.

**Registry fields** (`images.models.<id>`, ids are free-form `[A-Za-z0-9._-]`;
`images.defaults` and `images.presets` work like the chat registry). Only https
URLs; unknown keys and secret-like fields are rejected.

| Field | Notes |
| --- | --- |
| `kind` | `image` (needs `diffusion_model`, `llm`, `vae`) or `video` (needs `diffusion_model`, `t5xxl`, `vae`) |
| `diffusion_model`, `llm`, `t5xxl`, `vae` | URLs; passed as `--diffusion-model`, `--llm`, `--t5xxl`, `--vae` |
| `cfg_scale`, `steps`, `width`, `height` | server defaults (`--cfg-scale`, `--steps`, `-W`, `-H`); a request can override them |
| `sampling`, `flow_shift`, `video_frames` | `--sampling-method`, `--flow-shift`, `--video-frames` |
| `offload_to_cpu` | `--offload-to-cpu`, trades speed for VRAM |
| `note` | free text: licence, sizes |

Shipped entries: `z-image-turbo` (Z-Image-Turbo Q8_0 + Qwen3-4B encoder +
FLUX VAE, about 9.4 GB of files) and `wan22-ti2v-5b` (Wan2.2 TI2V-5B Q8_0 +
umt5-xxl encoder + Wan2.2 VAE, 12.85 GB of files, video). Nobody has measured
their VRAM use on this setup; the sd.cpp docs only claim Z-Image runs in 4 GB
with offloading.

**Swap only.** Image models are ordinary llama-swap models in the same default
group as the chat models, so only one of them is loaded at a time. An image
request unloads the LLM (and the next chat request reloads it, which costs the
full model load), and the reverse. `serve.py` is not told to leave room for the
image model. A video job runs inside `sd-server` memory: if any chat request
swaps the model out before the job finishes, the job is lost, so keep other
traffic off the endpoint until it completes. Image models have `ttl: 0`, so
they are never unloaded by idleness, only by another model's request.

**GPU architecture.** The sd.cpp CUDA build has native kernels for sm_86
(RTX 30xx, A40) and sm_89 (RTX 40xx, L40) only. sm_75, sm_80 (A100) and sm_90
(H100) run by JIT-compiling PTX, which makes the first start slow (the cache
lives in `~/.nv/ComputeCache`; `imgserve.py` raises its size limit). RTX 5090
(sm_120) can only JIT from sm_90 PTX; that is unverified on real hardware.
Nothing here has been run on a GPU yet.

### SillyTavern

Extensions → Image Generation → Source **Stable Diffusion cpp (sdcpp)**, URL
`https://<your-tunnel>`, then pick the image model id in the model dropdown.
SillyTavern's server fetches that URL, so it only has to reach the tunnel.
Things to know:

- The dropdown lists llama-swap's `/v1/models`, which by default shows only the
  active model (see [Exposing only the active model](#exposing-only-the-active-model)).
  Turn that off (`EXPOSE_ACTIVE_ONLY=0`, or the checkbox in `/editor`) so the
  image id appears; it then also lists every LLM, so pick the image entry.
- SillyTavern sends no API key and no LoRA field. The ping only checks that
  llama-swap answers; it does not prove `sd-server` is up.
- Video is not available in SillyTavern; use the API below.

### Video API

llama-swap does not route `/sdcpp/v1/*` by model, so go through its
`/upstream/<id>/` prefix, which loads the model and forwards the call. A job is
asynchronous: `vid_gen` returns 202 with an id, then poll until it completes.

```sh
curl -s https://<your-tunnel>/upstream/wan22-ti2v-5b/sdcpp/v1/vid_gen \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a lovely cat","width":832,"height":480,"video_frames":33,"fps":16,"output_format":"webm"}'
# -> {"id":"job_...","status":"queued","poll_url":"/sdcpp/v1/jobs/job_..."}

curl -s https://<your-tunnel>/upstream/wan22-ti2v-5b/sdcpp/v1/jobs/job_...
# states: queued, generating, completed, failed, cancelled
```

The finished job carries the WebM as base64 in its `result`. Polling the job
through `/upstream/<id>/` reloads the model if it was swapped out meanwhile, and
a swapped-out job is gone, so keep chat traffic away while it runs. If a client
cannot use `/upstream/`, the fixed alternative is to give `sd-server` its own
tunnel route to its port; that is not set up here.

### Binary source and fallback

sd.cpp publishes no Linux CUDA release, so `init.sh` extracts the binaries
layer of the official `ghcr.io/leejet/stable-diffusion.cpp:master-cuda` image
(pinned by layer digest, git rev 52491bd, verified with sha256, about 350 MB
download and 1.3 GB on disk; needs `libgomp1` and `libnccl2`, installed with
apt if missing). The layer is content-addressed but could disappear from ghcr
one day. The fallback is to build `sd-server` once with CUDA and
`-DCMAKE_CUDA_ARCHITECTURES` for your GPUs, publish the `bin/` directory as a
`.tar.gz` with an `sd.cpp/bin/` prefix (for example as a release asset of this
repo), and set `SD_LAYER_URL` to its URL and `SD_LAYER_DIGEST=sha256:<hash of
that file>`. This repo does not ship such a build.

### Licences

Z-Image-Turbo, Wan2.2 TI2V-5B and Qwen3-4B-Instruct-2507 are apache-2.0, and so
are the GGUF conversions used (HF metadata, 2026-11-25). The FLUX `ae` VAE is
fetched from the Comfy-Org mirror because the Black Forest Labs repo is gated;
that mirror's licence was not verified. sd.cpp is MIT.

## Configuring the embedding model

In `/editor` → **Embeddings**, choose a preset or paste any GGUF URL, then
**Save & Regenerate**. Settings:

| Setting | Default | Notes |
| --- | --- | --- |
| `EMBED_MODEL_URL` | *(unset)* | blank disables embeddings entirely |
| `EMBED_GPU_LAYERS` | `0` | `0` = CPU (leaves VRAM for the chat model), `99` = GPU (much faster; the reserved VRAM is subtracted from the chat model's budget) |
| `EMBED_CTX` | `4096` | max tokens per embedded chunk; `batch`/`ubatch` follow it because most embedding models are non-causal |
| `EMBED_PARALLEL` | `2` | concurrent embedding slots |
| `EMBED_POOLING` | *(auto)* | `mean` / `cls` / `last`; leave blank to use the GGUF's own setting |
| `EMBED_EXTRA_ARGS` | *(none)* | extra `llama-server` flags, e.g. `--rope-scaling yarn --rope-freq-scale .75` |

Each can also be set as a container env var; `/app/params.json` (written by the
UI) wins when present.

Suggested small models — all good quality for their size:

| Model | Dim | Notes |
| --- | --- | --- |
| Qwen3-Embedding-0.6B | 1024 | best quality/size, multilingual, long context (`--pooling last`) |
| EmbeddingGemma-300M | 768 | smaller and quicker, strong for its size |
| nomic-embed-text-v1.5 | 768 | long context |
| bge-small-en-v1.5 | 384 | tiny, English only |

**Test** in the Embeddings section runs a real request and reports the vector
dimension; **Restart embeddings** reloads the sidecar without touching the chat
model.

- [Dead-man guard](docs/deadman.md): auto-destroy a Vast instance at a deadline.
- [Known issues](docs/known-issues.md): open problems, each labelled observed or hypothesis.
- [Test fakes](tools/fakeserve/README.md): GPU-free fake llama-server and llama-swap for tests.
