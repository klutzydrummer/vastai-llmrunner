# vastai-llmrunner

Boots a llama.cpp stack on a Vast.ai (or any CUDA) container: `llama-swap` for
hot-swappable chat models, a small always-on embedding model, a config UI, and a
Cloudflare tunnel — all behind one URL, which is what SillyTavern wants.

## Endpoints (port 5000 / your tunnel URL)

| Path | Served by | Notes |
| --- | --- | --- |
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | guard → llama-swap | chat model, swapped on demand |
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
- Pick the model from the list (`…-p1` = full context for a single user,
  `-p2`/`-p4` split it between slots).

**Vector storage (embeddings)** — Extensions → Vector Storage:

- Source **vLLM** (or any OpenAI-compatible source): URL `https://<your-tunnel>`,
  model name anything (the sidecar serves whichever model it loaded).
- or Source **llama.cpp**: URL `https://<your-tunnel>` — this hits `/embedding`.

Re-vectorize your chats after changing the embedding model: different models
produce different vector spaces and dimensions.

## Model IDs in the editor

Each row in the editor's **Models** table becomes four entries in
`config.yaml` — `<name>-p1`/`-p2`/`-p4`/`-p8` — where `<name>` comes from the
GGUF file name in the model URL, minus the `.gguf` extension and any
`?download=true` style query string.

Two different models can share a file name (the same quant published by two
repos, for example). Those names are made unique: the first model to claim a
name keeps it, and a later one is prefixed with its repo owner
(`unsloth-Model-Q4_K_M-p1`), falling back to a counter. Adding a second row
with an identical model URL is dropped on save, since it would only ever
produce the same entry.

After **Save & Regenerate**, the **Default model** dropdown refreshes in
place — no page reload needed.

## Exposing only the active model

`/editor` has an **Expose only active model to /v1/models** checkbox. With it on,
`/v1/models` lists just the model that is actually loaded (or, if nothing is
loaded, the configured default model) instead of every `-p1`/`-p2`/`-p4`/`-p8`
variant in `config.yaml`. Clients like SillyTavern then can't pick a model that
would trigger a swap, and their model dropdown stays readable.

It is off by default; the container env var `EXPOSE_ACTIVE_ONLY=1` turns it on
at boot, and `/app/expose_active_only` (written by the UI) wins when present.
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

## Context length

The status bar in `/editor` shows the loaded model's context — tokens per slot,
total context, slot count, and the context the model was trained for — with a
**copy context length** button that copies the per-slot number, which is the
value to put in a client's context size setting.

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
