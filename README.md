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
