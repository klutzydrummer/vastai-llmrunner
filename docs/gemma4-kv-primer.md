# Gemma4 / Pantheon KV cache primer

Checked 2026-11-25 against `mradermacher/Pantheon-Reasoning-26B-A4B-1.1-V2-GGUF`
(Q8_0), base `Gryphe/Pantheon-Reasoning-26B-A4B-1.1-V2`. Method: HF `config.json`,
the GGUF header (HTTP range read), and llama.cpp master source reading. No Gemma4
model was loaded, so real VRAM use is not measured.

Sources:
- https://huggingface.co/Gryphe/Pantheon-Reasoning-26B-A4B-1.1-V2/raw/main/config.json
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/constants.py
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/gemma4.cpp
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-kv-cache-iswa.cpp
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-kv-cache.cpp

## Layout (verified)

- 30 layers, 16 attention heads, trained context 262144.
- 25 sliding-window (SWA) layers: window 1024, 8 KV heads, head dim 256.
- 5 full-attention layers (indices 5, 11, 17, 23, 29): 2 KV heads, head dim 512.
- In the GGUF, `head_count_kv` is a per-layer array (`[8,8,8,8,8,2,...]`) and
  `sliding_window_pattern` is a per-layer bool array. Head dims come from
  `key_length`/`value_length` (512) and the `_swa` variants (256).

## Per-token cost at Q8_0

- Full layers: 5 x 2 x (512+512) = 10,240 elements, about 10,880 B per token of context.
- SWA layers: 25 x 8 x (256+256) = 102,400 elements, about 108,800 B per cell, but
  the cell count is capped near the window, not the context:
  `GGML_PAD(min(ctx, n_swa*n_seq_max + n_ubatch), 256)` (`llama-kv-cache-iswa.cpp:73`;
  `--swa-full` at line 76 makes SWA layers scale with ctx instead).
- The old flat formula would charge about 89.8 KB per token, hence the per-layer budget.

## Resulting auto context (`tests/test_kv_budget.py`)

| VRAM | auto ctx |
|------|----------|
| 40 GiB | 526,848 |
| 48 GiB | 1,221,632 |

Weights are about 25.75 GiB, consistent with the README's 27.7 GB for Q8_0.

## attention_k_eq_v

HF sets `attention_k_eq_v: true`, and the GGUF has an `attn_v` tensor only on the 25
SWA layers; full layers have none. llama.cpp (`llama-kv-cache.cpp`, around line 210)
still allocates a V cache for every layer using `n_embd_v_gqa(il)`, with no skip.
So serve.py's separate K and V accounting matches the allocator, and if anything
overestimates, which makes auto context a safe underestimate. (Code reading only.)

## Still unverified

- Q8_0 = 34 B per 32 elements (1.0625 B/element): standard ggml knowledge, the
  `block_q8_0` definition was not read.
- Actual VRAM use of a loaded Gemma4 model. serve.py's weight, ubatch (2240) and
  slot settings are config choices, not model facts.
