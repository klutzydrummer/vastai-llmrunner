# Known issues

## Pantheon MTP draft OOM
- Observed: the MTP draft model's compute buffer needed 6229 MiB and `cudaMalloc` failed (OOM).
- Observed: the planner had estimated 43 GB for this run at ctx 1,238,016,
  parallel 4, batch = ctx, ubatch 2048, KV q8_0, yarn.
- Hypothesis (still unconfirmed): the huge context with batch == ctx inflates
  the draft compute buffer. Not measured: no llama-server binary or GGUF was
  available locally (2026-11-26), so no compute-buffer table exists. Public
  sources only say that a larger physical batch enlarges the compute buffer.
- Planner change (precaution, not proof): `--batch-size` is now capped at 4096
  (never below ubatch) and a draft-model compute-buffer term, scaled by ubatch,
  is subtracted from the KV budget. Its size is an unverified estimate.
- Confirming on a GPU still needs an approved rental.
- Not isolated: the template env `CTX_OVERFLOW`, `CACHE_TYPE` and `PARALLEL`
  were not varied one at a time, so their contribution is unknown.
- Observed: the llama-swap `logFile` was never written, so there is no server log for this failure.
- Deferred: reproduction waits for a future user-approved rental.

## Media routing (RTR) limits
- Every swap re-prefills the role-play model. The RP always loads with its
  mmproj, and llama-server cannot save a slot while an mmproj is loaded, so no
  slot save/restore is built. After a swap the RP's next reply re-reads the
  whole chat. There is no RP context cap, so at long context this re-prefill
  is likely the largest part of a swap. RTR reports it as
  `timings.prefill_rp_s` when the request sends `rp_prefill`; the real time has
  not been measured [U], so the extension's swap estimate stays "unknown" for
  that term until it has.
- Shared single slot when the media model is the role-play model [U]. With a
  `PARALLEL=1` media model serving as the RP (same model, or a running media
  model counted as the RP), a chat request queues behind the media extract on
  the one slot. A long extract can use up guard's 600 s generation timeout,
  and the extract replaces the chat's cached prompt, so the next chat reply
  re-reads the whole chat. Nothing is built for this; it is to be measured on
  Run 2 first (see below).
- Qwen3-Omni Q4_K_M VRAM is unmeasured. Its `overhead_mb` is copied from the
  Q8 measurement (34,054 MiB at ctx 8192 on an A6000), so the ~21 GiB figure
  and the Q4 coexist sets built from it are estimates. Q4 quality is also
  unchecked.
- Transcript fallback: when the SillyTavern side cannot upload a transcript
  file, the dossier is kept without it ("transcript not kept"), and when a
  pointed-to file is missing or fails its check on this device the item shows
  "transcript not on this device". In both cases a follow-up that needs the
  transcript processes the media again.
- RTR listens on 127.0.0.1 only and is reached through guard. During a swap
  guard holds other generation requests for up to `SWAP_HOLD_S` (600 s), then
  answers 503 "media swap in progress" with `Retry-After`. A request that
  passed guard's check just before a swap started can still reach llama-swap
  while the RP is being unloaded; llama-swap then waits for it. Not seen live.
- Tested on fakes only. All RTR and guard tests run against the fake
  llama-swap and llama-server in `tests/`. The real llama-swap v277 matrix
  solver, real VRAM behaviour, the `nvidia-smi` free-memory parsing (summed
  over GPUs) and the `llama-server --version` parsing have not been run
  against the real programs. Wrong VRAM or version data falls back to swap or
  is informational only.

## Run 2 measurement items
Not measured yet; to be timed on the next approved rental (Run 2). None of
these is run or approved by this repo.
- R1, shared single slot: with a `PARALLEL=1` media model as the RP, how long
  chat waits behind an extract, whether it reaches guard's 600 s timeout, and
  the re-prefill cost after the extract replaces the cached prompt.
- `timings.prefill_rp_s` on a long chat after a swap reload.
- Real browser CORS from the SillyTavern origin against guard's media route.

## Download verification
- Every downloaded model file is hashed once after its download (see
  "sha256 verification" in the README). This adds one full read of each file
  before a fresh model is ready; the time on real files has not been measured.
- A file with nothing to compare against stays "unverified". It keeps its
  `.verify` marker, so every boot asks the Hub again and repeats the warning,
  and hashes the file once the Hub answers.
- The expected-hash cache is keyed by the URL's revision as written (usually
  `main`). If `main` moves while the Hub lookup is also failing, the stale
  cached hash shows up as a mismatch error.
- The `models.json` `sha256` field covers only an entry's `url`. A non-HF
  mmproj, draft or embedding URL has no hash source and is always unverified.
