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
