# Known issues

## Pantheon MTP draft OOM
- Observed: the MTP draft model's compute buffer needed 6229 MiB and `cudaMalloc` failed (OOM).
- Observed: the planner had estimated 43 GB for this run at ctx 1,238,016,
  parallel 4, batch = ctx, ubatch 2048, KV q8_0, yarn.
- Hypothesis: the huge context with batch == ctx inflates the draft compute
  buffer. Not confirmed.
- Not isolated: the template env `CTX_OVERFLOW`, `CACHE_TYPE` and `PARALLEL`
  were not varied one at a time, so their contribution is unknown.
- Observed: the llama-swap `logFile` was never written, so there is no server log for this failure.
- Deferred: reproduction waits for a future user-approved rental.
