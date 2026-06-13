import os

PASS_KEYS = ["HF_TOKEN","DOWNLOADER","HF_BACKEND","CACHE_TYPE_K","CACHE_TYPE_V",
             "GPU_LAYERS","MLOCK","IMAGE_MIN_TOKENS","IMAGE_MAX_TOKENS","COMPUTE_FRACTION"]
lines = ["healthCheckTimeout: 3600", "sendLoadingState: true", "models:"]
suffixes = [""] + [f"_{i}" for i in range(2, 20)]

models = []
for sfx in suffixes:
    MU = os.environ.get(f"MODEL_URL{sfx}", "")
    if not MU:
        break
    MMU = os.environ.get(f"MMPROJ_URL{sfx}", "")
    DMU = os.environ.get(f"DRAFT_MODEL_URL{sfx}", "")
    models.append((MU, MMU, DMU))

multi = len(models) > 1

for idx, (MU, MMU, DMU) in enumerate(models, 1):
    prefix = f"{idx}-" if multi else ""
    variants = [False]
    if MMU:
        variants.append(True)  # True = text-only (no mmproj)
    for text_only in variants:
        for par in [1, 2, 4, 8]:
            mid = f"{prefix}p{par}{'-text' if text_only else ''}"
            lines.append(f"  {mid!r}:")
            lines.append(f'    proxy: "http://127.0.0.1:${{PORT}}"')
            lines.append(f"    env:")
            lines.append(f'      - "MODEL_URL={MU}"')
            if MMU and not text_only:
                lines.append(f'      - "MMPROJ_URL={MMU}"')
            if DMU:
                lines.append(f'      - "DRAFT_MODEL_URL={DMU}"')
            for k in PASS_KEYS:
                v = os.environ.get(k)
                if v:
                    lines.append(f'      - "{k}={v}"')
            lines.append(f'      - "PARALLEL={par}"')
            lines.append(f"    cmd: python3 /tmp/serve.py ${{PORT}}")
            lines.append(f"    logFile: /tmp/serve-{mid}.log")

open("/app/config.yaml", "w").write("\n".join(lines) + "\n")
print(f"[init] wrote /app/config.yaml with {len(models)} model(s)", flush=True)
