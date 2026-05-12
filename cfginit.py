import os

PASS_KEYS = ["HF_TOKEN","DOWNLOADER","HF_BACKEND","CACHE_TYPE_K","CACHE_TYPE_V",
             "GPU_LAYERS","MLOCK","IMAGE_MIN_TOKENS","IMAGE_MAX_TOKENS","COMPUTE_FRACTION"]
lines = ["healthCheckTimeout: 3600", "sendLoadingState: true", "models:"]
suffixes = [""] + [f"_{i}" for i in range(2, 20)]
found = 0

for sfx in suffixes:
    MU = os.environ.get(f"MODEL_URL{sfx}", "")
    if not MU:
        break
    MMU = os.environ.get(f"MMPROJ_URL{sfx}", "")
    MN = MU.split("/")[-1].replace(".gguf", "")
    found += 1
    for par in [1, 2, 4, 8]:
        mid = f"{MN}-p{par}"
        lines.append(f"  {mid!r}:")
        lines.append(f'    proxy: "http://127.0.0.1:${{PORT}}"')
        lines.append(f"    env:")
        lines.append(f'      - "MODEL_URL={MU}"')
        if MMU:
            lines.append(f'      - "MMPROJ_URL={MMU}"')
        for k in PASS_KEYS:
            v = os.environ.get(k)
            if v:
                lines.append(f'      - "{k}={v}"')
        lines.append(f'      - "PARALLEL={par}"')
        lines.append(f"    cmd: python3 /tmp/serve.py ${{PORT}}")

open("/app/config.yaml", "w").write("\n".join(lines) + "\n")
print(f"[init] wrote /app/config.yaml with {found} model(s)", flush=True)