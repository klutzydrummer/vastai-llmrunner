import os, json

PARAMS_FILE = "/app/params.json"
PASS_KEYS = ["HF_TOKEN","DOWNLOADER","HF_BACKEND","CACHE_TYPE_K","CACHE_TYPE_V",
             "GPU_LAYERS","MLOCK","IMAGE_MIN_TOKENS","IMAGE_MAX_TOKENS",
             "MTMD_BATCH_MAX_TOKENS","COMPUTE_FRACTION"]

def _env_params():
    models = []
    suffixes = [""] + [f"_{i}" for i in range(2, 20)]
    for sfx in suffixes:
        mu = os.environ.get(f"MODEL_URL{sfx}", "")
        if not mu:
            break
        models.append({
            "model_url": mu,
            "mmproj_url": os.environ.get(f"MMPROJ_URL{sfx}", ""),
            "draft_model_url": os.environ.get(f"DRAFT_MODEL_URL{sfx}", ""),
        })
    settings = {k: os.environ.get(k, "") for k in PASS_KEYS}
    return {"models": models, "settings": settings}

def load_params():
    """Params saved via the cfgedit UI (/app/params.json) take priority over
    the container's env vars, mirroring the /app/downloader and /app/cache_type
    override files. Falls back to env vars (MODEL_URL, MODEL_URL_2, ...) on
    first boot before any params have been saved."""
    if os.path.exists(PARAMS_FILE):
        try:
            p = json.load(open(PARAMS_FILE))
            if p.get("models"):
                p.setdefault("settings", {})
                for k in PASS_KEYS:
                    p["settings"].setdefault(k, "")
                return p
        except Exception:
            pass
    return _env_params()

def save_params(params):
    models = [{"model_url": (m.get("model_url") or "").strip(),
               "mmproj_url": (m.get("mmproj_url") or "").strip(),
               "draft_model_url": (m.get("draft_model_url") or "").strip()}
              for m in params.get("models", []) if (m.get("model_url") or "").strip()]
    settings = {k: (params.get("settings", {}).get(k) or "").strip() for k in PASS_KEYS}
    p = {"models": models, "settings": settings}
    open(PARAMS_FILE, "w").write(json.dumps(p, indent=2))
    return p

def build_config(params):
    lines = ["healthCheckTimeout: 3600", "sendLoadingState: true", "models:"]
    found = 0
    for m in params.get("models", []):
        MU = (m.get("model_url") or "").strip()
        if not MU:
            continue
        MMU = (m.get("mmproj_url") or "").strip()
        DMU = (m.get("draft_model_url") or "").strip()
        MN = MU.split("/")[-1].replace(".gguf", "")
        found += 1
        # variants: (suffix, include_mmproj, include_draft, no_mtp)
        variants = [('', True, True, False)]        # default: mmproj + draft
        if MMU:
            variants.append(('-nomtp', True, False, True))   # mmproj, no draft, spec disabled
            variants.append(('-text', False, True, False))   # no mmproj, draft only
        for sfxv, use_mm, use_dm, no_mtp in variants:
            for par in [1, 2, 4, 8]:
                mid = f"{MN}-p{par}{sfxv}"
                lines.append(f"  {mid!r}:")
                lines.append(f'    proxy: "http://127.0.0.1:${{PORT}}"')
                lines.append(f"    env:")
                lines.append(f'      - "MODEL_URL={MU}"')
                # Always set MMPROJ_URL/DRAFT_MODEL_URL explicitly (even blank) so the
                # spawned serve.py subprocess never falls back to inheriting the
                # container's unsuffixed (model 1) env vars for models that don't
                # define their own mmproj/draft model.
                lines.append(f'      - "MMPROJ_URL={MMU if use_mm else ""}"')
                lines.append(f'      - "DRAFT_MODEL_URL={DMU if use_dm else ""}"')
                if no_mtp:
                    lines.append(f'      - "NO_MTP=1"')
                for k in PASS_KEYS:
                    v = params.get("settings", {}).get(k)
                    if v:
                        lines.append(f'      - "{k}={v}"')
                lines.append(f'      - "PARALLEL={par}"')
                lines.append(f"    cmd: python3 /tmp/serve.py ${{PORT}}")
                lines.append(f"    logFile: /tmp/serve-{mid}.log")
    return "\n".join(lines) + "\n", found

def main():
    params = load_params()
    cfg, found = build_config(params)
    open("/app/config.yaml", "w").write(cfg)
    print(f"[init] wrote /app/config.yaml with {found} model(s)", flush=True)

if __name__ == "__main__":
    main()
