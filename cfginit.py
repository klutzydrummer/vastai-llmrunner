import os, json, re

PARAMS_FILE = "/app/params.json"
PASS_KEYS = ["HF_TOKEN","DOWNLOADER","HF_BACKEND","DOWNLOAD_CONNECTIONS",
             "DOWNLOAD_PARALLEL","CACHE_TYPE_K","CACHE_TYPE_V",
             "GPU_LAYERS","MLOCK","IMAGE_MIN_TOKENS","IMAGE_MAX_TOKENS",
             "MTMD_BATCH_MAX_TOKENS","COMPUTE_FRACTION","CTX_OVERFLOW","FIT"]
# Settings for the always-on embeddings sidecar (embed.py), which runs outside
# llama-swap so embeddings stay available while chat models are swapped.
EMBED_KEYS = ["EMBED_MODEL_URL","EMBED_GPU_LAYERS","EMBED_CTX","EMBED_POOLING",
              "EMBED_PARALLEL","EMBED_EXTRA_ARGS"]

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
    embedding = {k: os.environ.get(k, "") for k in EMBED_KEYS}
    return {"models": models, "settings": settings, "embedding": embedding}

def load_params(with_source=False):
    """Params saved via the cfgedit UI (/app/params.json) take priority over
    the container's env vars, mirroring the /app/downloader and /app/cache_type
    override files. Falls back to env vars (MODEL_URL, MODEL_URL_2, ...) only
    when no usable params file exists — a saved file is honoured even when its
    model list is empty, because falling back there would silently resurrect
    the container's env models over what was just saved."""
    source = "env"
    if os.path.exists(PARAMS_FILE):
        try:
            p = json.load(open(PARAMS_FILE))
            if isinstance(p.get("models"), list):
                p.setdefault("settings", {})
                for k in PASS_KEYS:
                    p["settings"].setdefault(k, "")
                # params.json files written before embeddings existed have no
                # "embedding" block — seed it from the env so EMBED_MODEL_URL
                # keeps working after an upgrade.
                p.setdefault("embedding", {})
                for k in EMBED_KEYS:
                    p["embedding"].setdefault(k, os.environ.get(k, ""))
                return (p, PARAMS_FILE) if with_source else p
        except Exception as e:
            print(f"[init] warn: {PARAMS_FILE} unusable ({e}), falling back to env", flush=True)
    p = _env_params()
    return (p, source) if with_source else p

def save_params(params):
    models = []
    seen = set()
    for m in params.get("models", []):
        mu = (m.get("model_url") or "").strip()
        # Two rows with the same model URL would collapse into one config entry
        # anyway, so drop the repeat instead of silently losing it later.
        if not mu or mu in seen:
            continue
        seen.add(mu)
        models.append({"model_url": mu,
                       "mmproj_url": (m.get("mmproj_url") or "").strip(),
                       "draft_model_url": (m.get("draft_model_url") or "").strip()})
    settings = {k: (params.get("settings", {}).get(k) or "").strip() for k in PASS_KEYS}
    embedding = {k: (params.get("embedding", {}).get(k) or "").strip() for k in EMBED_KEYS}
    p = {"models": models, "settings": settings, "embedding": embedding}
    open(PARAMS_FILE, "w").write(json.dumps(p, indent=2))
    return p

def model_stem(url):
    """Config-safe name for a GGUF URL: the file name, without any query string
    or fragment, without the .gguf extension, and with anything that would be
    awkward in a model ID (?, &, spaces, ...) folded to '_'."""
    path = url.split("#")[0].split("?")[0].rstrip("/")
    base = path.split("/")[-1]
    if base.lower().endswith(".gguf"):
        base = base[:-len(".gguf")]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_.-")
    return base or "model"

def _url_owner(url):
    """Repo owner from a Hugging Face style URL, used to tell apart two models
    whose file names happen to be identical (e.g. the same quant from two
    repos). Empty when the URL has no usable owner segment."""
    path = url.split("#")[0].split("?")[0]
    path = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", path)
    parts = [p for p in path.split("/") if p]
    owner = parts[1] if len(parts) > 2 else ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", owner).strip("_.-")

def model_stems(models):
    """One unique stem per model, in order. The first model to claim a name
    keeps it unchanged, so a default_model saved earlier keeps resolving; later
    collisions are qualified by repo owner, then by a counter."""
    stems, taken = [], set()
    for m in models:
        mu = (m.get("model_url") or "").strip()
        if not mu:
            continue
        base = model_stem(mu)
        cand = base
        if cand in taken:
            owner = _url_owner(mu)
            cand = f"{owner}-{base}" if owner else base
            n = 2
            while cand in taken:
                cand = f"{base}-{n}"
                n += 1
        taken.add(cand)
        stems.append(cand)
    return stems

def build_config(params):
    lines = ["healthCheckTimeout: 3600", "sendLoadingState: true", "models:"]
    found = 0
    entries = [m for m in params.get("models", []) if (m.get("model_url") or "").strip()]
    stems = model_stems(entries)
    emitted = set()
    for m, MN in zip(entries, stems):
        MU = (m.get("model_url") or "").strip()
        MMU = (m.get("mmproj_url") or "").strip()
        DMU = (m.get("draft_model_url") or "").strip()
        found += 1
        # variants: (suffix, include_mmproj, include_draft, no_mtp)
        variants = [('', True, True, False)]        # default: mmproj + draft
        if MMU:
            variants.append(('-nomtp', True, False, True))   # mmproj, no draft, spec disabled
            variants.append(('-text', False, True, False))   # no mmproj, draft only
        for sfxv, use_mm, use_dm, no_mtp in variants:
            for par in [1, 2, 4, 8]:
                mid = f"{MN}-p{par}{sfxv}"
                # Stems are unique, but a stem that already ends in a variant
                # suffix could still collide; a duplicate YAML key would drop a
                # model from /v1/models silently, so keep every id distinct.
                if mid in emitted:
                    n = 2
                    while f"{mid}-{n}" in emitted:
                        n += 1
                    mid = f"{mid}-{n}"
                emitted.add(mid)
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

def write_config(cfg, path="/app/config.yaml"):
    """Write config.yaml atomically: llama-swap runs with --watch-config, and a
    plain truncate-then-write lets it reload a partial (or empty) file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(cfg)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def main():
    params = load_params()
    cfg, found = build_config(params)
    write_config(cfg)
    print(f"[init] wrote /app/config.yaml with {found} model(s)", flush=True)

if __name__ == "__main__":
    main()
