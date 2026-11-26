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

# ── repo model registry (models.json) ────────────────────────────────────────
# init.sh fetches models.json next to the scripts. Selected with MODELS=id1,id2
# (ids or preset names). Only active when MODELS is set; then a valid registry
# beats params.json and the MODEL_URL* env vars for the model list.
REGISTRY_FILE = os.environ.get("MODELS_FILE", "/tmp/models.json")
REGISTRY_GOOD = "/app/models.json.good"
CACHE_TYPES = {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1"}
# registry knob -> env var serve.py reads
KNOBS = {"cache_type_k":"CACHE_TYPE_K","cache_type_v":"CACHE_TYPE_V","ctx_size":"CTX_SIZE",
         "gpu_layers":"GPU_LAYERS","fit":"FIT","ctx_overflow":"CTX_OVERFLOW","no_mtp":"NO_MTP"}
DEFAULT_ONLY = {"image_min_tokens":"IMAGE_MIN_TOKENS","image_max_tokens":"IMAGE_MAX_TOKENS",
                "mtmd_batch_max_tokens":"MTMD_BATCH_MAX_TOKENS","mlock":"MLOCK"}
URLS = {"url":"model_url","mmproj":"mmproj_url","draft":"draft_model_url"}
SECRET_WORDS = {"token","secret","password","passwd","credential","credentials","apikey","auth","bearer"}

def _secretish(key):
    w = {x for x in re.split(r"[^a-z0-9]+", key.lower()) if x}
    return bool(w & SECRET_WORDS) or "apikey" in key.lower().replace("_", "")

def _check_url(where, u, errs):
    if not isinstance(u, str) or not u.startswith("https://"):
        errs.append(f"{where}: must be an https:// url"); return
    rest = u[8:].split("/")[0]
    if "@" in rest or re.search(r"[?&](token|key|secret|sig|auth)[a-z_]*=", u, re.I):
        errs.append(f"{where}: url carries credentials")

def _check_knobs(where, d, allowed, errs):
    for k, v in d.items():
        if k in ("note", *URLS) and where != "defaults":
            continue
        if _secretish(k):
            errs.append(f"{where}: secret-like field {k!r} not allowed"); continue
        if k not in allowed:
            errs.append(f"{where}: unknown key {k!r}"); continue
        if k in ("cache_type_k", "cache_type_v", "cache_type"):
            if v not in CACHE_TYPES: errs.append(f"{where}.{k}: {v!r} not in {sorted(CACHE_TYPES)}")
        elif k in ("ctx_size", "gpu_layers", "image_min_tokens", "image_max_tokens", "mtmd_batch_max_tokens"):
            if isinstance(v, bool) or not isinstance(v, int) or v < 0: errs.append(f"{where}.{k}: must be a non-negative integer")
        elif k == "fit":
            if v not in ("auto", "on", "off"): errs.append(f"{where}.fit: must be auto|on|off")
        elif k == "ctx_overflow":
            if v not in ("clamp", "yarn", "linear", "none"): errs.append(f"{where}.ctx_overflow: must be clamp|yarn|linear|none")
        elif k in ("no_mtp", "mlock"):
            if not isinstance(v, bool): errs.append(f"{where}.{k}: must be true/false")

def validate_registry(reg, selected=None):
    """Return a list of problems (empty = valid). `selected` is the MODELS list."""
    errs = []
    if not isinstance(reg, dict): return ["top level must be an object"]
    for k in reg:
        if _secretish(k): errs.append(f"secret-like top-level field {k!r} not allowed")
        elif k not in ("models", "defaults", "presets", "note", "images"): errs.append(f"unknown top-level key {k!r}")
    models = reg.get("models")
    if not isinstance(models, dict) or not models: return errs + ["'models' must be a non-empty object"]
    knobs = set(KNOBS) | {"cache_type"}
    for mid, m in models.items():
        w = f"models.{mid}"
        if not isinstance(m, dict): errs.append(f"{w}: must be an object"); continue
        if "url" not in m: errs.append(f"{w}: missing url"); continue
        for k in URLS:
            if k in m and (k == "url" or m[k]): _check_url(f"{w}.{k}", m[k], errs)
        if isinstance(m["url"], str) and model_stem(m["url"]) != mid:
            errs.append(f"{w}: id must equal the url's file stem {model_stem(m['url'])!r}")
        if "note" in m and not isinstance(m["note"], str): errs.append(f"{w}.note: must be a string")
        _check_knobs(w, {k: v for k, v in m.items() if k not in ("note", *URLS)}, knobs, errs)
    d = reg.get("defaults", {})
    if not isinstance(d, dict): errs.append("defaults: must be an object")
    else: _check_knobs("defaults", d, knobs | set(DEFAULT_ONLY), errs)
    pr = reg.get("presets", {})
    if not isinstance(pr, dict): errs.append("presets: must be an object")
    else:
        for n, ids in pr.items():
            if not isinstance(ids, list) or not all(i in models for i in ids):
                errs.append(f"presets.{n}: must be a list of known model ids")
    for sel in selected or []:
        if sel not in models and sel not in (pr if isinstance(pr, dict) else {}):
            errs.append(f"MODELS: unknown id or preset {sel!r}")
    for s in re.findall(r"hf_[A-Za-z0-9]{20,}", json.dumps(reg)):
        errs.append("a Hugging Face token appears in the registry"); break
    return errs

def registry_selection():
    return [x.strip() for x in os.environ.get("MODELS", "").split(",") if x.strip()]

def registry_active():
    return bool(registry_selection())

def _registry_rows(reg, selected):
    ids = []
    for s in selected:
        for i in (reg.get("presets", {}).get(s) or [s]):
            if i not in ids: ids.append(i)
    d = reg.get("defaults", {})
    rows = []
    for i in ids:
        m = reg["models"][i]
        def expand(b):  # cache_type shorthand sets both; explicit _k/_v in the same block wins
            o = {k: v for k, v in b.items() if k != "cache_type"}
            if "cache_type" in b:
                for k in ("cache_type_k", "cache_type_v"): o.setdefault(k, b["cache_type"])
            return o
        merged = {**expand(d), **expand(m)}
        env = {}
        for k, v in merged.items():
            e = KNOBS.get(k) or DEFAULT_ONLY.get(k)
            if e: env[e] = ("1" if v else "0") if isinstance(v, bool) else str(v)
        if env.get("NO_MTP") == "0": del env["NO_MTP"]
        rows.append({"model_url": m["url"], "mmproj_url": m.get("mmproj", ""),
                     "draft_model_url": m.get("draft", ""), "env": env, "note": m.get("note", "")})
    return rows

def load_registry():
    """Rows from models.json for the models named in MODELS, or None when MODELS
    is unset or no valid registry (fresh copy, then last-good copy) exists. Never
    raises: a bad registry must not stop llama-swap from starting."""
    sel = registry_selection()
    if not sel: return None
    for path, label in ((REGISTRY_FILE, "registry"), (REGISTRY_GOOD, "last-good registry")):
        try:
            raw = open(path).read()
            reg = json.loads(raw)
            errs = validate_registry(reg, sel)
        except Exception as e:
            print(f"[init] ERROR: {label} {path} unusable ({e})", flush=True); continue
        if errs:
            print(f"[init] ERROR: {label} {path} invalid, rejecting it:", flush=True)
            for e in errs: print(f"[init]   - {e}", flush=True)
            continue
        if path == REGISTRY_FILE:
            try:
                tmp = REGISTRY_GOOD + ".tmp"
                open(tmp, "w").write(raw); os.replace(tmp, REGISTRY_GOOD)
            except OSError as e:
                print(f"[init] warn: could not save last-good registry ({e})", flush=True)
        else:
            print(f"[init] WARNING: using {label}; the fresh models.json was missing or invalid", flush=True)
        return _registry_rows(reg, sel)
    print("[init] ERROR: MODELS is set but no valid registry; falling back to params.json/env models", flush=True)
    return None

# ── image / video models (models.json "images", sd.cpp sd-server) ────────────
# Selected with IMAGE_MODELS=id1,id2 (ids or preset names). Unset = disabled:
# nothing is downloaded or written to config.yaml. A bad "images" section only
# disables images; it never affects the chat registry or stops llama-swap.
IMG_KINDS = ("image", "video")
IMG_URLS = {"diffusion_model": "DIFFUSION", "llm": "LLM", "t5xxl": "T5XXL", "vae": "VAE"}
IMG_FLAG_NAMES = {"diffusion_model": "--diffusion-model", "llm": "--llm", "t5xxl": "--t5xxl", "vae": "--vae"}
IMG_REQUIRED = {"image": ("diffusion_model", "llm", "vae"), "video": ("diffusion_model", "t5xxl", "vae")}
IMG_SAMPLERS = {"euler","euler_a","heun","dpm2","dpm++2s_a","dpm++2m","dpm++2mv2","ipndm","ipndm_v","lcm","ddim_trailing","tcd"}
# knob -> (sd-server flag, kind of value, lo, hi)
IMG_NUM = {"cfg_scale": ("--cfg-scale", float, 0, 30), "steps": ("--steps", int, 1, 200),
           "width": ("-W", int, 64, 4096), "height": ("-H", int, 64, 4096),
           "flow_shift": ("--flow-shift", float, 0, 20), "video_frames": ("--video-frames", int, 1, 257)}
IMG_KNOBS = set(IMG_NUM) | {"sampling", "offload_to_cpu"}
IMG_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

def _check_img_knobs(where, d, errs):
    for k, v in d.items():
        if k in IMG_NUM:
            _f, typ, lo, hi = IMG_NUM[k]
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and (typ is float or isinstance(v, int))
            if not ok or not lo <= v <= hi: errs.append(f"{where}.{k}: must be a number in {lo}..{hi}")
            elif k in ("width", "height") and v % 8: errs.append(f"{where}.{k}: must be a multiple of 8")
        elif k == "sampling":
            if v not in IMG_SAMPLERS: errs.append(f"{where}.sampling: not in {sorted(IMG_SAMPLERS)}")
        elif k == "offload_to_cpu":
            if not isinstance(v, bool): errs.append(f"{where}.offload_to_cpu: must be true/false")

def validate_images(img, selected=None):
    """Problems (empty = valid) with the registry's "images" section."""
    errs = []
    if not isinstance(img, dict): return ["images: must be an object"]
    for k in img:
        if _secretish(k): errs.append(f"images: secret-like field {k!r} not allowed")
        elif k not in ("defaults", "models", "presets", "note"): errs.append(f"images: unknown key {k!r}")
    if "note" in img and not isinstance(img["note"], str): errs.append("images.note: must be a string")
    models = img.get("models")
    if not isinstance(models, dict) or not models: return errs + ["images.models: must be a non-empty object"]
    for mid, m in models.items():
        w = f"images.models.{mid}"
        if not IMG_ID_RE.match(str(mid)): errs.append(f"{w}: id must match [A-Za-z0-9._-]{{1,64}}")
        if not isinstance(m, dict): errs.append(f"{w}: must be an object"); continue
        for k, v in m.items():
            if _secretish(k): errs.append(f"{w}: secret-like field {k!r} not allowed")
            elif k in IMG_URLS: _check_url(f"{w}.{k}", v, errs)
            elif k == "kind":
                if v not in IMG_KINDS: errs.append(f"{w}.kind: must be one of {list(IMG_KINDS)}")
            elif k == "note":
                if not isinstance(v, str): errs.append(f"{w}.note: must be a string")
            elif k not in IMG_KNOBS: errs.append(f"{w}: unknown key {k!r}")
        if m.get("kind") not in IMG_KINDS:
            if "kind" not in m: errs.append(f"{w}: missing kind")
        else:
            for k in IMG_REQUIRED[m["kind"]]:
                if k not in m: errs.append(f"{w}: kind {m['kind']!r} needs {k}")
        _check_img_knobs(w, {k: v for k, v in m.items() if k in IMG_KNOBS}, errs)
    d = img.get("defaults", {})
    if not isinstance(d, dict): errs.append("images.defaults: must be an object")
    else:
        for k in d:
            if _secretish(k): errs.append(f"images.defaults: secret-like field {k!r} not allowed")
            elif k not in IMG_KNOBS: errs.append(f"images.defaults: unknown key {k!r}")
        _check_img_knobs("images.defaults", {k: v for k, v in d.items() if k in IMG_KNOBS}, errs)
    pr = img.get("presets", {})
    if not isinstance(pr, dict): errs.append("images.presets: must be an object")
    else:
        for n, ids in pr.items():
            if not isinstance(ids, list) or not all(i in models for i in ids):
                errs.append(f"images.presets.{n}: must be a list of known image ids")
    for sel in selected or []:
        if sel not in models and sel not in (pr if isinstance(pr, dict) else {}):
            errs.append(f"IMAGE_MODELS: unknown id or preset {sel!r}")
    return errs

def image_selection():
    return [x.strip() for x in os.environ.get("IMAGE_MODELS", "").split(",") if x.strip()]

def _image_rows(img, selected):
    ids = []
    for s in selected:
        for i in (img.get("presets", {}).get(s) or [s]):
            if i not in ids: ids.append(i)
    d = img.get("defaults", {})
    rows = []
    for i in ids:
        m = {**d, **img["models"][i]}
        flags = []
        for k, (flag, typ, _lo, _hi) in IMG_NUM.items():
            if k in m: flags += [flag, str(m[k])]
        if "sampling" in m: flags += ["--sampling-method", m["sampling"]]
        if m.get("offload_to_cpu"): flags.append("--offload-to-cpu")
        rows.append({"id": i, "kind": m["kind"], "note": m.get("note", ""), "flags": flags,
                     "urls": {k: m[k] for k in IMG_URLS if k in m}})
    return rows

def load_images():
    """Rows for the entries named in IMAGE_MODELS, or [] when it is unset or no
    valid images section exists (fresh file, then the last-good copy). Never
    raises."""
    sel = image_selection()
    if not sel: return []
    for path, label in ((REGISTRY_FILE, "registry"), (REGISTRY_GOOD, "last-good registry")):
        try:
            raw = open(path).read()
            reg = json.loads(raw)
            if not isinstance(reg, dict) or "images" not in reg: raise ValueError("no images section")
            errs = validate_images(reg["images"], sel)
            if re.search(r"hf_[A-Za-z0-9]{20,}", raw): errs.append("a Hugging Face token appears in the registry")
        except Exception as e:
            print(f"[init] ERROR: {label} {path} unusable for images ({e})", flush=True); continue
        if errs:
            print(f"[init] ERROR: {label} {path} images section invalid, rejecting it:", flush=True)
            for e in errs: print(f"[init]   - {e}", flush=True)
            continue
        if path == REGISTRY_FILE and not validate_registry(reg):
            try:
                tmp = REGISTRY_GOOD + ".tmp"
                open(tmp, "w").write(raw); os.replace(tmp, REGISTRY_GOOD)
            except OSError as e:
                print(f"[init] warn: could not save last-good registry ({e})", flush=True)
        elif path != REGISTRY_FILE:
            print(f"[init] WARNING: using {label} for images; the fresh models.json was missing or invalid", flush=True)
        return _image_rows(reg["images"], sel)
    print("[init] ERROR: IMAGE_MODELS is set but no valid images section; image generation is DISABLED", flush=True)
    return []

def load_params(with_source=False):
    """Model list: valid models.json (when MODELS is set) > params.json > env.
    Settings and embedding still come from params.json/env. "images" is the
    image/video rows from IMAGE_MODELS (never saved to params.json)."""
    p, src = _load_params()
    rows = load_registry()
    if rows is not None:
        p = dict(p, models=rows); src = "registry"
    p = dict(p, images=load_images())
    return (p, src) if with_source else p

def _load_params():
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
                return p, PARAMS_FILE
        except Exception as e:
            print(f"[init] warn: {PARAMS_FILE} unusable ({e}), falling back to env", flush=True)
    return _env_params(), source

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
            # One entry per variant: serve.py picks the slot count itself
            # (PARALLEL=auto), and the editor's slot slider overrides it per model.
            mid = f"{MN}{sfxv}"
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
            own = m.get("env") or {}
            for k, v in own.items():
                if k != "NO_MTP" or not no_mtp:
                    lines.append(f'      - "{k}={v}"')
            for k in PASS_KEYS:
                v = params.get("settings", {}).get(k)
                if v and k not in own:
                    lines.append(f'      - "{k}={v}"')
            lines.append(f'      - "PARALLEL=auto"')
            lines.append(f'      - "MODEL_ID={mid}"')
            lines.append(f"    cmd: python3 /tmp/serve.py ${{PORT}}")
            lines.append(f"    logFile: /tmp/serve-{mid}.log")
    # Image/video models: no group is emitted, same as the chat models above, so
    # they share llama-swap's one default group and swap with the chat models.
    for r in params.get("images") or []:
        mid = r["id"]
        if mid in emitted:
            print(f"[init] ERROR: image model id {mid!r} clashes with a chat model id; skipped", flush=True)
            continue
        emitted.add(mid); found += 1
        lines.append(f"  {mid!r}:")
        lines.append(f'    proxy: "http://127.0.0.1:${{PORT}}"')
        lines.append(f"    env:")
        for k, v in r["urls"].items():
            lines.append(f'      - "IMG_URL_{IMG_URLS[k]}={v}"')
        lines.append(f'      - "IMG_FLAGS={" ".join(r["flags"])}"')
        for k in PASS_KEYS:
            v = params.get("settings", {}).get(k)
            if v and (k.startswith("DOWNLOAD") or k.startswith("HF_")):
                lines.append(f'      - "{k}={v}"')
        lines.append(f'      - "MODEL_ID={mid}"')
        lines.append(f"    cmd: python3 /tmp/imgserve.py ${{PORT}}")
        lines.append(f"    checkEndpoint: /sdcpp/v1/capabilities")
        lines.append(f"    ttl: 0")
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

DEFAULT_MODEL_FILE = "/app/default_model"

def migrate_default_model(cfg):
    """Model ids used to carry a slot count (<name>-p1, <name>-p2-text, ...).
    Point a default model saved under such an id at the id that replaced it."""
    try:
        cur = open(DEFAULT_MODEL_FILE).read().strip()
    except OSError:
        return
    ids = set(re.findall(r"^  '(.*)':$", cfg, re.M))
    new = re.sub(r"-p[1248](?=(-nomtp|-text)?$)", "", cur)
    if cur and cur not in ids and new in ids:
        open(DEFAULT_MODEL_FILE, "w").write(new)
        print(f"[init] default model {cur!r} -> {new!r}", flush=True)

def main():
    params = load_params()
    cfg, found = build_config(params)
    write_config(cfg)
    migrate_default_model(cfg)
    print(f"[init] wrote /app/config.yaml with {found} model(s)", flush=True)

if __name__ == "__main__":
    main()
