"""Launcher llama-swap runs for one sd.cpp image/video model (cfginit writes the
entry). Downloads the model files with serve.py's downloader (same .part and
size checks, same cache dir), then execs sd-server on the given port."""
import os, sys, hashlib

SD_BIN = os.environ.get("SD_BIN", "/opt/sd/sd-server")
URL_FLAGS = (("DIFFUSION", "--diffusion-model"), ("LLM", "--llm"), ("T5XXL", "--t5xxl"), ("VAE", "--vae"))

def sd_args(port, paths, flags):
    """sd-server argv (after the binary). `paths` maps DIFFUSION/LLM/T5XXL/VAE to local files."""
    a = ["--listen-ip", "127.0.0.1", "--listen-port", str(port)]
    for key, flag in URL_FLAGS:
        if paths.get(key): a += [flag, paths[key]]
    return a + flags.split()

def _load_serve(port):
    """serve.py runs a whole model launch at import, so exec only its top half (the
    download helpers), as tests/test_downloads.py does. serve.py itself is untouched."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "serve.py")).read()
    src = src[:src.index("import hashlib as _hl")]
    ns = {"__name__": "serve_downloads"}
    sys.argv = ["serve.py", str(port)]
    exec(compile(src, "serve.py", "exec"), ns)
    return ns

def main():
    port = sys.argv[1]
    urls = {k: os.environ.get(f"IMG_URL_{k}", "") for k, _ in URL_FLAGS}
    if not urls["DIFFUSION"]:
        print("[imgserve] ERROR: IMG_URL_DIFFUSION is not set", flush=True); sys.exit(1)
    if not os.path.isfile(SD_BIN):
        print(f"[imgserve] ERROR: {SD_BIN} is missing (init.sh could not install sd.cpp); "
              "see /tmp/init.log or the boot log for the reason", flush=True); sys.exit(1)
    os.environ["MODEL_URL"] = urls["DIFFUSION"]
    ns = _load_serve(port)
    want = [u for u in urls.values() if u]
    keep = {f'{ns["MODEL_DIR"]}/{hashlib.md5(u.encode()).hexdigest()[:8]}_{u.split("/")[-1]}' for u in want}
    got = ns["dl_all"](want, keep)
    paths = {k: got[u] for k, u in urls.items() if u}
    args = sd_args(port, paths, os.environ.get("IMG_FLAGS", ""))
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")  # PTX JIT on GPUs without native kernels
    print(f"[imgserve] exec {SD_BIN} {args}", flush=True)
    os.execv(SD_BIN, [SD_BIN] + args)

if __name__ == "__main__":
    main()
