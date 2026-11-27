#!/usr/bin/env bash
set -euo pipefail

LOG(){ echo "[$(date '+%H:%M:%S')] [init] $*"; }
ERR(){ echo "[$(date '+%H:%M:%S')] [init] ERROR: $*" >&2; }
die(){ ERR "$*"; exit 1; }

# Release assets are named by Go arch. Some hosts (e.g. GH200) are aarch64,
# where an amd64 binary fails with "Exec format error".
case "$(uname -m)" in
    x86_64|amd64)  ARCH=amd64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) die "unsupported CPU architecture: $(uname -m)" ;;
esac

# llama-swap is pinned: cfginit's config relies on its routing keys, which
# change between releases. Newest release when pinned (2026-11-27). Override
# with LLAMA_SWAP_TAG=vNNN, or LLAMA_SWAP_TAG=latest to resolve at boot.
LLAMA_SWAP_TAG="${LLAMA_SWAP_TAG:-v277}"

# retry MAX INITIAL_DELAY_SECS CMD [ARGS...]  — exponential backoff wrapper
retry(){
    local max="$1" delay="$2"; shift 2
    local n=1
    while true; do
        if "$@"; then return 0; fi
        if (( n >= max )); then ERR "retry: \"$*\" failed after $max attempts"; return 1; fi
        ERR "retry: attempt $n/$max for \"$*\" failed, retrying in ${delay}s"
        sleep "$delay"
        delay=$(( delay * 2 ))
        n=$(( n + 1 ))
    done
}

# wait_port NAME PORT [TIMEOUT_SECS]  — block until port accepts connections
wait_port(){
    local name="$1" port="$2" timeout="${3:-30}"
    local deadline=$(( SECONDS + timeout ))
    LOG "waiting for $name on :$port (${timeout}s)"
    while (( SECONDS < deadline )); do
        (echo > /dev/tcp/localhost/"$port") 2>/dev/null && {
            LOG "$name is up on :$port"; return 0
        }
        sleep 1
    done
    ERR "$name did not bind :$port within ${timeout}s"
    [ -f "/tmp/${name}.log" ] && tail -20 "/tmp/${name}.log" >&2
    return 1
}

# ── script auto-update ────────────────────────────────────────────────────────
update_scripts(){
    local base="https://raw.githubusercontent.com/klutzydrummer/vastai-llmrunner/${REPO_REF:-main}"
    local f failed=0
    # raw.githubusercontent.com sends Cache-Control: max-age=300, so a container
    # started just after a push can otherwise boot the previous copy. A unique
    # query string plus no-cache skips the edge cache.
    local cb; cb=$(date +%s%N)
    for f in serve.py dlverify.py cfginit.py cfgedit.py guard.py embed.py imgserve.py models.json; do
        if curl -fsSL --max-time 30 -H 'Cache-Control: no-cache' "$base/$f?cb=$cb" -o "/tmp/$f"; then
            LOG "updated $f"
        else
            LOG "warn: could not update $f (keeping existing)"
            failed=$(( failed + 1 ))
        fi
    done
    return "$failed"
}

# ── pip deps ──────────────────────────────────────────────────────────────────
install_pip_deps(){
    LOG "installing websockets"
    pip install -q websockets --break-system-packages
    if [ "${DOWNLOADER:-aria2c}" = "hf" ]; then
        LOG "installing huggingface_hub"
        pip install -q huggingface_hub --break-system-packages
        if [ "${HF_BACKEND:-hf_xet}" = "hf_transfer" ]; then
            pip install -q hf_transfer --break-system-packages
        else
            pip install -q hf_xet --break-system-packages
        fi
    fi
}

# ── ffmpeg (required by llama.cpp's mtmd video input; it shells out to ffmpeg
#    to decode video frames) ────────────────────────────────────────────────────
install_ffmpeg(){
    if command -v ffmpeg >/dev/null 2>&1; then
        LOG "ffmpeg already installed"
        return 0
    fi
    LOG "installing ffmpeg"
    apt-get update -qq && apt-get install -y -qq ffmpeg \
        || die "ffmpeg install failed"
    command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not in PATH after install"
    LOG "ffmpeg installed OK"
}

# ── binary installer ──────────────────────────────────────────────────────────
# get_latest_tag REPO  →  prints tag (e.g. v211, 2026.3.0)
get_latest_tag(){
    local repo="$1"
    local url tag json

    # Strategy 1: follow redirect, validate result looks like a version
    url=$(curl -sfL --max-time 30 -o /dev/null -w '%{url_effective}' \
          "https://github.com/${repo}/releases/latest" 2>/dev/null) || true
    tag="${url##*/}"
    [[ "$tag" =~ ^v?[0-9] ]] && { echo "$tag"; return 0; }

    # Strategy 2: GitHub JSON API
    LOG "get_latest_tag: redirect failed for $repo, trying JSON API"
    json=$(curl -sfL --max-time 30 \
          "https://api.github.com/repos/${repo}/releases/latest" 2>/dev/null) || true
    tag=$(printf '%s' "$json" | grep -o '"tag_name":"[^"]*"' \
          | sed 's/"tag_name":"//;s/"//') || true
    [[ "${tag:-}" =~ ^v?[0-9] ]] && { echo "$tag"; return 0; }

    ERR "get_latest_tag: both strategies failed for $repo"
    return 1
}

# runs NAME  — false if the binary cannot be executed at all (wrong arch,
# corrupt file). Exit 126/127 come from the shell; any other status means the
# binary ran, whatever it thought of --version.
runs(){
    local rc=0
    timeout 10 "$1" --version >/dev/null 2>&1 || rc=$?
    (( rc != 126 && rc != 127 ))
}

# download_bin NAME REPO TAG_TO_VER_SED ASSET_TPL IS_TAR [TAG]
#   NAME         binary name in PATH
#   REPO         e.g. mostlygeek/llama-swap
#   TAG_TO_VER   sed expression to strip prefix (e.g. 's/^v//')
#   ASSET_TPL    asset filename template; TAG, VER and ARCH are substituted
#   IS_TAR       true|false
#   TAG          exact release tag; empty or "latest" resolves the newest
download_bin(){
    local name="$1"
    local repo="$2"
    local tag_strip="$3"
    local asset_tpl="$4"
    local is_tar="${5:-true}"
    local pin="${6:-}"

    if command -v "$name" >/dev/null 2>&1; then
        if runs "$name"; then
            LOG "$name already installed"
            return 0
        fi
        LOG "$name is installed but will not run on $ARCH, reinstalling"
        rm -f "$(command -v "$name")"
    fi

    local tag ver asset url tmp archive
    if [ -n "$pin" ] && [ "$pin" != latest ]; then
        tag="$pin"
        LOG "using pinned release $tag for $repo"
    else
        LOG "resolving latest release for $repo"
        tag=$(retry 3 5 get_latest_tag "$repo") \
            || die "could not resolve latest tag for $repo after retries"
    fi
    ver=$(echo "$tag" | sed "$tag_strip")
    asset=$(echo "$asset_tpl" | sed "s/TAG/$tag/g; s/VER/$ver/g; s/ARCH/$ARCH/g")
    url="https://github.com/${repo}/releases/download/${tag}/${asset}"

    LOG "downloading $name  tag=$tag  asset=$asset"
    tmp=$(mktemp -d)

    if [ "$is_tar" = "true" ]; then
        archive="$tmp/archive.tar.gz"
        curl -fsSL --max-time 300 --retry 3 --retry-delay 5 -o "$archive" "$url" \
            || { rm -rf "$tmp"; die "download failed for $name ($url)"; }
        tar xz -C "$tmp" -f "$archive" \
            || { rm -rf "$tmp"; die "extract failed for $name"; }
        [ -f "$tmp/$name" ] \
            || die "$name not found in archive (got: $(ls "$tmp"))"
        mv "$tmp/$name" /usr/local/bin/"$name"
    else
        curl -fsSL --max-time 120 --retry 3 --retry-delay 5 -o "$tmp/$name" "$url" \
            || { rm -rf "$tmp"; die "download failed for $name ($url)"; }
        mv "$tmp/$name" /usr/local/bin/"$name"
    fi

    chmod +x /usr/local/bin/"$name"
    rm -rf "$tmp"
    command -v "$name" >/dev/null 2>&1 || die "$name not in PATH after install"
    runs "$name" || die "$name was installed but will not run on $ARCH ($url)"
    LOG "$name installed OK"
}

# ── sd.cpp sd-server (image/video generation) ─────────────────────────────────
# Only when IMAGE_MODELS is set. sd.cpp publishes no Linux CUDA release binary,
# so we take the binaries layer of the official CUDA image, ghcr.io/leejet/
# stable-diffusion.cpp:master-cuda, straight from the registry (no docker).
# Pinned (2026-11-25): layer blob sha256:8e9cc7f2... = `COPY /sd.cpp/build/bin`,
# from git rev 52491bd9f1fc15281791723bdff13baab7fd6819, amd64 manifest
# sha256:dd56cb1a03b79ec1e74672b56cee8167f113fce89b48398f868cb57225338a9d.
# The blob is content-addressed and verified by sha256. Fallback mirror: a
# tarball of the same bin/ dir (e.g. a release asset of this repo) via
# SD_LAYER_URL, with SD_LAYER_DIGEST naming the sha256 of that file.
# Native CUDA kernels: sm_86 and sm_89 only; other GPUs JIT from PTX.
SD_DIR=/opt/sd
SD_LAYER_DIGEST="${SD_LAYER_DIGEST:-sha256:8e9cc7f29be6542ef8b2ff781e6e2d40f29097da5f9d55705d2ca5bf8b9e236b}"
SD_GHCR_REPO=leejet/stable-diffusion.cpp

_sd_fetch(){
    local want="${SD_LAYER_DIGEST#sha256:}" tmp="$1" url="${SD_LAYER_URL:-}" tok
    if [ -z "$url" ]; then
        tok=$(curl -fsS --max-time 30 "https://ghcr.io/token?scope=repository:${SD_GHCR_REPO}:pull" \
              | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])') || return 1
        curl -fL --retry 5 --retry-delay 5 --max-time 1800 -H "Authorization: Bearer $tok" \
            "https://ghcr.io/v2/${SD_GHCR_REPO}/blobs/${SD_LAYER_DIGEST}" -o "$tmp/layer.tgz" || return 1
    else
        curl -fL --retry 5 --retry-delay 5 --max-time 1800 "$url" -o "$tmp/layer.tgz" || return 1
    fi
    echo "$want  $tmp/layer.tgz" | sha256sum -c - >/dev/null || { ERR "sd.cpp archive sha256 mismatch"; return 1; }
    tar -xzf "$tmp/layer.tgz" -C "$SD_DIR" --strip-components=2 sd.cpp/bin
}

install_sd(){
    local stamp="$SD_DIR/.layer-digest" tmp
    if [ -x "$SD_DIR/sd-server" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$SD_LAYER_DIGEST" ]; then
        LOG "sd-server already installed ($SD_LAYER_DIGEST)"
    else
        [ "$ARCH" = amd64 ] || { ERR "sd.cpp binaries are amd64 only; image generation disabled"; return 1; }
        mkdir -p "$SD_DIR" /models
        # ~350 MB download, ~1.3 GB extracted
        local free_mb; free_mb=$(df -Pm "$SD_DIR" | awk 'NR==2{print $4}')
        if [ "${free_mb:-0}" -lt 1800 ]; then
            ERR "only ${free_mb:-?} MB free for sd.cpp (need ~1800); image generation disabled"; return 1
        fi
        tmp=$(mktemp -d)
        LOG "installing sd-server from $SD_LAYER_DIGEST"
        if ! _sd_fetch "$tmp"; then
            rm -rf "$tmp"; ERR "sd-server download/extract failed; image generation disabled"; return 1
        fi
        rm -rf "$tmp"
        [ -x "$SD_DIR/sd-server" ] || { ERR "sd-server not found after extract"; return 1; }
        echo "$SD_LAYER_DIGEST" > "$stamp"
        LOG "sd-server installed OK"
    fi
    # Runtime libs ggml-cuda needs that the llama.cpp image may lack (best effort).
    local pkgs=""
    ldconfig -p 2>/dev/null | grep -q 'libnccl.so.2' || pkgs="$pkgs libnccl2"
    ldconfig -p 2>/dev/null | grep -q 'libgomp.so.1' || pkgs="$pkgs libgomp1"
    if [ -n "$pkgs" ]; then
        LOG "installing sd.cpp runtime libs:$pkgs"
        apt-get update -qq && apt-get install -y -qq $pkgs \
            || ERR "could not install$pkgs; image generation may fail to start"
    fi
    return 0
}

# ── config generation ─────────────────────────────────────────────────────────
gen_config(){
    LOG "generating /app/config.yaml"
    python3 /tmp/cfginit.py || die "cfginit.py failed"
    LOG "config.yaml written ($(wc -l < /app/config.yaml) lines)"
}

# ── embeddings sidecar ────────────────────────────────────────────────────────
# embed.py runs its own llama-server outside llama-swap, so embeddings stay up
# while chat models are swapped. It self-configures from /app/params.json (the
# cfgedit UI) and falls back to EMBED_MODEL_URL; it exits quietly if unset.
start_embed(){
    LOG "starting embeddings sidecar"
    python3 /tmp/embed.py > /tmp/embed.log 2>&1 &
    EMBED_PID=$!
}

# ── service startup ───────────────────────────────────────────────────────────
start_services(){
    LOG "starting cfgedit"
    python3 /tmp/cfgedit.py > /tmp/cfgedit.log 2>&1 &
    CFGEDIT_PID=$!
    wait_port cfgedit 5005 30 || die "cfgedit failed to bind port 5005"

    LOG "starting guard"
    python3 /tmp/guard.py > /tmp/guard.log 2>&1 &
    GUARD_PID=$!
    wait_port guard 8081 30 || die "guard failed to bind port 8081"

    LOG "starting llama-swap"
    llama-swap --config /app/config.yaml --listen 0.0.0.0:8080 --watch-config \
        > /tmp/llama-swap.log 2>&1 &
    LLAMA_SWAP_PID=$!

    LOG "writing Caddyfile"
    cat > /tmp/Caddyfile << 'CADDY'
:5000 {
  handle /terminal* {
    reverse_proxy localhost:5006
  }
  handle /editor* {
    uri strip_prefix /editor
    reverse_proxy localhost:5005
  }
  # Embeddings sidecar. /v1/embeddings is the OpenAI-compatible form (used by
  # SillyTavern's "vllm"/OpenAI-compatible vector sources); /embedding is
  # llama.cpp's native form (used by SillyTavern's "llamacpp" vector source).
  handle /v1/embeddings* {
    reverse_proxy localhost:8090
  }
  handle /embedding* {
    reverse_proxy localhost:8090
  }
  handle /ui* {
    reverse_proxy localhost:8080
  }
  handle /logs* {
    reverse_proxy localhost:8080
  }
  handle {
    reverse_proxy localhost:8081
  }
}
CADDY

    LOG "starting caddy"
    caddy run --config /tmp/Caddyfile > /tmp/caddy.log 2>&1 &
    CADDY_PID=$!
    wait_port caddy 5000 30 || die "caddy failed to bind port 5000"
}

# ── health wait ───────────────────────────────────────────────────────────────
wait_healthy(){
    LOG "waiting for llama-swap"
    local deadline=$(( SECONDS + 120 ))
    while (( SECONDS < deadline )); do
        if curl -sf localhost:8080/health >/dev/null 2>&1; then
            LOG "llama-swap healthy"
            return 0
        fi
        if ! kill -0 "$LLAMA_SWAP_PID" 2>/dev/null; then
            ERR "llama-swap process died early"
            tail -20 /tmp/llama-swap.log >&2
            die "llama-swap failed to start"
        fi
        sleep 3
    done
    ERR "llama-swap not healthy after 120s — last log:"
    tail -20 /tmp/llama-swap.log >&2
    die "llama-swap health timeout"
}

# ── model preload ─────────────────────────────────────────────────────────────
preload_model(){
    [ -z "${DEFAULT_MODEL:-}" ] && return 0
    LOG "preloading $DEFAULT_MODEL (background)"
    {
        sleep 15
        LOG "sending preload request for $DEFAULT_MODEL"
        curl -sm 7200 localhost:8080/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${DEFAULT_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            -o /tmp/preload.json \
        && LOG "preloaded $DEFAULT_MODEL" \
        || ERR "preload request failed"
    } &
}

# ── tunnel ────────────────────────────────────────────────────────────────────
_launch_cloudflared(){
    if [ -n "${CLOUDFLARE_TOKEN:-}" ]; then
        LOG "starting cloudflared tunnel (named)"
        cloudflared tunnel run --token "$CLOUDFLARE_TOKEN" >> /tmp/cloudflared.log 2>&1 &
    else
        LOG "starting cloudflared tunnel (quick)"
        cloudflared tunnel --url http://localhost:5000 >> /tmp/cloudflared.log 2>&1 &
    fi
    TUNNEL_PID=$!
}

tunnel_watchdog(){
    local max_restarts=10 restarts=0
    _launch_cloudflared
    LOG "cloudflared started (pid=$TUNNEL_PID)"

    while kill -0 "$LLAMA_SWAP_PID" 2>/dev/null; do
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            restarts=$(( restarts + 1 ))
            if (( restarts > max_restarts )); then
                ERR "cloudflared died $max_restarts times; continuing without tunnel"
                break
            fi
            ERR "cloudflared exited unexpectedly (restart $restarts/$max_restarts)"
            tail -5 /tmp/cloudflared.log >&2
            sleep 5
            _launch_cloudflared
            LOG "cloudflared restarted (pid=$TUNNEL_PID)"
        fi
        sleep 5
    done

    ERR "llama-swap (pid=$LLAMA_SWAP_PID) has exited — shutting down"
    tail -20 /tmp/llama-swap.log >&2
    exit 1
}

# ── main ──────────────────────────────────────────────────────────────────────
main(){
    LOG "=== init start ==="
    mkdir -p /app /models

    update_scripts || LOG "warn: some scripts failed to update, continuing with existing versions"

    install_pip_deps
    install_ffmpeg

    # llama-swap: tag=v211 → ver=211 → llama-swap_211_linux_ARCH.tar.gz
    download_bin llama-swap \
        mostlygeek/llama-swap \
        's/^v//' \
        'llama-swap_VER_linux_ARCH.tar.gz' \
        true \
        "$LLAMA_SWAP_TAG"
    local swver
    if swver=$(timeout 10 llama-swap --version 2>&1 | head -1) && [ -n "$swver" ]; then
        LOG "llama-swap $swver (pin $LLAMA_SWAP_TAG)"
    else
        ERR "warn: llama-swap --version failed or printed nothing (pin $LLAMA_SWAP_TAG), continuing"
    fi

    # caddy: tag=v2.11.2 → ver=2.11.2 → caddy_2.11.2_linux_ARCH.tar.gz
    download_bin caddy \
        caddyserver/caddy \
        's/^v//' \
        'caddy_VER_linux_ARCH.tar.gz' \
        true

    # cloudflared: tag=2026.3.0, asset name has no version
    download_bin cloudflared \
        cloudflare/cloudflared \
        's/.*//' \
        'cloudflared-linux-ARCH' \
        false

    if [ -n "${IMAGE_MODELS:-}" ]; then
        install_sd || ERR "sd.cpp setup failed; continuing, image models will not start"
    fi

    gen_config
    start_embed
    start_services
    wait_healthy
    preload_model
    tunnel_watchdog
}

main
