#!/usr/bin/env bash
set -euo pipefail

LOG(){ echo "[$(date '+%H:%M:%S')] [init] $*"; }
ERR(){ echo "[$(date '+%H:%M:%S')] [init] ERROR: $*" >&2; }
die(){ ERR "$*"; exit 1; }

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
    local base="https://raw.githubusercontent.com/klutzydrummer/vastai-llmrunner/main"
    local f failed=0
    for f in serve.py cfginit.py cfgedit.py guard.py; do
        if curl -fsSL --max-time 30 "$base/$f" -o "/tmp/$f"; then
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

# download_bin NAME REPO TAG_TO_VER_SED ASSET_TPL IS_TAR
#   NAME         binary name in PATH
#   REPO         e.g. mostlygeek/llama-swap
#   TAG_TO_VER   sed expression to strip prefix (e.g. 's/^v//')
#   ASSET_TPL    asset filename template; TAG and VER are substituted
#   IS_TAR       true|false
download_bin(){
    local name="$1"
    local repo="$2"
    local tag_strip="$3"
    local asset_tpl="$4"
    local is_tar="${5:-true}"

    if command -v "$name" >/dev/null 2>&1; then
        LOG "$name already installed"
        return 0
    fi

    LOG "resolving latest release for $repo"
    local tag ver asset url tmp archive
    tag=$(retry 3 5 get_latest_tag "$repo") \
        || die "could not resolve latest tag for $repo after retries"
    ver=$(echo "$tag" | sed "$tag_strip")
    asset=$(echo "$asset_tpl" | sed "s/TAG/$tag/g; s/VER/$ver/g")
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
    LOG "$name installed OK"
}

# ── config generation ─────────────────────────────────────────────────────────
gen_config(){
    LOG "generating /app/config.yaml"
    python3 /tmp/cfginit.py || die "cfginit.py failed"
    LOG "config.yaml written ($(wc -l < /app/config.yaml) lines)"
}

# ── optional embed model ──────────────────────────────────────────────────────
start_embed(){
    [ -z "${EMBED_MODEL_URL:-}" ] && return 0
    LOG "starting embed model"
    local name
    name=$(printf '%s' "$EMBED_MODEL_URL" | md5sum | cut -c1-8)_$(basename "$EMBED_MODEL_URL")
    aria2c -x16 -s16 --file-allocation=none -d /models -o "$name" \
        ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} \
        "$EMBED_MODEL_URL" \
    && llama-server --model /models/"$name" --port 8090 --embedding \
        -ngl 0 -c 8192 -b 8192 --rope-scaling yarn --rope-freq-scale .75 \
        >/tmp/embed.log 2>&1 &
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
  handle /v1/embeddings* {
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

    # llama-swap: tag=v211 → ver=211 → llama-swap_211_linux_amd64.tar.gz
    download_bin llama-swap \
        mostlygeek/llama-swap \
        's/^v//' \
        'llama-swap_VER_linux_amd64.tar.gz' \
        true

    # caddy: tag=v2.11.2 → ver=2.11.2 → caddy_2.11.2_linux_amd64.tar.gz
    download_bin caddy \
        caddyserver/caddy \
        's/^v//' \
        'caddy_VER_linux_amd64.tar.gz' \
        true

    # cloudflared: tag=2026.3.0, asset name has no version
    download_bin cloudflared \
        cloudflare/cloudflared \
        's/.*//' \
        'cloudflared-linux-amd64' \
        false

    gen_config
    start_embed
    start_services
    wait_healthy
    preload_model
    tunnel_watchdog
}

main
