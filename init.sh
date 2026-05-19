#!/usr/bin/env bash
set -euo pipefail

LOG(){ echo "[$(date '+%H:%M:%S')] [init] $*"; }
ERR(){ echo "[$(date '+%H:%M:%S')] [init] ERROR: $*" >&2; }
die(){ ERR "$*"; exit 1; }

# ── pip deps ──────────────────────────────────────────────────────────────────
install_pip_deps(){
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
    local loc tag
    loc=$(curl -sfI --connect-timeout 15 --max-time 30 \
          "https://github.com/${repo}/releases/latest" \
          | grep -i '^location:' | tr -d '\r' | sed 's/.*location: //')
    tag="${loc##*/}"
    [ -n "$tag" ] || return 1
    echo "$tag"
}

# download_bin NAME REPO TAG_TO_VER_FN ASSET_FN IS_TAR
#   NAME         binary name in PATH
#   REPO         e.g. mostlygeek/llama-swap
#   TAG_TO_VER   sed expression to strip prefix (e.g. 's/^v//')
#   ASSET_TPL    asset filename template; TAG and VER are substituted
#   IS_TAR       true|false
download_bin(){
    local name="$1"
    local repo="$2"
    local tag_strip="$3"    # sed expr: 's/^v//'
    local asset_tpl="$4"    # e.g. 'llama-swap_VER_linux_amd64.tar.gz'
    local is_tar="${5:-true}"

    if command -v "$name" >/dev/null 2>&1; then
        LOG "$name already installed"
        return 0
    fi

    LOG "resolving latest release for $repo"
    local tag ver asset url tmp
    tag=$(get_latest_tag "$repo") || die "could not resolve latest tag for $repo"
    ver=$(echo "$tag" | sed "$tag_strip")
    asset=$(echo "$asset_tpl" | sed "s/TAG/$tag/g; s/VER/$ver/g")
    url="https://github.com/${repo}/releases/download/${tag}/${asset}"

    LOG "downloading $name  tag=$tag  asset=$asset"
    tmp=$(mktemp -d)

    if [ "$is_tar" = "true" ]; then
        curl -fsSL --connect-timeout 15 --speed-limit 1 --speed-time 60 \
             --retry 3 --retry-delay 2 "$url" | tar xz -C "$tmp" \
            || { rm -rf "$tmp"; die "download/extract failed for $name ($url)"; }
        [ -f "$tmp/$name" ] \
            || die "$name not found in archive (got: $(ls "$tmp"))"
        mv "$tmp/$name" /usr/local/bin/"$name"
    else
        curl -fsSL --connect-timeout 15 --speed-limit 1 --speed-time 60 \
             --retry 3 --retry-delay 2 -o "$tmp/$name" "$url" \
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

    LOG "starting guard"
    python3 /tmp/guard.py > /tmp/guard.log 2>&1 &
    GUARD_PID=$!

    LOG "starting llama-swap"
    llama-swap --config /app/config.yaml --listen 0.0.0.0:8080 --watch-config \
        > /tmp/llama-swap.log 2>&1 &
    LLAMA_SWAP_PID=$!

    LOG "writing Caddyfile"
    cat > /tmp/Caddyfile << 'CADDY'
:5000 {
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
start_tunnel(){
    if [ -n "${CLOUDFLARE_TOKEN:-}" ]; then
        LOG "starting cloudflared tunnel (named)"
        cloudflared tunnel run --token "$CLOUDFLARE_TOKEN" 2>&1 | tee /tmp/cloudflared.log &
    else
        LOG "starting cloudflared tunnel (quick)"
        cloudflared tunnel --url http://localhost:5000 2>&1 | tee /tmp/cloudflared.log &
    fi
    TUNNEL_PID=$!
    wait $TUNNEL_PID
}

# ── main ──────────────────────────────────────────────────────────────────────
main(){
    LOG "=== init start ==="
    mkdir -p /app /models

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
    start_tunnel
}

main