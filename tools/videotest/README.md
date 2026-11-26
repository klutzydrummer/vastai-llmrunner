# Video upload test page

A small page for sending a video to an OpenAI-compatible chat endpoint (for example a llama-swap instance serving LFM3-Omni) and reading the reply. Python standard library only.

llama.cpp has no native video input, so the page extracts frames and audio in the browser:

- N frames (default 8, max edge 512 px) become `image_url` JPEG data URIs.
- The audio track is decoded with WebAudio and resampled to a 16 kHz mono WAV, sent as `input_audio`. If decoding fails or there is no audio, only frames are sent and the status says so.
- The "raw video" checkbox instead sends the whole file as a `video_url` data URI, for servers that accept it (for example Qwen on Vast).

## Run

```
python3 tools/videotest/server.py          # listens on 0.0.0.0:18810 (VIDEOTEST_PORT to change)
```

Open `http://<host>:18810/` on a desktop or phone. The same-origin proxy (`POST /proxy`, `GET /models`) forwards to the endpoint you pick, so there is no CORS setup. Uploads up to 200 MB are streamed, with a 300 s upstream timeout.

## Endpoints

`endpoints.json` maps a name to a base URL. `home` is `http://127.0.0.1:18800/v1`. An entry with `"base": null` (`custom`) takes a base URL typed in the page; only http and https are accepted.

API keys are never stored in files. Set `VIDEOTEST_KEY_<NAME>` (name upper-cased) in the server's environment, for example `VIDEOTEST_KEY_HOME`. For `custom`, the key field in the page is sent in a header and kept in memory only.

Redirects are not followed and request bodies are never logged. The server has no authentication of its own, so run it on a trusted network only.

## Container deployment (Vulkan, AMD)

Two containers on the `homelab` Docker network. llama-swap publishes no host port; only the page is meant to be reached through a reverse proxy.

- `videotest-llamaswap`: `ghcr.io/mostlygeek/llama-swap:unified-vulkan` (includes a Vulkan llama-server). Config: `llama-swap.container.yaml`, mounted at `/etc/llama-swap/config/config.yaml`. Models are mounted read-only at `/models/base` and `/models/rp`. GPU: `--device /dev/dri --group-add <render gid>`.
- `videotest`: the page, built from `Dockerfile` (python alpine, stdlib only). `VIDEOTEST_BASE_HOME` overrides the `home` preset base URL; the image defaults it to `http://videotest-llamaswap:8080/v1`.

```
docker build -t videotest:local tools/videotest
docker run -d --name videotest-llamaswap --network homelab --device /dev/dri --group-add 303 \
  -v <base-models>:/models/base:ro -v <rp-models>:/models/rp:ro \
  -v $PWD/tools/videotest/llama-swap.container.yaml:/etc/llama-swap/config/config.yaml:ro \
  ghcr.io/mostlygeek/llama-swap:unified-vulkan
docker run -d --name videotest --network homelab videotest:local
```

The page is an open forwarder with no authentication: expose it on the LAN only (a `*.home.internal` route, no public router).
