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
