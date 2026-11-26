"use strict";
// Pure helpers (exported for node tests) + page wiring (only when a DOM exists).

function floatToWav16(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buf);
}

function bytesToB64(bytes) {
  let out = "";
  for (let i = 0; i < bytes.length; i += 0x8000)
    out += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(out);
}

function frameTimes(duration, n) {  // midpoints of n equal slices
  const t = [];
  for (let i = 0; i < n; i++) t.push(duration * (i + 0.5) / n);
  return t;
}

function fitSize(w, h, maxEdge) {
  const s = Math.min(1, maxEdge / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

function buildMessage(prompt, frames, audioB64, rawVideoUrl) {
  const c = [{type: "text", text: prompt}];
  if (rawVideoUrl) c.push({type: "video_url", video_url: {url: rawVideoUrl}});
  else {
    for (const f of frames) c.push({type: "image_url", image_url: {url: f}});
    if (audioB64) c.push({type: "input_audio", input_audio: {data: audioB64, format: "wav"}});
  }
  return {role: "user", content: c};
}

async function extractFrames(file, n, maxEdge) {
  const url = URL.createObjectURL(file);
  const vid = document.createElement("video");
  vid.muted = true; vid.playsInline = true; vid.preload = "auto"; vid.src = url;
  await new Promise((ok, no) => { vid.onloadedmetadata = ok; vid.onerror = () => no(new Error("cannot decode video in this browser")); });
  const [w, h] = fitSize(vid.videoWidth, vid.videoHeight, maxEdge);
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d"), out = [];
  for (const t of frameTimes(vid.duration, n)) {
    await new Promise(ok => { vid.onseeked = ok; vid.currentTime = t; });
    ctx.drawImage(vid, 0, 0, w, h);
    out.push(cv.toDataURL("image/jpeg", 0.8));
  }
  URL.revokeObjectURL(url);
  return out;
}

async function extractAudioWav(file) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  try {
    const dec = await ctx.decodeAudioData(await file.arrayBuffer());
    const len = Math.max(1, Math.ceil(dec.duration * 16000));
    const off = new OfflineAudioContext(1, len, 16000);  // mixes down to mono
    const src = off.createBufferSource(); src.buffer = dec; src.connect(off.destination); src.start();
    const r = await off.startRendering();
    return bytesToB64(floatToWav16(r.getChannelData(0), 16000));
  } finally { ctx.close(); }
}

function fileToDataUrl(file) {
  return new Promise((ok, no) => {
    const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = () => no(r.error); r.readAsDataURL(file);
  });
}

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const PRESETS = {
    "Describe the video": "Describe what happens in this video in detail.",
    "Describe the sounds": "Describe all the sounds you hear in the audio.",
    "Transcribe speech": "Transcribe any speech in the audio exactly.",
    "Custom": "",
  };
  for (const k of Object.keys(PRESETS)) $("preset").add(new Option(k));
  $("prompt").value = PRESETS["Describe the video"];
  $("preset").onchange = () => { if ($("preset").value !== "Custom") $("prompt").value = PRESETS[$("preset").value]; };
  const q = () => "endpoint=" + encodeURIComponent($("endpoint").value) + ($("endpoint").selectedOptions[0].dataset.custom ? "&base=" + encodeURIComponent($("base").value) : "");
  const hdr = () => { const h = {}; if ($("key").value) h["X-Custom-Key"] = $("key").value; return h; };
  const status = s => { $("status").textContent = s; };

  async function loadModels() {
    $("modelSel").innerHTML = "";
    try {
      const r = await fetch("/models?" + q(), {headers: hdr()});
      const j = await r.json();
      const ids = (j.data || []).map(m => m.id);
      for (const id of ids) $("modelSel").add(new Option(id));
      if (ids.length && !$("model").value) $("model").value = ids[0];
      if (!ids.length) status("no models listed: " + JSON.stringify(j).slice(0, 300));
    } catch (e) { status("model list failed: " + e.message); }
  }
  $("modelSel").onchange = () => { $("model").value = $("modelSel").value; };
  $("refresh").onclick = loadModels;
  $("endpoint").onchange = () => { $("customBox").hidden = !$("endpoint").selectedOptions[0].dataset.custom; loadModels(); };
  $("file").onchange = () => {
    const f = $("file").files[0]; if (!f) return;
    $("preview").src = URL.createObjectURL(f); $("preview").hidden = false;
  };

  $("go").onclick = async () => {
    const file = $("file").files[0];
    if (!file) return status("choose a video first");
    $("go").disabled = true; $("reply").textContent = ""; $("frames").innerHTML = "";
    const notes = [], t0 = performance.now();
    try {
      let frames = [], audio = null, raw = null;
      if ($("raw").checked) {
        status("reading raw video..."); raw = await fileToDataUrl(file);
        notes.push("raw video mode: " + (file.size / 1e6).toFixed(1) + " MB file");
      } else {
        status("extracting frames...");
        frames = await extractFrames(file, Math.max(1, Math.min(64, +$("nframes").value || 8)), +$("edge").value);
        for (const f of frames) { const i = new Image(); i.src = f; $("frames").appendChild(i); }
        if ($("audio").checked) {
          status("decoding audio...");
          try { audio = await extractAudioWav(file); notes.push("audio: sent"); }
          catch (e) { notes.push("audio: NOT sent (decode failed or no audio track: " + e.message + ")"); }
        } else notes.push("audio: disabled");
      }
      const body = JSON.stringify({model: $("model").value, max_tokens: +$("maxtok").value || 400, temperature: 0,
        messages: [buildMessage($("prompt").value, frames, audio, raw)]});
      const t1 = performance.now();
      status("sending " + (body.length / 1e6).toFixed(2) + " MB...\n" + notes.join("\n"));
      const r = await fetch("/proxy?" + q(), {method: "POST", headers: Object.assign({"Content-Type": "application/json"}, hdr()), body});
      const j = await r.json(), t2 = performance.now();
      $("reply").textContent = j.choices ? j.choices[0].message.content : JSON.stringify(j, null, 2);
      status(["HTTP " + r.status, "frames: " + frames.length, ...notes,
        "request size: " + (body.length / 1e6).toFixed(2) + " MB",
        "prep: " + ((t1 - t0) / 1000).toFixed(1) + " s", "model round trip: " + ((t2 - t1) / 1000).toFixed(1) + " s",
        j.usage ? "usage: " + JSON.stringify(j.usage) : ""].filter(Boolean).join("\n"));
    } catch (e) { status("error: " + e.message + "\n" + notes.join("\n")); }
    $("go").disabled = false;
  };

  fetch("/endpoints").then(r => r.json()).then(eps => {
    for (const e of eps) { const o = new Option(e.name); if (e.custom) o.dataset.custom = "1"; $("endpoint").add(o); }
    loadModels();
  });
}

if (typeof module !== "undefined") module.exports = {floatToWav16, bytesToB64, frameTimes, fitSize, buildMessage};
