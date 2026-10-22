// Pip front-end: captures audio (mic, MELD clip or file), streams 16 kHz int16 PCM over a
// WebSocket in 100 ms chunks, and renders the server's events.
const EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"];
const SR = 16000, CHUNK = 1600;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();

let ws = null, micCtx = null, micStream = null, micNode = null, busyStreaming = false;
let truthForNext = null;           // ground-truth label of a streamed MELD clip
const turns = new Map();           // utterance_id -> {el, truth}
const rawLog = [];

// ------------------------------------------------------------------ websocket
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); setTimeout(connect, 1500); };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
}
function setConn(on) {
  $("connDot").className = "dot " + (on ? "on" : "off");
  $("connText").textContent = on ? "connected (local)" : "disconnected, retrying…";
}
const sendPCM = (int16) => { if (ws && ws.readyState === 1) ws.send(int16.buffer); };
const sendCmd = (type) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type })); };

// ------------------------------------------------------------------ events
function handle(ev) {
  if (ev.type !== "response_delta") {
    rawLog.unshift(ev);
    rawLog.length = Math.min(rawLog.length, 40);
    $("rawEvents").textContent = rawLog.map((e) => JSON.stringify(e)).join("\n\n");
  }
  switch (ev.type) {
    case "ready":
      $("engineInfo").textContent = JSON.stringify(ev.engine, null, 2);
      break;
    case "vad":
      if (ev.state === "speech_start") setBadge("speaking", "speaking");
      else if (ev.state === "speech_end") setBadge("thinking", "thinking");
      else setBadge("idle", "silence");
      break;
    case "partial_state":
      showLive(ev, true);
      break;
    case "final_state":
      showLive(ev, false);
      addTurn(ev);
      break;
    case "response_delta":
      appendReply(ev.utterance_id, ev.text);
      break;
    case "response_done":
      finishTurn(ev);
      setBadge("idle", "silence");
      break;
    case "error":
      console.warn("server error", ev);
      break;
  }
}

function setBadge(cls, text) { const b = $("vadBadge"); b.className = "badge " + cls; b.textContent = text; }

function renderBars(probs) {
  $("bars").innerHTML = EMOTIONS.map((e) => {
    const p = probs ? probs[e] || 0 : 0;
    return `<div class="bar"><span>${e}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%;background:${css(e)}"></div></div><span class="val">${(p * 100).toFixed(0)}%</span></div>`;
  }).join("");
}

function showLive(ev, provisional) {
  const el = $("liveEmotion");
  el.textContent = ev.emotion;
  el.style.color = css(ev.emotion);
  el.className = "live-emotion" + (provisional ? " provisional" : "");
  $("liveMeta").textContent = provisional
    ? `provisional · last ${ev.window_s} s of speech · ${ev.latency_ms.emotion} ms`
    : `final · utterance #${ev.utterance_id} (${ev.duration_s} s) · state ${ev.latency_ms.end_of_speech_to_state} ms after speech ended`;
  renderBars(ev.probs);
}

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function addTurn(ev) {
  const empty = $("turns").querySelector(".empty-state");
  if (empty) empty.remove();
  const truth = truthForNext;
  const el = document.createElement("div");
  el.className = "turn";
  el.style.borderLeftColor = css(ev.emotion);
  const truthHtml = truth ? `<span class="truth ${truth.emotion === ev.emotion ? "ok" : "no"}">MELD label: ${truth.emotion} ${truth.emotion === ev.emotion ? "✓" : "✗"}</span>` : "";
  el.innerHTML = `
    <div class="turn-top">
      <div><span class="turn-emo" style="color:${css(ev.emotion)}">${ev.emotion}</span>
        <span class="meta"> ${(ev.confidence * 100).toFixed(0)}% · #${ev.utterance_id} · ${ev.duration_s} s</span></div>
      ${truthHtml}
    </div>
    <div class="said ${ev.transcript ? "" : "empty"}">${ev.transcript ? "“" + esc(ev.transcript) + "”" : "(no words recognised)"}</div>
    <div class="chips">${ev.tags.map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div>
    <div class="reply"><b>Pip:</b> <span class="text"></span><span class="cursor"></span></div>
    <div class="lat meta">state +${ev.latency_ms.end_of_speech_to_state} ms (emotion ${ev.latency_ms.emotion} ms ∥ whisper ${ev.latency_ms.asr} ms)<span class="more"></span></div>`;
  $("turns").prepend(el);
  turns.set(ev.utterance_id, { el, truth });
}

function appendReply(id, text) {
  const t = turns.get(id);
  if (t) t.el.querySelector(".reply .text").textContent += text;
}

function finishTurn(ev) {
  const t = turns.get(ev.utterance_id);
  if (!t) return;
  t.el.querySelector(".cursor")?.remove();
  t.el.querySelector(".reply .text").textContent = ev.text;
  const l = ev.latency_ms;
  t.el.querySelector(".more").textContent =
    ` · first word +${l.end_of_speech_to_first_token} ms · done +${l.end_of_speech_to_last_token} ms (${ev.tokens_per_s} tok/s)`;
}

// ------------------------------------------------------------------ microphone
async function toggleMic() {
  if (micCtx) return stopMic();
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
  } catch (e) {
    alert("Microphone permission denied: " + e.message);
    return;
  }
  micCtx = new AudioContext();
  await micCtx.audioWorklet.addModule("/static/pcm-worklet.js");
  const src = micCtx.createMediaStreamSource(micStream);
  const lp = micCtx.createBiquadFilter();
  lp.type = "lowpass"; lp.frequency.value = 7000;
  micNode = new AudioWorkletNode(micCtx, "pcm-capture", { processorOptions: { targetRate: SR, chunkSamples: CHUNK } });
  micNode.port.onmessage = (e) => sendPCM(new Int16Array(e.data));
  const sink = micCtx.createGain(); sink.gain.value = 0;   // keep the graph pulling without audible output
  src.connect(lp).connect(micNode).connect(sink).connect(micCtx.destination);
  truthForNext = null;
  $("micBtn").textContent = "■ Stop microphone";
  $("micBtn").classList.add("recording");
  setStreamingButtons(true);
}

function stopMic() {
  sendCmd("flush");
  micStream?.getTracks().forEach((t) => t.stop());
  micCtx?.close();
  micCtx = micStream = micNode = null;
  $("micBtn").textContent = "● Start microphone";
  $("micBtn").classList.remove("recording");
  setStreamingButtons(false);
}

function setStreamingButtons(active) {
  busyStreaming = active;
  $("clipBtn").disabled = active;
  $("fileInput").disabled = active;
}

// ------------------------------------------------------------------ clip / file replay
async function streamArrayBuffer(buf, truth) {
  if (busyStreaming) return;
  setStreamingButtons(true);
  $("micBtn").disabled = true;
  try {
    const ctx = new AudioContext();
    const decoded = await ctx.decodeAudioData(buf);
    const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * SR), SR);  // resample + downmix
    const s = off.createBufferSource(); s.buffer = decoded; s.connect(off.destination); s.start();
    const pcm = (await off.startRendering()).getChannelData(0);
    const play = ctx.createBufferSource(); play.buffer = decoded; play.connect(ctx.destination);
    truthForNext = truth;
    const tail = new Float32Array(Math.round(0.8 * SR));          // silence lets the VAD close the turn
    const all = new Float32Array(pcm.length + tail.length); all.set(pcm);
    const t0 = performance.now();
    play.start();
    for (let i = 0; i < all.length; i += CHUNK) {
      const f = all.subarray(i, i + CHUNK);
      const out = new Int16Array(f.length);
      for (let j = 0; j < f.length; j++) { const v = Math.max(-1, Math.min(1, f[j])); out[j] = v < 0 ? v * 0x8000 : v * 0x7fff; }
      sendPCM(out);
      const wait = t0 + ((i + CHUNK) / SR) * 1000 - performance.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }
    sendCmd("flush");
    setTimeout(() => ctx.close(), 500);
  } catch (e) {
    alert("Could not decode that audio: " + e.message);
  } finally {
    setStreamingButtons(false);
    $("micBtn").disabled = false;
  }
}

async function loadClips() {
  try {
    const r = await (await fetch("/api/clips")).json();
    const sel = $("clipSelect");
    if (!r.clips.length) { sel.options[0].textContent = "no MELD clips (data not prepared)"; return; }
    for (const c of r.clips) {
      const o = document.createElement("option");
      o.value = JSON.stringify(c);
      const text = c.utterance.length > 48 ? c.utterance.slice(0, 46) + "…" : c.utterance;
      o.textContent = `[${c.emotion}] ${c.speaker}: ${text} (${c.duration_s}s)`;
      sel.appendChild(o);
    }
  } catch (e) { console.warn(e); }
}

$("micBtn").onclick = toggleMic;
$("clipBtn").onclick = async () => {
  const v = $("clipSelect").value;
  if (!v) return;
  const c = JSON.parse(v);
  const buf = await (await fetch(c.url)).arrayBuffer();
  streamArrayBuffer(buf, { emotion: c.emotion, id: c.id });
};
$("fileInput").onchange = async (e) => {
  const f = e.target.files[0];
  if (f) streamArrayBuffer(await f.arrayBuffer(), null);
  e.target.value = "";
};
$("resetBtn").onclick = () => {
  sendCmd("reset");
  turns.clear();
  $("turns").innerHTML = `<div class="empty-state">Conversation cleared.</div>`;
};

$("turns").innerHTML = `<div class="empty-state">Nothing yet. Start the microphone or stream a MELD clip.</div>`;
renderBars(null);
connect();
loadClips();
