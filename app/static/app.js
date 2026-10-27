// Pip front-end: captures audio (mic / MELD clip / file) and optionally video frames (camera / clip
// video), streams them over one WebSocket, and renders the server's events.
//   binary frames: 0x01 + int16 PCM @ 16 kHz (100 ms)   |   0x02 + JPEG (~5 per second)
const EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"];
const SR = 16000, CHUNK = 1600, FRAME_MS = 200;
// Purpose: Find a page element by its ID.
// Input: id: DOM element ID.
// Output: Matching HTMLElement, or null.
const $ = (id) => document.getElementById(id);
// Purpose: Read a theme color or other CSS custom property.
// Input: name: custom-property name without the -- prefix.
// Output: Trimmed CSS property value.
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();

let ws = null, micCtx = null, micStream = null, camStream = null, frameTimer = null, busyStreaming = false;
let truthForNext = null;           // ground-truth label of a streamed MELD clip
let inputEpoch = 0;                // stop a replay if its WebSocket session disappears
let frameEpoch = 0;                // discard JPEG encodes completed after capture stops
const truthByUtterance = new Map(); // capture the source label before inference can queue
const turns = new Map();           // utterance_id -> {el, truth}
const rawLog = [];
const grab = document.createElement("canvas");

// ------------------------------------------------------------------ websocket
// Purpose: Open the local event/media WebSocket and install reconnect handlers.
// Input: None explicitly; uses location and shared UI/session state.
// Output: Undefined; assigns ws and registers its lifecycle/message callbacks.
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen =
    /* Purpose: Mark the local connection ready when WebSocket opens.
     * Input: None; captures shared UI state.
     * Output: Undefined; updates the connection indicator.
     */
    () => setConn(true);
  ws.onclose =
    /* Purpose: Stop captured media and schedule reconnect after a dropped connection.
     * Input: None; captures current input/session/media state.
     * Output: Undefined; invalidates replay work, cleans up media, and schedules connect().
     */
    () => {
    inputEpoch++;
    setConn(false);
    if (micCtx || micStream) stopMic();
    stopCamera();
    $("mediaVideo").pause();
    stopFrames();
    setTimeout(connect, 1500);
  };
  ws.onmessage =
    /* Purpose: Decode a server message and dispatch its event.
     * Input: e: WebSocket MessageEvent containing JSON.
     * Output: Undefined; passes the parsed event to handle().
     */
    (e) => handle(JSON.parse(e.data));
}
// Purpose: Render current WebSocket connection status.
// Input: on: connection-ready boolean.
// Output: Undefined; updates status classes and text.
function setConn(on) {
  $("connDot").className = "dot " + (on ? "on" : "off");
  $("connText").textContent = on ? "connected (local)" : "disconnected, retrying…";
}
// Purpose: Prefix a binary media packet with its protocol type and transmit it.
// Input: kind: packet byte, PCM=1 or JPEG=2; bytes: ArrayBuffer or typed-array view.
// Output: Undefined; sends when the socket is open, otherwise returns without sending.
function sendTyped(kind, bytes) {
  if (!ws || ws.readyState !== 1) return;
  const buf = new Uint8Array(1 + bytes.byteLength);
  buf[0] = kind;
  buf.set(new Uint8Array(bytes.buffer || bytes, bytes.byteOffset || 0, bytes.byteLength), 1);
  ws.send(buf.buffer);
}
// Purpose: Send a mono PCM audio chunk using the audio packet type.
// Input: int16: signed 16-bit samples at 16 kHz.
// Output: Undefined; delegates to sendTyped().
const sendPCM = (int16) => sendTyped(0x01, int16);
// Purpose: Send one JSON session command if connected.
// Input: type: command name such as flush or reset.
// Output: Undefined; sends the command when the socket is open.
const sendCmd = (type) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type })); };

// ------------------------------------------------------------------ video frames
// Purpose: Sample video frames every 200 ms and send bounded-size JPEGs.
// Input: videoEl: playing HTMLVideoElement.
// Output: Undefined; replaces the frame timer and captures an epoch for stale-result rejection.
function startFrames(videoEl) {
  stopFrames();
  const epoch = frameEpoch;
  frameTimer = setInterval(
    /* Purpose: Capture one playable video frame for JPEG encoding.
     * Input: None; captures videoEl, canvas, and frame epoch.
     * Output: Undefined; resizes/draws the frame and starts asynchronous encoding.
     */
    () => {
    if (!videoEl.videoWidth || videoEl.paused || videoEl.ended) return;
    const k = Math.min(1, 480 / Math.max(videoEl.videoWidth, videoEl.videoHeight));
    grab.width = Math.round(videoEl.videoWidth * k);
    grab.height = Math.round(videoEl.videoHeight * k);
    grab.getContext("2d").drawImage(videoEl, 0, 0, grab.width, grab.height);
    grab.toBlob(
      /* Purpose: Send an encoded frame only if capture is still current.
       * Input: b: encoded JPEG Blob or null; captures frame epoch.
       * Output: Promise resolving to undefined; sends JPEG bytes unless capture stopped meanwhile.
       */
      async (b) => {
      if (!b) return;
      const bytes = new Uint8Array(await b.arrayBuffer());
      if (epoch === frameEpoch) sendTyped(0x02, bytes);
    }, "image/jpeg", 0.75);
  }, FRAME_MS);
}
// Purpose: Stop video sampling and invalidate JPEG encodes still in progress.
// Input: None explicitly; uses frame timer/epoch and overlay state.
// Output: Undefined; clears the timer and face overlay.
function stopFrames() { frameEpoch++; if (frameTimer) clearInterval(frameTimer); frameTimer = null; clearOverlay(); }

// Purpose: Render the detected face rectangle over the displayed video.
// Input: box: normalized x/y/width/height or null; emotion: theme color label.
// Output: Undefined; clears and redraws the overlay canvas.
function drawBox(box, emotion) {
  const v = $("mediaVideo"), c = $("overlay");
  c.width = v.videoWidth || 640; c.height = v.videoHeight || 480;
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  if (!box) return;
  const [x, y, w, h] = box;
  g.strokeStyle = css(emotion) || "#fff"; g.lineWidth = Math.max(2, c.width / 200);
  g.strokeRect(x * c.width, y * c.height, w * c.width, h * c.height);
}
// Purpose: Remove the displayed face rectangle.
// Input: None explicitly; uses the overlay canvas.
// Output: Undefined; clears canvas pixels.
function clearOverlay() { const c = $("overlay"); c.getContext("2d").clearRect(0, 0, c.width, c.height); }
// Purpose: Update video visibility, mirroring, and the media hint.
// Input: on: visible flag; mirror: camera mirroring flag; hint: optional hint text.
// Output: Undefined; updates media classes and hint text.
function setMedia(on, mirror, hint) {
  $("media").className = "media" + (on ? " on" : "") + (mirror ? " mirror" : "");
  $("mediaHint").textContent = hint || "camera off";
}

// ------------------------------------------------------------------ events
// Purpose: Apply one pipeline event to live state, modality panels, and turns.
// Input: ev: decoded server event dict.
// Output: Undefined; updates the appropriate UI/session state and bounded event log.
function handle(ev) {
  if (ev.type !== "response_delta" && ev.type !== "face_state") {
    rawLog.unshift(ev);
    rawLog.length = Math.min(rawLog.length, 40);
    $("rawEvents").textContent = rawLog.map(
      /* Purpose: Format an event for the raw event log.
       * Input: e: event object.
       * Output: JSON string.
       */
      (e) => JSON.stringify(e)).join("\n\n");
  }
  switch (ev.type) {
    case "ready":
      truthForNext = null;
      clearConversation("New local session. Start an input to begin.");
      $("engineInfo").textContent = JSON.stringify({ modalities: ev.modalities, ...ev.engine }, null, 2);
      $("camBtn").disabled = !ev.modalities.face;
      $("camBtn").title = ev.modalities.face ? "" : "face model not loaded on the server";
      break;
    case "vad":
      if (ev.state === "speech_start") {
        truthByUtterance.set(ev.utterance_id, truthForNext);
        setBadge("speaking", "speaking");
      }
      else if (ev.state === "speech_end") setBadge("thinking", "thinking");
      else setBadge("idle", "silence");
      break;
    case "reset":
      clearConversation("Conversation cleared.");
      break;
    case "face_state":
      setMod("face", ev.face ? ev : null, ev.face ? "" : "no face");
      drawBox(ev.face ? ev.box : null, ev.emotion);
      break;
    case "partial_state":
      showLive(ev, true);
      setMod("voice", ev.modalities.voice);
      break;
    case "final_state":
      showLive(ev, false);
      setMod("voice", ev.modalities.voice);
      setMod("words", ev.modalities.words, "no words");
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

// Purpose: Clear rendered turns, captured labels, live predictions, and event history.
// Input: message: empty-state text.
// Output: Undefined; resets conversation UI and tracking maps.
function clearConversation(message) {
  turns.clear();
  truthByUtterance.clear();
  rawLog.length = 0;
  $("rawEvents").textContent = "";
  $("turns").innerHTML = `<div class="empty-state">${esc(message)}</div>`;
  $("liveEmotion").textContent = "–";
  $("liveMeta").textContent = "Waiting for speech";
  renderBars(null);
  for (const name of ["voice", "words", "face"]) setMod(name, null);
  setBadge("idle", "silence");
}

// Purpose: Render the VAD/interaction status badge.
// Input: cls: badge state class; text: visible label.
// Output: Undefined; updates badge class and text.
function setBadge(cls, text) { const b = $("vadBadge"); b.className = "badge " + cls; b.textContent = text; }
// Purpose: Render one modality's label/confidence or missing-input hint.
// Input: name: voice/words/face panel; r: reading or null; emptyText: optional fallback.
// Output: Undefined; replaces the panel content.
function setMod(name, r, emptyText) {
  $(`mod-${name}`).innerHTML = r
    ? `<b style="color:${css(r.emotion)}">${r.emotion}</b> ${(r.confidence * 100).toFixed(0)}%` + (r.frames ? ` <span class="meta">(${r.frames} frames)</span>` : "")
    : `<span class="meta">${emptyText || "–"}</span>`;
}

// Purpose: Render seven emotion probability bars.
// Input: probs: emotion-to-probability map or null.
// Output: Undefined; replaces bar markup, using zero values when missing.
function renderBars(probs) {
  $("bars").innerHTML = EMOTIONS.map(
    /* Purpose: Build one emotion's probability bar.
     * Input: e: MELD emotion name; captures probs.
     * Output: HTML string for that label and probability.
     */
    (e) => {
    const p = probs ? probs[e] || 0 : 0;
    return `<div class="bar"><span>${e}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%;background:${css(e)}"></div></div><span class="val">${(p * 100).toFixed(0)}%</span></div>`;
  }).join("");
}

// Purpose: Render a provisional or final emotion state and its latency.
// Input: ev: state event; provisional: partial-state flag.
// Output: Undefined; updates label, metadata, and probability bars.
function showLive(ev, provisional) {
  const el = $("liveEmotion");
  el.textContent = ev.emotion;
  el.style.color = css(ev.emotion);
  el.className = "live-emotion" + (provisional ? " provisional" : "");
  $("liveMeta").textContent = provisional
    ? `provisional · voice${ev.modalities.face ? " + face" : ""} · last ${ev.window_s} s · ${ev.latency_ms.emotion} ms`
    : `final · fused ${ev.fusion.used.join(" + ")} · #${ev.utterance_id} (${ev.duration_s} s) · ${ev.latency_ms.end_of_speech_to_state} ms after speech ended`;
  renderBars(ev.probs);
}

// Purpose: Escape text before embedding it in generated HTML.
// Input: s: value to convert into display text.
// Output: String with ampersand, angle brackets, and double quotes escaped.
function esc(s) { return String(s).replace(/[&<>"]/g,
  /* Purpose: Replace one HTML-sensitive character with its entity.
   * Input: c: matched character.
   * Output: HTML entity string.
   */
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
// Purpose: Build compact markup for a turn's modality reading.
// Input: name: display label; r: reading or null.
// Output: HTML span string with emotion and optional confidence.
const modTxt = (name, r) => `<span>${name} <b style="color:${r ? css(r.emotion) : "inherit"}">${r ? r.emotion : "–"}</b>${r ? " " + (r.confidence * 100).toFixed(0) + "%" : ""}</span>`;

// Purpose: Create a completed-input turn card and preserve its captured ground truth.
// Input: ev: final_state event.
// Output: Undefined; prepends markup and records the card under its utterance ID.
function addTurn(ev) {
  const empty = $("turns").querySelector(".empty-state");
  if (empty) empty.remove();
  const truth = truthByUtterance.get(ev.utterance_id) || null;
  truthByUtterance.delete(ev.utterance_id);
  const m = ev.modalities, a = ev.agreement || { kind: "unknown" };
  const el = document.createElement("div");
  el.className = "turn";
  el.style.borderLeftColor = css(ev.emotion);
  const truthHtml = truth ? `<span class="truth ${truth.emotion === ev.emotion ? "ok" : "no"}">MELD label: ${truth.emotion} ${truth.emotion === ev.emotion ? "✓" : "✗"}</span>` : "";
  const agreeHtml = a.kind !== "unknown" ? `<span class="agree ${a.kind}" title="${esc(a.note)}">${a.kind.replace("_", " ")}</span>` : "";
  el.innerHTML = `
    <div class="turn-top">
      <div><span class="turn-emo" style="color:${css(ev.emotion)}">${ev.emotion}</span>
        <span class="meta"> ${(ev.confidence * 100).toFixed(0)}% fused · #${ev.utterance_id} · ${ev.duration_s} s</span> ${agreeHtml}</div>
      ${truthHtml}
    </div>
    <div class="modrow">${modTxt("voice", m.voice)}${modTxt("words", m.words)}${modTxt("face", m.face)}</div>
    <div class="said ${ev.transcript ? "" : "empty"}">${ev.transcript ? "“" + esc(ev.transcript) + "”" : "(no words recognised)"}</div>
    ${a.kind !== "unknown" && a.kind !== "agree" ? `<div class="meta">${esc(a.note)}</div>` : ""}
    <div class="chips">${ev.tags.map(
      /* Purpose: Build escaped display markup for one structured tag.
       * Input: t: tag text.
       * Output: HTML chip string.
       */
      (t) => `<span class="chip">${esc(t)}</span>`).join("")}</div>
    <div class="reply"><b>Pip:</b> <span class="text"></span><span class="cursor"></span></div>
    <div class="lat meta">state +${ev.latency_ms.end_of_speech_to_state} ms (voice ${ev.latency_ms.emotion} ms ∥ whisper ${ev.latency_ms.asr} ms${ev.latency_ms.text_emotion ? ` → words ${ev.latency_ms.text_emotion} ms` : ""})<span class="more"></span></div>`;
  $("turns").prepend(el);
  turns.set(ev.utterance_id, { el, truth });
}

// Purpose: Append a streamed reply delta to an existing turn.
// Input: id: utterance ID; text: visible reply delta.
// Output: Undefined; updates reply text when the turn exists.
function appendReply(id, text) {
  const t = turns.get(id);
  if (t) t.el.querySelector(".reply .text").textContent += text;
}

// Purpose: Replace streamed text with the completed reply and show final latency.
// Input: ev: response_done event.
// Output: Undefined; removes the cursor and updates reply/timing fields when the turn exists.
function finishTurn(ev) {
  const t = turns.get(ev.utterance_id);
  if (!t) return;
  t.el.querySelector(".cursor")?.remove();
  t.el.querySelector(".reply .text").textContent = ev.text;
  const l = ev.latency_ms;
  t.el.querySelector(".more").textContent =
    ` · first word +${l.end_of_speech_to_first_token} ms · done +${l.end_of_speech_to_last_token} ms (${ev.tokens_per_s} deltas/s)`;
}

// ------------------------------------------------------------------ microphone
// Purpose: Start microphone capture or stop an active microphone session.
// Input: None explicitly; uses WebSocket, capture state, and browser media APIs.
// Output: Promise resolving to undefined; sets up PCM capture/UI or reports and cleans up setup failures.
async function toggleMic() {
  if (micCtx) return stopMic();
  if (busyStreaming) return;
  if (!ws || ws.readyState !== 1) { alert("Wait for the local connection before starting the microphone."); return; }
  const epoch = inputEpoch;
  $("micBtn").disabled = true;
  setStreamingButtons(true);
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
    if (epoch !== inputEpoch) throw new Error("The local connection was interrupted; start the microphone again.");
    micCtx = new AudioContext();
    await micCtx.audioWorklet.addModule("/static/pcm-worklet.js");
    const src = micCtx.createMediaStreamSource(micStream);
    const lp = micCtx.createBiquadFilter();
    lp.type = "lowpass"; lp.frequency.value = 7000;
    const node = new AudioWorkletNode(micCtx, "pcm-capture", { processorOptions: { targetRate: SR, chunkSamples: CHUNK } });
    node.port.onmessage =
      /* Purpose: Forward a PCM worklet message to the server.
       * Input: e: MessageEvent containing a transferred Int16 buffer.
       * Output: Undefined; sends its audio samples.
       */
      (e) => sendPCM(new Int16Array(e.data));
    const sink = micCtx.createGain(); sink.gain.value = 0;   // keep the graph pulling without audible output
    src.connect(lp).connect(node).connect(sink).connect(micCtx.destination);
    truthForNext = null;
    $("micBtn").textContent = "■ Stop microphone";
    $("micBtn").classList.add("recording");
    setStreamingButtons(true);
  } catch (e) {
    stopMic();
    alert("Could not start microphone: " + e.message);
  } finally {
    $("micBtn").disabled = false;
  }
}

// Purpose: Flush pending speech and release microphone capture resources.
// Input: None explicitly; uses the microphone stream/context.
// Output: Undefined; stops tracks, closes audio, and restores capture controls.
function stopMic() {
  sendCmd("flush");
  micStream?.getTracks().forEach(
    /* Purpose: Release one microphone track.
     * Input: t: MediaStreamTrack.
     * Output: Undefined; stops the track.
     */
    (t) => t.stop());
  micCtx?.close().catch(console.warn);
  micCtx = micStream = null;
  $("micBtn").textContent = "● Start microphone";
  $("micBtn").classList.remove("recording");
  setStreamingButtons(false);
}

// ------------------------------------------------------------------ camera
// Purpose: Start camera capture and frame sampling, or stop an active camera.
// Input: None explicitly; uses browser media APIs and shared camera state.
// Output: Promise resolving to undefined; updates preview/controls or displays acquisition errors.
async function toggleCam() {
  const v = $("mediaVideo");
  if (camStream) {
    stopCamera();
    return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  } catch (e) {
    alert("Camera permission denied: " + e.message);
    return;
  }
  v.srcObject = camStream; v.muted = true; await v.play();
  setMedia(true, true);
  startFrames(v);
  $("camBtn").textContent = "■ Stop camera";
}

// Purpose: Release camera tracks and clear the video/face UI.
// Input: None explicitly; uses current camera state.
// Output: Undefined; stops camera/frame capture and restores controls.
function stopCamera() {
  if (!camStream) return;
  camStream.getTracks().forEach(
    /* Purpose: Release one camera track.
     * Input: t: MediaStreamTrack.
     * Output: Undefined; stops the track.
     */
    (t) => t.stop());
  camStream = null;
  $("mediaVideo").srcObject = null;
  stopFrames(); setMedia(false); setMod("face", null);
  $("camBtn").textContent = "◎ Start camera";
}

// Purpose: Prevent concurrent clip/file inputs while streaming.
// Input: active: whether an input currently owns streaming.
// Output: Undefined; updates busyStreaming and clip/file control availability.
function setStreamingButtons(active) {
  busyStreaming = active;
  $("clipBtn").disabled = active;
  $("fileInput").disabled = active;
}

// ------------------------------------------------------------------ clip / file replay (audio + optional video)
// Purpose: Decode, downmix, and resample media audio for the server.
// Input: buf: encoded media ArrayBuffer.
// Output: Promise of mono Float32Array at 16 kHz; closes the temporary audio context even on failure.
async function decodeTo16k(buf) {
  const ctx = new AudioContext();
  try {
    const decoded = await ctx.decodeAudioData(buf);
    const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * SR), SR);  // resample + downmix
    const s = off.createBufferSource(); s.buffer = decoded; s.connect(off.destination); s.start();
    return (await off.startRendering()).getChannelData(0);
  } finally {
    await ctx.close();
  }
}

// Purpose: Load a video source with bounded waiting and listener cleanup.
// Input: video: HTMLVideoElement; url: media URL.
// Output: Promise resolving when playable, or rejecting on decode/load error or 15-second timeout.
function loadVideo(video, url) {
  return new Promise(
    /* Purpose: Install video readiness/error listeners and a timeout.
     * Input: resolve/reject: loadVideo Promise settlement callbacks; captures video/url.
     * Output: Undefined; begins loading and arranges cleanup on every terminal outcome.
     */
    (resolve, reject) => {
    // Purpose: Settle video loading and remove its timer/listeners.
    // Input: error: optional load failure.
    // Output: Undefined; resolves or rejects the captured Promise.
    const finish = (error) => {
      clearTimeout(timer);
      video.removeEventListener("canplay", loaded);
      video.removeEventListener("error", failed);
      if (error) reject(error); else resolve();
    };
    // Purpose: Finish video loading successfully.
    // Input: None; captures finish().
    // Output: Undefined; resolves the load Promise after cleanup.
    const loaded = () => finish();
    // Purpose: Report a video decode/load failure.
    // Input: None; captures finish().
    // Output: Undefined; rejects the load Promise after cleanup.
    const failed = () => finish(new Error("The video could not be loaded or decoded."));
    const timer = setTimeout(
      /* Purpose: Report that video loading exceeded its deadline.
       * Input: None; captures finish().
       * Output: Undefined; rejects the load Promise with a timeout error.
       */
      () => finish(new Error("Video loading timed out.")), 15000);
    video.addEventListener("canplay", loaded);
    video.addEventListener("error", failed);
    try { video.src = url; video.load(); } catch (error) { finish(error); }
  });
}

// Purpose: Replay media at real-time cadence with aligned audio/video and a silence tail.
// Input: buf: encoded media bytes; videoUrl: optional video URL; truth: optional MELD label/ID.
// Output: Promise resolving to undefined; streams and flushes input, then releases media/context/object-URL resources.
async function streamMedia(buf, videoUrl, truth) {
  const objectUrl = videoUrl?.startsWith("blob:") ? videoUrl : null;
  if (busyStreaming) { if (objectUrl) URL.revokeObjectURL(objectUrl); return; }
  stopCamera();                                    // the clip's video replaces the camera
  setStreamingButtons(true);
  $("micBtn").disabled = true;
  const v = $("mediaVideo");
  const epoch = inputEpoch;
  let playbackCtx = null, playbackSource = null;
  try {
    const pcm = await decodeTo16k(buf.slice(0));
    if (!ws || ws.readyState !== 1 || epoch !== inputEpoch) throw new Error("The local connection is unavailable.");
    truthForNext = truth;
    let play;
    if (videoUrl) {                                  // video element plays picture + sound
      v.srcObject = null; v.muted = false;
      await loadVideo(v, videoUrl);
      setMedia(true, false);
      await v.play();
      startFrames(v);
    } else {                                         // audio only: play it through the speakers
      setMedia(false, false, "audio only");
      playbackCtx = new AudioContext();
      const decoded = await playbackCtx.decodeAudioData(buf.slice(0));
      playbackSource = playbackCtx.createBufferSource();
      playbackSource.buffer = decoded; playbackSource.connect(playbackCtx.destination);
      play = new Promise(
        /* Purpose: Resolve audio playback when the source finishes.
         * Input: r: Promise resolve callback; captures playbackSource.
         * Output: Assigned callback value; registers r as onended.
         */
        (r) => (playbackSource.onended = r));
      playbackSource.start();
    }
    const tail = new Float32Array(Math.round(0.8 * SR));          // silence lets the VAD close the turn
    const all = new Float32Array(pcm.length + tail.length); all.set(pcm);
    const t0 = performance.now();
    for (let i = 0; i < all.length; i += CHUNK) {
      if (!ws || ws.readyState !== 1 || epoch !== inputEpoch) throw new Error("The local connection was interrupted; replay the input.");
      const f = all.subarray(i, i + CHUNK);
      const out = new Int16Array(f.length);
      for (let j = 0; j < f.length; j++) { const x = Math.max(-1, Math.min(1, f[j])); out[j] = x < 0 ? x * 0x8000 : x * 0x7fff; }
      sendPCM(out);
      const wait = t0 + ((i + CHUNK) / SR) * 1000 - performance.now();
      if (wait > 0) await new Promise(
        /* Purpose: Schedule the next replay chunk at its intended stream time.
         * Input: r: Promise resolve callback; captures wait milliseconds.
         * Output: Timer ID from setTimeout; its enclosing Promise ignores that return value.
         */
        (r) => setTimeout(r, wait));
    }
    sendCmd("flush");
    await play;
  } catch (e) {
    alert("Could not stream that media: " + e.message);
  } finally {
    stopFrames();
    if (videoUrl) { v.pause(); v.removeAttribute("src"); v.load(); }
    if (playbackSource) { try { playbackSource.stop(); } catch (_) { /* already ended */ } }
    if (playbackCtx) await playbackCtx.close().catch(console.warn);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    setStreamingButtons(false);
    $("micBtn").disabled = false;
  }
}

// Purpose: Fetch prepared demo clip metadata and populate the selector.
// Input: None explicitly; calls /api/clips.
// Output: Promise resolving to undefined; adds clip options or displays preparation status.
async function loadClips() {
  try {
    const r = await (await fetch("/api/clips")).json();
    const sel = $("clipSelect");
    if (!r.clips.length) { sel.options[0].textContent = "no MELD clips (data not prepared)"; return; }
    for (const c of r.clips) {
      const o = document.createElement("option");
      o.value = JSON.stringify(c);
      const text = c.utterance.length > 48 ? c.utterance.slice(0, 46) + "…" : c.utterance;
      o.textContent = `[${c.emotion}] ${c.speaker}: ${text} (${c.duration_s}s)${c.video_url ? " 🎞" : ""}`;
      sel.appendChild(o);
    }
  } catch (e) { console.warn(e); }
}

$("micBtn").onclick = toggleMic;
$("camBtn").onclick = toggleCam;
$("clipBtn").onclick =
  /* Purpose: Fetch and replay the selected MELD demo clip.
   * Input: None explicitly; reads the selected clip option.
   * Output: Promise resolving to undefined; starts streaming with the clip's emotion/ID ground truth.
   */
  async () => {
  const v = $("clipSelect").value;
  if (!v) return;
  const c = JSON.parse(v);
  const url = c.video_url || c.url;
  const buf = await (await fetch(url)).arrayBuffer();
  streamMedia(buf, c.video_url, { emotion: c.emotion, id: c.id });
};
$("fileInput").onchange =
  /* Purpose: Read and replay a user-selected audio/video file.
   * Input: e: file input change event.
   * Output: Promise resolving to undefined; starts media streaming and clears the file picker value.
   */
  async (e) => {
  const f = e.target.files[0];
  if (f) streamMedia(await f.arrayBuffer(), f.type.startsWith("video/") ? URL.createObjectURL(f) : null, null);
  e.target.value = "";
};
$("resetBtn").onclick =
  /* Purpose: Reset the server conversation and clear the local turn UI.
   * Input: None explicitly; uses the open session.
   * Output: Undefined; sends reset and clears displayed conversation state.
   */
  () => {
  sendCmd("reset");
  clearConversation("Conversation cleared.");
};

$("turns").innerHTML = `<div class="empty-state">Nothing yet. Start the microphone (and camera), or stream a MELD clip.</div>`;
renderBars(null);
connect();
loadClips();
