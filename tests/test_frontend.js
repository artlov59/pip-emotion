const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
// Purpose: Build a lightweight DOM element stub for browser logic tests.
// Input: None.
// Output: Object with mutable UI fields and minimal DOM/canvas/media methods.
const element = () => ({
  style: {}, classList: {
    /* Purpose: Accept a class-list add operation in the DOM stub.
     * Input: None explicitly; arguments are ignored.
     * Output: Undefined; no-op fixture.
     */
    add() {},
    /* Purpose: Accept a class-list remove operation in the DOM stub.
     * Input: None explicitly; arguments are ignored.
     * Output: Undefined; no-op fixture.
     */
    remove() {} }, options: [{}],
  textContent: '', innerHTML: '', disabled: false, title: '',
  // Purpose: Simulate absence of nested DOM elements.
  // Input: None explicitly; selector arguments are ignored.
  // Output: Null.
  querySelector() { return null; },
    /* Purpose: Accept turn markup insertion in the DOM stub.
     * Input: None explicitly; inserted arguments are ignored.
     * Output: Undefined; no-op fixture.
     */
    prepend() {},
    /* Purpose: Accept selector-option insertion in the DOM stub.
     * Input: None explicitly; child arguments are ignored.
     * Output: Undefined; no-op fixture.
     */
    appendChild() {},
  // Purpose: Provide the canvas operation used by overlay cleanup.
  // Input: None explicitly; context-type arguments are ignored.
  // Output: Context stub exposing clearRect().
  getContext() { return {
    /* Purpose: Accept overlay clearing without rendering a canvas.
     * Input: None explicitly; coordinates are ignored.
     * Output: Undefined; no-op fixture.
     */
    clearRect() {} }; },
    /* Purpose: Accept a media pause operation.
     * Input: None.
     * Output: Undefined; no-op fixture.
     */
    pause() {},
    /* Purpose: Accept a media load operation in the test stub.
     * Input: None.
     * Output: Undefined; no-op fixture.
     */
    load() {},
    /* Purpose: Accept media-source removal in the DOM stub.
     * Input: None explicitly; attribute arguments are ignored.
     * Output: Undefined; no-op fixture.
     */
    removeAttribute() {},
});
const timers = new Map(); let timerID = 0;
const alerts = [];
const context = vm.createContext({
  console, Map, Uint8Array, Int16Array, Float32Array, ArrayBuffer,
  document: { documentElement: {},
    /* Purpose: Reuse or create a fake page element.
     * Input: id: requested element ID.
     * Output: Stable element stub stored in the elements map.
     */
    getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }, createElement: element },
  // Purpose: Provide the theme-style interface for browser tests.
  // Input: None explicitly; element arguments are ignored.
  // Output: Style stub exposing getPropertyValue().
  getComputedStyle() { return {
    /* Purpose: Return a deterministic theme color.
     * Input: None explicitly; property arguments are ignored.
     * Output: String #fff.
     */
    getPropertyValue() { return '#fff'; } }; },
  location: { protocol: 'http:', host: 'localhost' },
  WebSocket: class {
    /* Purpose: Simulate an immediately open WebSocket.
     * Input: None explicitly; URL arguments are ignored.
     * Output: New socket stub with readyState=1.
     */
    constructor() { this.readyState = 1; }
    /* Purpose: Accept browser packets without network access.
     * Input: None explicitly; packet arguments are ignored.
     * Output: Undefined; no-op fixture.
     */
    send() {} },
  fetch:
    /* Purpose: Simulate the demo-clips endpoint without network access.
     * Input: None explicitly; request arguments are ignored.
     * Output: Promise of a response stub exposing json().
     */
    async () => ({ json:
    /* Purpose: Decode the fake clips response.
     * Input: None.
     * Output: Promise of an object with an empty clips array.
     */
    async () => ({ clips: [] }) }),
  // Purpose: Record a timeout for controlled delivery by the test.
  // Input: fn: callback; ms: requested delay.
  // Output: Synthetic timer ID; stores callback/delay without waiting.
  setTimeout(fn, ms) { const id = ++timerID; timers.set(id, { fn, ms }); return id; },
  // Purpose: Cancel a recorded timeout.
  // Input: id: synthetic timer ID.
  // Output: Undefined; removes its map entry.
  clearTimeout(id) { timers.delete(id); },
    /* Purpose: Disable automatic frame sampling in browser tests.
     * Input: None explicitly; interval arguments are ignored.
     * Output: Undefined; no timer is scheduled.
     */
    setInterval() {},
    /* Purpose: Accept interval cleanup in the timer stub.
     * Input: None explicitly; timer arguments are ignored.
     * Output: Undefined; no-op fixture.
     */
    clearInterval() {},
  // Purpose: Capture an error message for assertions.
  // Input: message: alert text.
  // Output: Undefined; appends to alerts.
  alert(message) { alerts.push(message); }, navigator: { mediaDevices: {} },
  URL: {
    /* Purpose: Accept blob-URL cleanup without creating browser resources.
     * Input: None explicitly; URL arguments are ignored.
     * Output: Undefined; no-op fixture.
     */
    revokeObjectURL() {} }, performance: { now:
    /* Purpose: Provide a fixed replay clock for browser tests.
     * Input: None.
     * Output: Number zero.
     */
    () => 0 },
});
vm.runInContext(fs.readFileSync(process.argv[2] || require('node:path').join(__dirname, '../app/static/app.js'), 'utf8'), context);
// Purpose: Evaluate browser code in the isolated fake-page context.
// Input: code: JavaScript source string.
// Output: Result of vm.runInContext().
const run = (code) => vm.runInContext(code, context);
// Purpose: Create a media fixture with controllable readiness/error listeners.
// Input: None.
// Output: Object exposing listener storage, registration/removal, load(), and fire().
function video() {
  const listeners = new Map();
  return { listeners,
    /* Purpose: Record one video listener for explicit delivery.
     * Input: k: event name; f: listener callback.
     * Output: Undefined; stores f under k.
     */
    addEventListener(k, f) { listeners.set(k, f); },
    /* Purpose: Remove one recorded video listener.
     * Input: k: event name.
     * Output: Undefined; deletes the listener.
     */
    removeEventListener(k) { listeners.delete(k); },
    /* Purpose: Accept a media load operation in the test stub.
     * Input: None.
     * Output: Undefined; no-op fixture.
     */
    load() {},
    /* Purpose: Deliver a recorded video event.
     * Input: k: event name.
     * Output: Undefined; invokes the matching fixture listener.
     */
    fire(k) { listeners.get(k)(); } };
}
(
  /* Purpose: Verify browser attribution, reconnect, and media cleanup regressions.
   * Input: None explicitly; uses the isolated page and controlled browser fixtures.
   * Output: Promise resolving after assertions; rejects on a failed assertion.
   */
  async () => {
  run(`truthForNext = {emotion:'joy',id:'first'}; handle({type:'vad',state:'speech_start',utterance_id:1}); truthForNext={emotion:'anger',id:'second'};`);
  context.final = { type: 'final_state', utterance_id: 1, emotion: 'joy', confidence: .7, duration_s: 1, transcript: 'hello', probs: {}, modalities: { voice: null, words: null, face: null }, fusion: { used: ['audio'] }, tags: [], latency_ms: { end_of_speech_to_state: 12, emotion: 1, asr: 2 } };
  run('handle(final)');
  assert.equal(run('turns.get(1).truth.id'), 'first');
  run(`handle({type:'ready',modalities:{face:false},engine:{}})`);
  assert.equal(run('turns.size'), 0);
  assert.equal(run('truthByUtterance.size'), 0);
  assert.equal(elements.get('camBtn').disabled, true);
  run(`handle({type:'ready',modalities:{face:true},engine:{}})`);
  assert.equal(elements.get('camBtn').disabled, false);

  context.testVideo = video();
  const errorLoad = run(`loadVideo(testVideo,'broken.mp4')`);
  context.testVideo.fire('error');
  await assert.rejects(errorLoad, /could not be loaded/);
  assert.equal(context.testVideo.listeners.size, 0);
  assert.equal(timers.size, 0);
  context.testVideo = video();
  const timeoutLoad = run(`loadVideo(testVideo,'stalled.mp4')`);
  [...timers.values()].find(
    /* Purpose: Find the video-loading deadline timer.
     * Input: t: recorded timer entry.
     * Output: Boolean indicating a 15000-ms timeout.
     */
    t => t.ms === 15000).fn();
  await assert.rejects(timeoutLoad, /timed out/);
  assert.equal(context.testVideo.listeners.size, 0);
  assert.equal(timers.size, 0);

  let closes = 0;
  context.AudioContext = class {
    /* Purpose: Simulate an audio decode failure.
     * Input: None explicitly; encoded bytes are ignored.
     * Output: Rejected Promise with a bad-audio error.
     */
    async decodeAudioData() { throw new Error('bad audio'); }
    /* Purpose: Record audio-context resource cleanup.
     * Input: None.
     * Output: Promise resolving to undefined; increments closes.
     */
    async close() { closes++; } };
  await assert.rejects(run('decodeTo16k(new ArrayBuffer(4))'), /bad audio/);
  assert.equal(closes, 1);

  let stops = 0;
  context.navigator.mediaDevices.getUserMedia =
    /* Purpose: Supply a controllable microphone stream without acquiring hardware.
     * Input: None explicitly; media constraints are ignored.
     * Output: Promise of a stream stub with getTracks().
     */
    async () => ({ getTracks:
    /* Purpose: Expose the microphone track fixture.
     * Input: None.
     * Output: Array containing a track with a stop() counter.
     */
    () => [{
    /* Purpose: Record microphone-track cleanup.
     * Input: None.
     * Output: Undefined; increments stops.
     */
    stop() { stops++; } }] });
  context.AudioContext = class {
    /* Purpose: Create an AudioContext fixture whose worklet setup fails.
     * Input: None.
     * Output: New context stub exposing a failing audioWorklet.addModule().
     */
    constructor() { this.audioWorklet = { addModule:
    /* Purpose: Simulate failure to load the microphone AudioWorklet.
     * Input: None explicitly; module URL is ignored.
     * Output: Rejected Promise with a missing-worklet error.
     */
    async () => { throw new Error('missing worklet'); } }; }
    /* Purpose: Record audio-context resource cleanup.
     * Input: None.
     * Output: Promise resolving to undefined; increments closes.
     */
    async close() { closes++; } };
  await run('toggleMic()');
  assert.equal(stops, 1);
  assert.equal(closes, 2);
  assert.equal(run('micCtx'), null);
  assert.equal(run('micStream'), null);
  assert.equal(run('busyStreaming'), false);
  assert.equal(elements.get('micBtn').disabled, false);
  assert.match(alerts.at(-1), /missing worklet/);
  console.log('PASS: truth attribution; reconnect state; video error/timeout cleanup; decode cleanup; mic setup cleanup');
})().catch(
  /* Purpose: Report browser regression failure to the test runner.
   * Input: error: rejected test error.
   * Output: Undefined; logs the error and sets process.exitCode=1.
   */
  error => { console.error(error); process.exitCode = 1; });
