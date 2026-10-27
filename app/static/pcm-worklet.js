// AudioWorklet: microphone (usually 44.1/48 kHz float) -> 16 kHz int16 chunks of 100 ms.
// A native BiquadFilter low-pass (7 kHz) sits in front of this node to limit aliasing.
class PCMCapture extends AudioWorkletProcessor {
  // Purpose: Initialize streaming resampling state and the transferable PCM buffer.
  // Input: options: AudioWorklet options with optional targetRate/chunkSamples.
  // Output: New processor state; default output is 16-kHz int16 chunks of 1600 samples.
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.step = sampleRate / (o.targetRate || 16000); // input samples per output sample
    this.chunk = o.chunkSamples || 1600;
    this.buf = new Int16Array(this.chunk);
    this.n = 0;
    this.t = 0;      // read position relative to the start of the current block (>= -1)
    this.prev = 0;   // last sample of the previous block (position -1)
  }

  // Purpose: Resample microphone blocks with interpolation and emit full PCM chunks.
  // Input: inputs: AudioWorklet input buses; reads the first mono channel.
  // Output: True to keep processing; posts completed Int16Array buffers and preserves partial/interpolation state.
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    const L = ch.length;
    let t = this.t;
    while (t < L - 1) {
      const i = Math.floor(t);
      const f = t - i;
      const a = i < 0 ? this.prev : ch[i];
      const b = ch[i + 1];
      let s = a + (b - a) * f;               // linear interpolation
      s = s > 1 ? 1 : s < -1 ? -1 : s;
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === this.chunk) {
        this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
        this.buf = new Int16Array(this.chunk);
        this.n = 0;
      }
      t += this.step;
    }
    this.t = t - L;
    this.prev = ch[L - 1];
    return true;
  }
}

registerProcessor("pcm-capture", PCMCapture);
