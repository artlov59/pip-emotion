"""Streaming pipeline: audio (+ optional webcam frames) in -> structured emotion state + streamed reply out.

    PCM chunks ─► VAD (32 ms frames) ─┬─ while speaking, every 0.5 s ─► AudioCNN(last 3 s) ⊕ face(last 3 s) ─► partial_state
                                      └─ 500 ms of silence ─► utterance
                                             ├─ AudioCNN(utterance) ─────────────────────┐ voice
                                             ├─ Whisper(utterance) ─► DistilRoBERTa ─────┤ words      late fusion
                                             ├─ face frames during the utterance ────────┤ face   ─►  + agreement ─► final_state ─► Qwen ─► response_*
                                             └─ prosody(utterance) ──────────────────────┘ (tags)
    JPEG frames (~5/s) ─► YuNet + HSEmotion ─► face_state  (continuously, also while nobody speaks)

Threads: the asyncio loop only does VAD + bookkeeping. The emotion CNN, text classifier and
prosody run on a "torch" worker thread; face analysis on a "vision" thread; Whisper and the
LLM on a single "mlx" thread (MLX models must be created and used on the same thread).
Partial updates and face updates therefore keep flowing while a reply is being generated.

Each extra modality is optional: if its model is missing or no camera is connected, the
system degrades to fewer modalities instead of failing.
"""
import asyncio
import collections
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app import fusion, prosody
from app.config import SAMPLE_RATE, Settings
from app.llm import TemplateResponder, build_messages


# Purpose: Measure elapsed time since a performance-counter timestamp.
# Input: t0: time.perf_counter() start value in seconds.
# Output: Elapsed milliseconds, rounded to one decimal place.
def _now_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


# Purpose: Keep only the public emotion fields of a modality reading.
# Input: r: reading dict or None.
# Output: New dict containing available emotion/confidence/probs/frames fields, or None.
def _lite(r):
    return None if r is None else {k: r[k] for k in ("emotion", "confidence", "probs", "frames") if k in r}


class Engine:
    """Owns the models and worker threads. Components can be injected for tests."""

    # Purpose: Own shared models and create dedicated inference worker pools.
    # Input: settings: runtime Settings; emotion/asr/responder/text/vision: optional components;
    #          vad_factory: optional VAD constructor; backend/cpu_threads: runtime selection.
    # Output: None; stores components, creates worker pools, and loads fusion coefficients.
    def __init__(self, settings: Settings, emotion=None, asr=None, responder=None, vad_factory=None,
                 text=None, vision=None, backend="mlx", cpu_threads=None):
        self.settings = settings
        self.backend, self.cpu_threads = backend, cpu_threads
        self.emotion, self.asr, self.responder, self.vad_factory = emotion, asr, responder, vad_factory
        self.text, self.vision = text, vision
        self.torch_pool = ThreadPoolExecutor(1, thread_name_prefix="torch")
        self.mlx_pool = ThreadPoolExecutor(1, thread_name_prefix="cpu-generation" if backend == "cpu" else "mlx")
        self.vision_pool = ThreadPoolExecutor(1, thread_name_prefix="vision")
        self.load_times = {}
        self.errors = {}
        self.weights, self.weights_all, self.weights_tuned = fusion.load_weights()
        self.agreement_calibration = fusion.load_agreement_calibration()

    # Purpose: Load and warm up components on their required worker threads.
    # Input: None explicitly; uses injected components and Settings.
    # Output: None; records load times/errors and disables unavailable optional components or uses
    #          template replies.
    def load(self):
        s = self.settings
        if self.vad_factory is None:
            from app.vad import SileroVAD
            self.vad_factory = SileroVAD
        if self.emotion is None:
            from app.emotion import EmotionClassifier
            self.emotion = EmotionClassifier(s.emotion_checkpoint, s.torch_device)
        if self.asr is None:
            from app.asr import WhisperASR
            self.asr = WhisperASR(s.asr_repo)
        if self.responder is None:
            if s.use_llm:
                from app.llm import Responder
                self.responder = Responder(s.llm_repo, s.max_response_tokens)
            else:
                self.responder = TemplateResponder()
        if self.text is None and s.use_text:
            from app.text_emotion import TextEmotionClassifier
            self.text = TextEmotionClassifier(s.text_model_dir, s.torch_device)
        if self.vision is None and s.use_vision:
            from app.vision import FaceEmotion
            self.vision = FaceEmotion(s.face_model)

        # Purpose: Load a component while recording its initialization time.
        # Input: name: reporting key; fn: zero-argument loading callable.
        # Output: None; runs fn, records seconds in load_times, and prints progress.
        def timed(name, fn):
            t0 = time.perf_counter()
            fn()
            self.load_times[name] = round(time.perf_counter() - t0, 2)
            print(f"[engine] loaded {name} in {self.load_times[name]} s", flush=True)

        # Purpose: Load an optional component and record an explicit degraded state on failure.
        # Input: name: reporting key; pool: component worker; attr: Engine component attribute name.
        # Output: None; loads the component or clears the attribute and records its error.
        def optional(name, pool, attr):
            comp = getattr(self, attr)
            if comp is None:
                return
            try:
                pool.submit(timed, name, comp.load).result()
            except Exception as ex:  # keep running with fewer modalities, but say so loudly
                self.errors[name] = repr(ex)
                setattr(self, attr, None)
                print(f"[engine] {name} unavailable ({ex!r}); continuing without it", flush=True)

        timed("vad", lambda: self.vad_factory())
        timed("prosody", lambda: prosody.analyze(np.zeros(SAMPLE_RATE, dtype=np.float32)))  # librosa warm-up
        self.torch_pool.submit(timed, "emotion_cnn", self.emotion.load).result()
        self.mlx_pool.submit(timed, "asr", self.asr.load).result()
        optional("text_emotion", self.torch_pool, "text")
        optional("vision", self.vision_pool, "vision")
        try:
            self.mlx_pool.submit(timed, "llm", self.responder.load).result()
        except Exception as ex:
            self.errors["llm"] = repr(ex)
            print(f"[engine] LLM failed to load ({ex!r}); falling back to template replies", flush=True)
            self.responder = TemplateResponder()
        if not self.weights_tuned:
            print("[engine] results/fusion_weights.json not found: using default fusion weights "
                  f"{self.weights} (provide results/fusion_weights.json)", flush=True)

    # Purpose: Select development weights for the current available modality subset.
    # Input: available: set of audio/text/vision names.
    # Output: Runtime modality-to-weight dict, using full-system defaults if no subset table exists.
    def weights_for(self, available: set) -> dict:
        """Dev-tuned weights for exactly this set of modalities, else the full-system weights."""
        return fusion.runtime_weights(available, {"live": self.weights, "all": self.weights_all})

    # Purpose: Describe loaded models, provenance, readiness, and startup failures.
    # Input: None explicitly; reads current Engine components and metadata.
    # Output: JSON-compatible runtime information dict.
    def info(self) -> dict:
        active = {"audio": self.emotion is not None, "text": self.text is not None,
                  "vision": self.vision is not None}
        response_mode = ("unavailable" if self.responder is None else
                         "templates" if isinstance(self.responder, TemplateResponder) else "llm")
        return {
            "backend": self.backend, "cpu_threads": self.cpu_threads,
            "backend_provenance": {
                name: {key: value for key, value in (getattr(component, "provenance", {}) or {}).items()
                       if key != "mapping_proof"}
                for name, component in (("asr", self.asr), ("llm", self.responder))
            },
            "emotion_checkpoint": str(self.settings.emotion_checkpoint),
            "emotion_meta": getattr(self.emotion, "meta", {}),
            "asr": getattr(self.asr, "repo", None),
            "llm": getattr(self.responder, "repo", None),
            "text_model": str(self.settings.text_model_dir) if self.text else None,
            "vision_model": getattr(self.vision, "repo", None),
            "fusion_weights": self.weights, "fusion_tuned_on_dev": self.weights_tuned,
            "agreement_calibrated_on_dev": self.agreement_calibration["calibrated"],
            "errors": self.errors,
            "load_times_s": self.load_times,
            "active_modalities": active,
            "response_mode": response_mode,
            "multimodal_ready": all(active.values()) and self.asr is not None and response_mode == "llm",
        }

    # Purpose: Run blocking audio/text/prosody work on the Torch worker.
    # Input: fn: callable; *args: positional arguments passed to fn.
    # Output: Awaited return value of fn; exceptions propagate.
    async def run_torch(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.torch_pool, fn, *args)

    # Purpose: Run blocking ASR/generation work on the backend's single generation worker.
    # Input: fn: callable; *args: positional arguments passed to fn.
    # Output: Awaited return value of fn; exceptions propagate.
    async def run_mlx(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.mlx_pool, fn, *args)

    # Purpose: Run blocking face analysis on the vision worker.
    # Input: fn: callable; *args: positional arguments passed to fn.
    # Output: Awaited return value of fn; exceptions propagate.
    async def run_vision(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.vision_pool, fn, *args)

    # Purpose: Bridge blocking responder generation into the session's async loop.
    # Input: messages: chat role/content dicts.
    # Output: Async iterator of reply text deltas; worker errors propagate and cancellation stops
    #          further publishing.
    async def stream_reply(self, messages):
        """Bridge the LLM's blocking token generator (mlx thread) into an async iterator."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stopped = threading.Event()

        # Purpose: Queue a generation event if the stream is still active.
        # Input: event: (kind, payload) tuple; captures the async queue and stop event.
        # Output: None; inserts the event into the queue unless cancelled.
        def enqueue(event):
            if not stopped.is_set():
                q.put_nowait(event)

        # Purpose: Schedule a worker event safely on the asyncio thread.
        # Input: event: (kind, payload) tuple; captures the loop and stream stop event.
        # Output: None; schedules enqueue, or marks the stream stopped if the loop has closed.
        def publish(event):
            if not stopped.is_set():
                try:
                    loop.call_soon_threadsafe(enqueue, event)
                except RuntimeError:  # the client/event loop may already have closed
                    stopped.set()

        # Purpose: Consume the blocking responder and publish deltas/errors/completion.
        # Input: None explicitly; captures messages, responder, publish, and cancellation state.
        # Output: None; publishes generation events to the async consumer.
        def work():
            try:
                if stopped.is_set():
                    return
                for delta in self.responder.stream(messages):
                    if stopped.is_set():
                        break
                    publish(("delta", delta))
            except Exception as ex:
                publish(("error", repr(ex)))
            finally:
                publish(("done", None))

        fut = loop.run_in_executor(self.mlx_pool, work)
        try:
            while True:
                kind, val = await q.get()
                if kind == "delta":
                    yield val
                elif kind == "error":
                    raise RuntimeError(val)
                else:
                    break
            await fut
        finally:
            # A running model call cannot be interrupted mid-token. Stop at its next
            # yielded token, and discard callbacks already queued for the old client.
            stopped.set()
            fut.cancel()

    # Purpose: Release the Engine's worker pools.
    # Input: None explicitly; uses the three worker executors.
    # Output: None; shuts down pools without waiting and cancels pending futures.
    def close(self):
        for pool in (self.torch_pool, self.mlx_pool, self.vision_pool):
            pool.shutdown(wait=False, cancel_futures=True)


# Purpose: Decode a JPEG packet into an OpenCV image.
# Input: data: encoded JPEG bytes.
# Output: BGR NumPy image, or None if decoding fails.
def decode_jpeg(data: bytes):
    import cv2
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


class Session:
    """One conversation (one WebSocket). feed() float32 PCM @ 16 kHz, feed_frame() JPEG/BGR frames; read .out."""

    # Purpose: Create independent VAD, turn buffers, task tracking, and event state.
    # Input: engine: loaded shared Engine.
    # Output: None; initializes one conversation and its output queue.
    def __init__(self, engine: Engine):
        self.e = engine
        self.s = s = engine.settings
        self.out: asyncio.Queue = asyncio.Queue()
        self.vad = engine.vad_factory()
        self.frame = self.vad.FRAME
        sr = SAMPLE_RATE
        self.end_silence = int(s.end_silence_ms * sr / 1000)
        self.min_speech = int(s.min_speech_ms * sr / 1000)
        self.max_utt = int(s.max_utterance_s * sr)
        self.partial_every = int(s.partial_every_s * sr)
        self.partial_min = int(s.partial_min_s * sr)
        self.partial_window = int(s.partial_window_s * sr)
        self.final_max = int(s.final_max_s * sr)
        self.pre_roll = collections.deque(maxlen=max(1, math.ceil(s.pre_roll_ms * sr / 1000 / self.frame)))
        self.pending = np.zeros(0, dtype=np.float32)
        self.samples_seen = 0
        self.in_speech = False
        self.utt: list = []
        self.utt_len = 0
        self.silence_run = 0
        self.voiced = 0          # samples in this utterance that the VAD did not call silence
        self.utt_start = 0
        self.last_partial_at = 0
        self.partial_busy = False
        self.turn_id = 0
        self.history: list = []
        self.baseline = prosody.ProsodyBaseline()
        self.turn_lock = asyncio.Lock()
        self.tasks: set = set()
        self._epoch = 0
        self._speech_generation = 0
        # vision
        self.faces = collections.deque(maxlen=int(s.face_history_s * 10))  # (stream_t_s, analysis)
        self.vision_busy = False
        self.frames_in = self.frames_dropped = 0

    # ---------------------------------------------------------------- input
    # Purpose: Report the session's current audio-stream time.
    # Input: None explicitly; uses samples_seen and SAMPLE_RATE.
    # Output: Elapsed input-audio seconds as a float.
    @property
    def now_s(self) -> float:
        return self.samples_seen / SAMPLE_RATE

    # Purpose: Publish one event for the session consumer.
    # Input: event: JSON-compatible event dict.
    # Output: None; adds the event to self.out.
    def emit(self, event: dict):
        event.setdefault("stream_t_s", round(self.now_s, 2))
        self.out.put_nowait(event)

    # Purpose: Describe the conversation's input format and active inference components.
    # Input: None explicitly; reads session settings and Engine readiness.
    # Output: Ready-event dict with sample rate, timing settings, modalities, and Engine information.
    def ready_event(self) -> dict:
        s = self.s
        return {"type": "ready", "sample_rate": SAMPLE_RATE, "chunk_ms": s.chunk_ms,
                "end_silence_ms": s.end_silence_ms, "partial_every_s": s.partial_every_s,
                "partial_window_s": s.partial_window_s,
                "modalities": {"voice": True, "words": self.e.text is not None, "face": self.e.vision is not None},
                "engine": self.e.info()}

    # Purpose: Accept audio, run VAD on complete frames, and schedule inference.
    # Input: pcm: one-dimensional mono float PCM samples at 16 kHz.
    # Output: None; updates stream buffers and schedules turn/partial work without waiting for models.
    def feed(self, pcm: np.ndarray):
        """Non-blocking: runs VAD and schedules model work on worker threads."""
        self.pending = np.concatenate([self.pending, pcm.astype(np.float32)])
        n = self.frame
        while self.pending.size >= n:
            frame, self.pending = self.pending[:n], self.pending[n:]
            self._on_frame(frame, self.vad.prob(frame))

    # Purpose: Accept a video frame and schedule face analysis if the worker is free.
    # Input: image: JPEG bytes or OpenCV BGR array.
    # Output: None; schedules analysis or records a dropped frame when vision is unavailable/busy.
    def feed_frame(self, image):
        """A webcam/video frame (JPEG bytes or BGR array). Dropped if the vision thread is still busy."""
        if self.e.vision is None:
            return
        self.frames_in += 1
        if self.vision_busy:
            self.frames_dropped += 1
            return
        self.vision_busy = True
        self._spawn(self._analyze_frame(image, self.now_s, self._epoch))

    # Purpose: Insert a precomputed face reading at its audio-stream timestamp.
    # Input: t_s: frame time in seconds; result: face-analysis dict.
    # Output: None; appends the reading to the session's face history.
    def add_face_result(self, t_s: float, result: dict):
        """Offline frame analysis: insert an already-computed frame analysis."""
        self.faces.append((t_s, result))

    # Purpose: Finalize buffered speech immediately at the end of an input.
    # Input: None explicitly; uses the current VAD/utterance buffers.
    # Output: None; closes an active utterance; does nothing when no utterance is open.
    def flush(self):
        """End the current utterance now (e.g. end of a replayed file / push-to-talk release)."""
        if self.in_speech:
            self._end_utterance("flush")

    # Purpose: Clear conversation state and invalidate pending results.
    # Input: None explicitly; uses current tasks, buffers, and history.
    # Output: None; cancels work, clears state, and emits reset while preserving monotonic time/IDs.
    def reset(self):
        """Clear a conversation, including in-flight work; keep stream time/IDs monotonic."""
        self._epoch += 1
        self._speech_generation += 1
        for task in list(self.tasks):
            task.cancel()
        while not self.out.empty():
            self.out.get_nowait()
        self.pending = np.zeros(0, dtype=np.float32)
        self.pre_roll.clear()
        self.in_speech = False
        self.utt = []
        self.utt_len = self.silence_run = self.voiced = self.last_partial_at = 0
        self.utt_start = self.samples_seen
        self.partial_busy = self.vision_busy = False
        self.faces.clear()
        self.frames_in = self.frames_dropped = 0
        self.vad.reset()
        self.history.clear()
        self.baseline = prosody.ProsodyBaseline()
        self.emit({"type": "reset"})

    # Purpose: Advance speech-boundary state and schedule periodic partial inference.
    # Input: frame: one VAD-sized mono audio array; p: speech probability.
    # Output: None; updates utterance state and emits/schedules VAD, partial, or completed-turn work.
    def _on_frame(self, frame: np.ndarray, p: float):
        s = self.s
        self.samples_seen += frame.size
        if not self.in_speech:
            self.pre_roll.append(frame)
            if p >= s.vad_threshold:
                self._speech_generation += 1
                self.in_speech = True
                self.utt = list(self.pre_roll)
                self.utt_len = sum(f.size for f in self.utt)
                self.utt_start = self.samples_seen - self.utt_len
                self.silence_run = 0
                self.voiced = frame.size
                self.last_partial_at = 0
                self.pre_roll.clear()
                self.emit({"type": "vad", "state": "speech_start", "utterance_id": self.turn_id + 1})
            return

        self.utt.append(frame)
        self.utt_len += frame.size
        if p < s.vad_neg_threshold:
            self.silence_run += frame.size
        else:
            self.silence_run = 0
            self.voiced += frame.size

        if self.silence_run >= self.end_silence:
            self._end_utterance("silence")
        elif self.utt_len >= self.max_utt:
            self._end_utterance("max_length")
        elif (not self.partial_busy and self.utt_len - self.silence_run >= self.partial_min
              and self.utt_len - self.last_partial_at >= self.partial_every and self.silence_run == 0):
            self.last_partial_at = self.utt_len
            self.partial_busy = True
            window = np.concatenate(self.utt)[-self.partial_window:]
            end_s = self.now_s
            self._spawn(self._partial(self.turn_id + 1, window,
                                     end_s - window.size / SAMPLE_RATE, end_s,
                                     self._epoch, self._speech_generation))

    # Purpose: Close a detected speech segment and queue final inference.
    # Input: reason: speech-end reason string.
    # Output: None; emits speech-end state and queues usable audio, discarding segments below the
    #          minimum duration.
    def _end_utterance(self, reason: str):
        t_detect = time.perf_counter()
        audio = np.concatenate(self.utt)
        keep_tail = int(0.1 * SAMPLE_RATE)
        trim = max(0, self.silence_run - keep_tail)
        if trim:
            audio = audio[:-trim]
        start_s = self.utt_start / SAMPLE_RATE
        end_s = start_s + audio.size / SAMPLE_RATE
        voiced = self.voiced
        self.in_speech, self.utt, self.utt_len, self.silence_run, self.voiced = False, [], 0, 0, 0
        self.partial_busy = False
        if voiced < self.min_speech:
            self.emit({"type": "vad", "state": "discarded", "reason": "too_short",
                       "voiced_s": round(voiced / SAMPLE_RATE, 2)})
            return
        self.turn_id += 1
        self.emit({"type": "vad", "state": "speech_end", "utterance_id": self.turn_id, "reason": reason})
        self._spawn(self.process_utterance(self.turn_id, audio, t_detect, start_s, end_s))

    # Purpose: Track an asynchronous background operation for this session.
    # Input: coro: coroutine to schedule.
    # Output: None; creates a tracked asyncio task and removes it when done.
    def _spawn(self, coro):
        task = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    # ---------------------------------------------------------------- vision
    # Purpose: Run face inference and publish results only for the current session epoch.
    # Input: image: JPEG bytes or BGR frame; t_s: captured frame time; epoch: reset-generation ID.
    # Output: None; emits face state and records valid results, releasing the busy flag on exit.
    async def _analyze_frame(self, image, t_s: float, epoch: int):
        try:
            # Purpose: Decode a captured frame and invoke the vision model.
            # Input: None explicitly; captures image and the Engine vision component.
            # Output: Face-analysis dict, or a face=False error reading for an undecodable
            #          image.
            def work():
                img = decode_jpeg(image) if isinstance(image, (bytes, bytearray, memoryview)) else image
                if img is None:
                    return {"face": False, "error": "undecodable frame"}
                return self.e.vision.analyze(img)
            r = await self.e.run_vision(work)
            if epoch != self._epoch:
                return
            self.faces.append((t_s, r))
            ev = {"type": "face_state", "face": r.get("face", False), "latency_ms": {"vision": r.get("ms")},
                  "frames_dropped": self.frames_dropped}
            if r.get("face"):
                top = max(r["probs"], key=r["probs"].get)
                ev.update({"emotion": top, "confidence": r["probs"][top], "probs": r["probs"], "box": r["box"]})
            self.emit(ev)
        except Exception as ex:
            if epoch == self._epoch:
                self.emit({"type": "error", "where": "vision", "message": repr(ex)})
        finally:
            if epoch == self._epoch:
                self.vision_busy = False

    # Purpose: Aggregate faces captured inside an utterance's time interval.
    # Input: t0/t1: inclusive audio-stream bounds in seconds.
    # Output: Mean face emotion reading, or None when the interval has no usable faces.
    def face_summary(self, t0: float, t1: float):
        from app.vision import mean_probs
        frames = [r for t, r in self.faces if t0 <= t <= t1]
        return mean_probs(frames) if frames else None

    # ---------------------------------------------------------------- models
    # Purpose: Classify a captured speech window and emit a provisional emotion state.
    # Input: utt_id: turn ID; window: waveform; start_s/end_s: captured bounds; epoch/speech_generation:
    #          stale-result guards.
    # Output: None; emits partial_state only if the captured speech is still current.
    async def _partial(self, utt_id: int, window: np.ndarray, start_s: float, end_s: float,
                       epoch: int, speech_generation: int):
        try:
            if epoch != self._epoch or speech_generation != self._speech_generation or not self.in_speech:
                return
            voice = await self.e.run_torch(self.e.emotion.predict, window)
            if epoch != self._epoch or speech_generation != self._speech_generation or not self.in_speech:
                return
            face = self.face_summary(start_s, end_s) if self.e.vision else None
            readings = {"audio": voice, "vision": face}
            fused = fusion.fuse(readings, self.e.weights_for({m for m, r in readings.items() if r}))
            self.emit({"type": "partial_state", "utterance_id": utt_id, "emotion": fused["emotion"],
                       "confidence": fused["confidence"], "probs": fused["probs"],
                       "modalities": {"voice": _lite(voice), "face": _lite(face)},
                       "window_s": round(window.size / SAMPLE_RATE, 2), "latency_ms": {"emotion": voice["ms"]}})
        except Exception as ex:
            if epoch == self._epoch and speech_generation == self._speech_generation and self.in_speech:
                self.emit({"type": "error", "where": "partial", "message": repr(ex)})
        finally:
            if epoch == self._epoch and speech_generation == self._speech_generation:
                self.partial_busy = False

    # Purpose: Fuse voice, recognized words, faces, and prosody, then stream a reply.
    # Input: utt_id: turn ID; audio: 16-kHz waveform; t_detect: optional detection clock; start_s/end_s:
    #          audio-stream bounds.
    # Output: None; emits final_state, response_delta, response_done, or error events and updates
    #          bounded history.
    async def process_utterance(self, utt_id: int, audio: np.ndarray, t_detect: float = None,
                                start_s: float = 0.0, end_s: float = None):
        """Voice + words + face -> fused final_state (with agreement), then stream the reply."""
        epoch = self._epoch
        t_detect = t_detect or time.perf_counter()
        end_s = end_s if end_s is not None else start_s + audio.size / SAMPLE_RATE
        async with self.turn_lock:  # answer turns in order; partials keep running meanwhile
            if epoch != self._epoch:
                return
            queue_ms = _now_ms(t_detect)
            t0 = time.perf_counter()
            emo_task = self.e.run_torch(self.e.emotion.predict, audio[-self.final_max:])
            asr_task = self.e.run_mlx(self.e.asr.transcribe, audio)
            pros_task = asyncio.get_running_loop().run_in_executor(None, prosody.analyze, audio)
            try:
                voice, asr, pros = await asyncio.gather(emo_task, asr_task, pros_task)
                words = await self.e.run_torch(self.e.text.predict, asr["text"]) if self.e.text else None
            except Exception as ex:
                if epoch == self._epoch:
                    self.emit({"type": "error", "where": "final", "utterance_id": utt_id, "message": repr(ex)})
                return
            if epoch != self._epoch:
                return
            face = self.face_summary(start_s - 0.2, end_s + 0.2) if self.e.vision else None
            readings = {"audio": voice, "text": words, "vision": face}
            available = {m for m, r in readings.items() if r}
            weights = self.e.weights_for(available)
            fused = fusion.fuse(readings, weights)
            agree = fusion.agreement(words, voice, face, getattr(self.e, "agreement_calibration", None))

            prosody.add_speech_rate(pros, asr["text"])
            labels = self.baseline.describe(pros)
            self.baseline.update(pros)
            delivery = prosody.describe_delivery(labels)
            tags = [f"emotion:{fused['emotion']}", f"voice:{voice['emotion']}"]
            tags += [f"words:{words['emotion']}"] if words else []
            tags += [f"face:{face['emotion']}"] if face else []
            tags += [f"agreement:{agree['kind']}"] if agree["kind"] != "unknown" else []
            tags += [f"{k}:{v}" for k, v in labels.items() if v and v != "normal"]
            state = {
                "type": "final_state",
                "utterance_id": utt_id,
                "start_s": round(start_s, 2),
                "duration_s": round(audio.size / SAMPLE_RATE, 2),
                "emotion": fused["emotion"],
                "confidence": fused["confidence"],
                "probs": fused["probs"],
                "modalities": {"voice": _lite(voice), "words": _lite(words), "face": _lite(face)},
                "fusion": {"used": fused["used"], "weights": {m: weights.get(m) for m in fused["used"]},
                           "tuned_on_dev": self.e.weights_tuned},
                "agreement": agree,
                "transcript": asr["text"],
                "tags": tags,
                "prosody": {**pros, "labels": labels},
                "delivery": delivery,
                "latency_ms": {"queue": queue_ms, "emotion": voice["ms"], "asr": asr["ms"],
                               "text_emotion": words["ms"] if words else None,
                               "parallel_stage": _now_ms(t0), "end_of_speech_to_state": _now_ms(t_detect)},
            }
            self.emit(state)

            history_limit = max(0, self.s.history_turns)
            messages = build_messages(state, self.history[-history_limit:] if history_limit else [])
            t_llm = time.perf_counter()
            first_ms, reply, n = None, "", 0
            try:
                async for delta in self.e.stream_reply(messages):
                    if epoch != self._epoch:
                        return
                    if first_ms is None:
                        first_ms = _now_ms(t_detect)
                    reply += delta
                    n += 1
                    self.emit({"type": "response_delta", "utterance_id": utt_id, "text": delta})
            except Exception as ex:
                if epoch == self._epoch:
                    self.emit({"type": "error", "where": "llm", "utterance_id": utt_id, "message": repr(ex)})
            if epoch != self._epoch:
                return
            gen_s = time.perf_counter() - t_llm
            reply = reply.strip()
            self.emit({"type": "response_done", "utterance_id": utt_id, "text": reply,
                       "latency_ms": {"end_of_speech_to_first_token": first_ms,
                                      "end_of_speech_to_last_token": _now_ms(t_detect),
                                      "generation": round(gen_s * 1000, 1)},
                       "tokens": n, "tokens_per_s": round(n / gen_s, 1) if gen_s > 0 else None,
                       "token_count_source": "visible_stream_deltas", "stream_deltas": n,
                       "stream_deltas_per_s": round(n / gen_s, 1) if gen_s > 0 else None})
            self.history.append({"transcript": state["transcript"], "emotion": state["emotion"], "reply": reply})
            del self.history[:max(0, len(self.history) - history_limit)]

    # Purpose: Wait for scheduled session work to finish within a timeout.
    # Input: timeout: maximum wait in seconds, default 60.
    # Output: None; awaits outstanding tasks and propagates timeout when exceeded.
    async def drain(self, timeout: float = 60.0):
        """Wait for all scheduled work to finish (used by replay/benchmark/offline analysis)."""
        t_end = time.perf_counter() + timeout
        while self.tasks and time.perf_counter() < t_end:
            await asyncio.wait(list(self.tasks), timeout=max(0.0, t_end - time.perf_counter()))

    # Purpose: Cancel remaining conversation work.
    # Input: None explicitly; uses session task tracking.
    # Output: None; requests cancellation of scheduled tasks.
    def close(self):
        for t in list(self.tasks):
            t.cancel()
