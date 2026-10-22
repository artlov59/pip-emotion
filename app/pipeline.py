"""Streaming pipeline: audio chunks in -> structured emotion state + streamed reply out.

    PCM chunks ─► VAD (32 ms frames) ─┬─ while speaking, every 0.5 s ─► AudioCNN(last 3 s) ─► partial_state
                                      └─ 500 ms of silence ─► utterance
                                             ├─ AudioCNN(utterance) ┐
                                             ├─ Whisper(utterance)  ├─► final_state ─► Qwen ─► response_delta… ─► response_done
                                             └─ prosody(utterance)  ┘

Threads: the asyncio loop only does VAD + bookkeeping. The emotion CNN and prosody run
on a "torch" worker thread; Whisper and the LLM on a single "mlx" worker thread (MLX
models must be created and used on the same thread). Partial updates therefore keep
flowing while a reply is being generated.
"""
import asyncio
import collections
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app import prosody
from app.config import SAMPLE_RATE, Settings
from app.llm import TemplateResponder, build_messages


def _now_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


class Engine:
    """Owns the models and worker threads. Components can be injected for tests."""

    def __init__(self, settings: Settings, emotion=None, asr=None, responder=None, vad_factory=None):
        self.settings = settings
        self.emotion, self.asr, self.responder, self.vad_factory = emotion, asr, responder, vad_factory
        self.torch_pool = ThreadPoolExecutor(1, thread_name_prefix="torch")
        self.mlx_pool = ThreadPoolExecutor(1, thread_name_prefix="mlx")
        self.load_times = {}
        self.responder_error = None

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

        def timed(name, fn):
            t0 = time.perf_counter()
            fn()
            self.load_times[name] = round(time.perf_counter() - t0, 2)
            print(f"[engine] loaded {name} in {self.load_times[name]} s", flush=True)

        timed("vad", lambda: self.vad_factory())
        timed("prosody", lambda: prosody.analyze(np.zeros(SAMPLE_RATE, dtype=np.float32)))  # librosa warm-up
        self.torch_pool.submit(timed, "emotion_cnn", self.emotion.load).result()
        self.mlx_pool.submit(timed, "asr", self.asr.load).result()
        try:
            self.mlx_pool.submit(timed, "llm", self.responder.load).result()
        except Exception as ex:  # keep the demo usable, but say so loudly
            self.responder_error = repr(ex)
            print(f"[engine] LLM failed to load ({ex!r}); falling back to template replies", flush=True)
            self.responder = TemplateResponder()

    def info(self) -> dict:
        return {
            "emotion_checkpoint": str(self.settings.emotion_checkpoint),
            "emotion_meta": getattr(self.emotion, "meta", {}),
            "asr": getattr(self.asr, "repo", None),
            "llm": getattr(self.responder, "repo", None),
            "llm_error": self.responder_error,
            "load_times_s": self.load_times,
        }

    async def run_torch(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.torch_pool, fn, *args)

    async def run_mlx(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.mlx_pool, fn, *args)

    async def stream_reply(self, messages):
        """Bridge the LLM's blocking token generator (mlx thread) into an async iterator."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def work():
            try:
                for delta in self.responder.stream(messages):
                    loop.call_soon_threadsafe(q.put_nowait, ("delta", delta))
            except Exception as ex:
                loop.call_soon_threadsafe(q.put_nowait, ("error", repr(ex)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        fut = loop.run_in_executor(self.mlx_pool, work)
        while True:
            kind, val = await q.get()
            if kind == "delta":
                yield val
            elif kind == "error":
                raise RuntimeError(val)
            else:
                break
        await fut

    def close(self):
        self.torch_pool.shutdown(wait=False, cancel_futures=True)
        self.mlx_pool.shutdown(wait=False, cancel_futures=True)


class Session:
    """One conversation (one WebSocket). Call feed() with float32 PCM @ 16 kHz; read events from .out."""

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

    # ---------------------------------------------------------------- input
    def emit(self, event: dict):
        event.setdefault("stream_t_s", round(self.samples_seen / SAMPLE_RATE, 2))
        self.out.put_nowait(event)

    def ready_event(self) -> dict:
        s = self.s
        return {"type": "ready", "sample_rate": SAMPLE_RATE, "chunk_ms": s.chunk_ms,
                "end_silence_ms": s.end_silence_ms, "partial_every_s": s.partial_every_s,
                "partial_window_s": s.partial_window_s, "engine": self.e.info()}

    def feed(self, pcm: np.ndarray):
        """Non-blocking: runs VAD and schedules model work on worker threads."""
        self.pending = np.concatenate([self.pending, pcm.astype(np.float32)])
        n = self.frame
        while self.pending.size >= n:
            frame, self.pending = self.pending[:n], self.pending[n:]
            self._on_frame(frame, self.vad.prob(frame))

    def flush(self):
        """End the current utterance now (e.g. end of a replayed file / push-to-talk release)."""
        if self.in_speech:
            self._end_utterance("flush")

    def reset(self):
        self.history.clear()
        self.baseline = prosody.ProsodyBaseline()
        self.emit({"type": "reset"})

    def _on_frame(self, frame: np.ndarray, p: float):
        s = self.s
        self.samples_seen += frame.size
        if not self.in_speech:
            self.pre_roll.append(frame)
            if p >= s.vad_threshold:
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
            self._spawn(self._partial(self.turn_id + 1, window))

    def _end_utterance(self, reason: str):
        t_detect = time.perf_counter()
        audio = np.concatenate(self.utt)
        keep_tail = int(0.1 * SAMPLE_RATE)
        trim = max(0, self.silence_run - keep_tail)
        if trim:
            audio = audio[:-trim]
        start_s = self.utt_start / SAMPLE_RATE
        voiced = self.voiced
        self.in_speech, self.utt, self.utt_len, self.silence_run, self.voiced = False, [], 0, 0, 0
        if voiced < self.min_speech:
            self.emit({"type": "vad", "state": "discarded", "reason": "too_short",
                       "voiced_s": round(voiced / SAMPLE_RATE, 2)})
            return
        self.turn_id += 1
        self.emit({"type": "vad", "state": "speech_end", "utterance_id": self.turn_id, "reason": reason})
        self._spawn(self.process_utterance(self.turn_id, audio, t_detect, start_s))

    def _spawn(self, coro):
        task = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    # ---------------------------------------------------------------- models
    async def _partial(self, utt_id: int, window: np.ndarray):
        try:
            r = await self.e.run_torch(self.e.emotion.predict, window)
            self.emit({"type": "partial_state", "utterance_id": utt_id, "emotion": r["emotion"],
                       "confidence": r["confidence"], "probs": r["probs"],
                       "window_s": round(window.size / SAMPLE_RATE, 2), "latency_ms": {"emotion": r["ms"]}})
        except Exception as ex:
            self.emit({"type": "error", "where": "partial", "message": repr(ex)})
        finally:
            self.partial_busy = False

    async def process_utterance(self, utt_id: int, audio: np.ndarray, t_detect: float = None, start_s: float = 0.0):
        """Final emotion + transcript + prosody -> final_state, then stream the reply."""
        t_detect = t_detect or time.perf_counter()
        async with self.turn_lock:  # answer turns in order; partials keep running meanwhile
            queue_ms = _now_ms(t_detect)
            t0 = time.perf_counter()
            emo_task = self.e.run_torch(self.e.emotion.predict, audio[-self.final_max:])
            asr_task = self.e.run_mlx(self.e.asr.transcribe, audio)
            pros_task = asyncio.get_running_loop().run_in_executor(None, prosody.analyze, audio)
            try:
                emo, asr, pros = await asyncio.gather(emo_task, asr_task, pros_task)
            except Exception as ex:
                self.emit({"type": "error", "where": "final", "utterance_id": utt_id, "message": repr(ex)})
                return
            prosody.add_speech_rate(pros, asr["text"])
            labels = self.baseline.describe(pros)
            self.baseline.update(pros)
            delivery = prosody.describe_delivery(labels)
            tags = [f"emotion:{emo['emotion']}"] + [f"{k}:{v}" for k, v in labels.items() if v and v != "normal"]
            state = {
                "type": "final_state",
                "utterance_id": utt_id,
                "start_s": round(start_s, 2),
                "duration_s": round(audio.size / SAMPLE_RATE, 2),
                "emotion": emo["emotion"],
                "confidence": emo["confidence"],
                "probs": emo["probs"],
                "transcript": asr["text"],
                "tags": tags,
                "prosody": {**pros, "labels": labels},
                "delivery": delivery,
                "latency_ms": {"queue": queue_ms, "emotion": emo["ms"], "asr": asr["ms"],
                               "parallel_stage": _now_ms(t0), "end_of_speech_to_state": _now_ms(t_detect)},
            }
            self.emit(state)

            messages = build_messages(state, self.history[-self.s.history_turns:])
            t_llm = time.perf_counter()
            first_ms, reply, n = None, "", 0
            try:
                async for delta in self.e.stream_reply(messages):
                    if first_ms is None:
                        first_ms = _now_ms(t_detect)
                    reply += delta
                    n += 1
                    self.emit({"type": "response_delta", "utterance_id": utt_id, "text": delta})
            except Exception as ex:
                self.emit({"type": "error", "where": "llm", "utterance_id": utt_id, "message": repr(ex)})
            gen_s = time.perf_counter() - t_llm
            reply = reply.strip()
            self.emit({"type": "response_done", "utterance_id": utt_id, "text": reply,
                       "latency_ms": {"end_of_speech_to_first_token": first_ms,
                                      "end_of_speech_to_last_token": _now_ms(t_detect),
                                      "generation": round(gen_s * 1000, 1)},
                       "tokens": n, "tokens_per_s": round(n / gen_s, 1) if gen_s > 0 else None})
            self.history.append({"transcript": state["transcript"], "emotion": state["emotion"], "reply": reply})

    async def drain(self, timeout: float = 60.0):
        """Wait for all scheduled work to finish (used by replay/benchmark/offline analysis)."""
        t_end = time.perf_counter() + timeout
        while self.tasks and time.perf_counter() < t_end:
            await asyncio.wait(list(self.tasks), timeout=max(0.0, t_end - time.perf_counter()))

    def close(self):
        for t in list(self.tasks):
            t.cancel()
