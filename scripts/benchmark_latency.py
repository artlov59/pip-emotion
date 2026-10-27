"""Latency + resources: stream MELD clips through the in-process local pipeline.

    python -m scripts.benchmark_latency --n 35               # require text + audio + vision + LLM
    python -m scripts.benchmark_latency --n 35 --audio-only  # diagnostic fallback

Excludes WebSocket/browser overhead. Latencies begin when VAD declares end of speech,
so exclude its end-silence wait. Writes per-utterance CSV and per-clip outcomes in JSON.
"""
import argparse
import asyncio
import json
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import psutil

from app.config import EMOTIONS, MELD_DIR, RESULTS_DIR, SAMPLE_RATE, Settings
from app.features import load_audio
from app.pipeline import Engine, Session
from app.backends import create_engine

CHUNK = SAMPLE_RATE // 10
LATENCY_METRICS = ["partial_emotion_ms", "face_frame_ms", "final_emotion_ms", "asr_ms", "text_emotion_ms",
                   "eos_to_state_ms", "eos_to_first_token_ms", "eos_to_last_token_ms", "tokens_per_s"]
MEASUREMENT_DEFINITION = {
    "transport": "in-process Session; excludes WebSocket, network and browser capture overhead",
    "latency_origin": "VAD-declared end of speech; excludes the configured end-silence wait",
    "population": "latencies cover detected utterances; clip_outcomes also includes misses and failures",
    "sampling": "round-robin by emotion from usable 1.5-8 s test clips, up to the requested count",
    "resources": "RSS, MLX peak and MPS driver allocations can overlap in unified memory; do not add them",
    "cpu": "process CPU percentage; 100% means one fully occupied CPU core",
    "token_count_source": "visible_stream_deltas; tokens/tokens_per_s retain legacy field names and are not tokenizer counts",
    "auxiliary_latency": "partial/face timings summarize per-utterance/per-clip mean compute respectively",
}


class MemorySampler(threading.Thread):
    # Purpose: Prepare background process-memory sampling.
    # Input: every: sampling interval in seconds.
    # Output: None; initializes the process handle, interval, stop flag, and peak RSS.
    def __init__(self, every=0.1):
        super().__init__(daemon=True)
        self.proc, self.every, self.peak_rss, self.stop_flag = psutil.Process(), every, 0, False

    # Purpose: Track peak process RSS until stopped.
    # Input: None explicitly; uses the sampler interval and stop event.
    # Output: None; updates peak_rss in bytes.
    def run(self):
        while not self.stop_flag:
            self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            time.sleep(self.every)


# Purpose: Run an optional hardware-reporting command without failing the benchmark.
# Input: cmd: subprocess argument list.
# Output: Stripped stdout string on success, or None when unavailable/failing.
def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


# Purpose: Describe the machine and runtime used for inference measurements.
# Input: backend: mlx or cpu; CPU mode avoids importing GPU runtime libraries.
# Output: Hardware/runtime information dict.
def hardware(backend="mlx"):
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "macos": platform.mac_ver()[0] or None, "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1), "cpu_cores": psutil.cpu_count(), "inference_backend": backend}
    try:
        import torch
        info["torch"] = torch.__version__
        info["mps_available"] = torch.backends.mps.is_available() if backend == "mlx" else None
    except Exception:
        pass
    try:
        info["mlx"] = importlib.metadata.version("mlx")
    except Exception:
        pass
    return info


# Purpose: Read MLX peak device memory when the backend exposes it.
# Input: backend: mlx or cpu.
# Output: Peak memory in GiB despite the legacy function name, or None when unavailable/in CPU mode.
def mlx_peak_gb(backend="mlx"):
    if backend == "cpu":
        return None
    try:
        import mlx.core as mx
        fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        return round(fn() / 2 ** 30, 2)
    except Exception:
        return None


# Purpose: Read Torch MPS driver memory when GPU reporting is applicable.
# Input: backend: mlx or cpu.
# Output: Driver memory in GiB despite the legacy function name, or None when unavailable/in CPU mode.
def mps_driver_gb(backend="mlx"):
    if backend == "cpu":
        return None
    try:
        import torch
        return round(torch.mps.driver_allocated_memory() / 2 ** 30, 2)
    except Exception:
        return None


# Purpose: Choose a seeded, approximately emotion-balanced test clip sample.
# Input: n: exact requested clip count when available; seed: RNG seed.
# Output: DataFrame of usable selected clips.
def pick_clips(n, seed):
    if n <= 0:
        raise ValueError("clip count must be positive")
    df = pd.read_csv(MELD_DIR / "test.csv")
    df = df[(df.status == "ok") & df.duration_s.between(1.5, 8.0)]
    groups = [df[df.emotion == e].sample(frac=1, random_state=seed) for e in EMOTIONS]
    selected = []
    for i in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if i < len(group):
                selected.append(group.iloc[[i]])
                if len(selected) == n:
                    return pd.concat(selected).reset_index(drop=True)
    return pd.concat(selected).reset_index(drop=True) if selected else df.iloc[:0]


# Purpose: Sample JPEG packets from a MELD test clip's original video.
# Input: clip_id: MELD clip identifier; fps: video sampling rate.
# Output: List of (timestamp_seconds, jpeg_bytes), or [] if no video directory is found.
def clip_frames(clip_id, fps=5.0):
    """JPEG frames (t_s, bytes) from the MELD source video, like the browser sends them."""
    from app.config import DATA_DIR
    from app.replay import video_frames
    from training.prepare_meld import find_video_dir
    vdir = find_video_dir(DATA_DIR / "MELD.Raw", "test")
    return video_frames(vdir / f"{clip_id}.mp4", fps) if vdir else []


# Purpose: Feed timed audio/video through one session and collect its results.
# Input: engine: loaded Engine; wav: mono 16-kHz array; speed: replay multiplier; frames: timestamp/JPEG
#          pairs.
# Output: Awaited tuple (events, frame_stats); closes the session after processing.
async def run_clip(engine, wav, speed, frames=()):
    session = Session(engine)
    try:
        stream = np.concatenate([wav, np.zeros(SAMPLE_RATE, dtype=np.float32)])  # close the turn
        t0, fi = time.perf_counter(), 0
        for i in range(0, stream.size, CHUNK):
            while fi < len(frames) and frames[fi][0] <= (i + CHUNK) / SAMPLE_RATE:
                session.feed_frame(frames[fi][1])
                fi += 1
            session.feed(stream[i:i + CHUNK])
            await asyncio.sleep(max(0.0, t0 + (i + CHUNK) / SAMPLE_RATE / speed - time.perf_counter()))
        session.flush()
        await session.drain(120)
        events = []
        while not session.out.empty():
            events.append(session.out.get_nowait())
        stats = {"frames_sent": session.frames_in, "frames_dropped": session.frames_dropped,
                 "timed_out": bool(session.tasks)}
        return events, stats
    finally:
        session.close()


# Purpose: Remove NaN and infinities from numeric measurements.
# Input: xs: iterable of numeric values.
# Output: NumPy array containing finite values only.
def finite_values(xs):
    values = np.asarray(list(xs), dtype=float)
    return values[np.isfinite(values)]


# Purpose: Calculate a finite-only percentile for a latency report.
# Input: xs: measurements; q: percentile from 0 to 100.
# Output: Float rounded to one decimal, or None for no finite values.
def pct(xs, q):
    values = finite_values(xs)
    return round(float(np.percentile(values, q)), 1) if values.size else None


# Purpose: Average one timing stage across events with finite measurements.
# Input: events: pipeline event dicts; stage: latency stage name.
# Output: Mean milliseconds rounded to one decimal, or None when unavailable.
def mean_ms(events, stage):
    values = finite_values(e.get("latency_ms", {}).get(stage) for e in events)
    return round(float(values.mean()), 1) if values.size else None


# Purpose: Summarize saved or current latency rows without rerunning inference.
# Input: rows: latency records or a DataFrame.
# Output: Dict of p50/p95 values for each supported metric; missing values become None.
def summarize_latency(rows):
    """Also accepts a saved latency DataFrame, allowing summaries without rerunning models."""
    df = pd.DataFrame(rows)
    return {m: {"p50": pct(df[m] if m in df else [], 50),
                "p95": pct(df[m] if m in df else [], 95)} for m in LATENCY_METRICS}


# Purpose: Classify one replay's speech, reply completion, and modality coverage.
# Input: clip_id: input identifier; events: collected session events; stats: frame counters.
# Output: Per-clip outcome dict including accepted/completed turns, failures, coverage, and frame
#          statistics.
def clip_outcome(clip_id, events, stats):
    finals = [e for e in events if e["type"] == "final_state"]
    errors = [e for e in events if e["type"] == "error"]
    done = {e["utterance_id"]: e for e in events if e["type"] == "response_done" and e.get("text", "").strip()}
    llm_errors = {e.get("utterance_id") for e in errors if e.get("where") == "llm"}
    accepted = {f["utterance_id"] for f in finals}
    accepted.update(e["utterance_id"] for e in events
                    if e["type"] == "vad" and e.get("state") == "speech_end")
    completed = sum(uid in done and uid not in llm_errors for uid in accepted)
    detected = bool(accepted)
    reply_complete = detected and completed == len(accepted) and not stats["timed_out"]
    status = "error" if errors else ("no_speech" if not detected else ("complete" if reply_complete else "incomplete"))
    return {"clip": str(clip_id), "status": status, "speech_detected": detected, "n_utterances": len(finals),
            "n_accepted_turns": len(accepted), "n_completed_replies": completed,
            "reply_complete": reply_complete, "errors": errors,
            "n_all_modalities": sum(set(f.get("fusion", {}).get("used", [])) >= {"audio", "text", "vision"}
                                    for f in finals), **stats}


# Purpose: Enforce the benchmark's requested audio-only or full multimodal mode.
# Input: engine: loaded Engine; audio_only: explicit reduced-mode flag.
# Output: None; raises if required components are not ready.
def require_mode(engine, audio_only):
    if not audio_only and not engine.info().get("multimodal_ready", False):
        raise RuntimeError("Multimodal benchmark requires loaded text, vision and local LLM models; "
                           "fix startup errors or explicitly use --audio-only")


# Purpose: Run timed local inference and save observed latency/resource evidence.
# Input: args: parsed benchmark options including backend, clips, timing, and output settings.
# Output: None; writes JSON/Markdown/CSV reports and cleans up Engine/sampler on exit.
async def main_async(args):
    if args.n <= 0 or not np.isfinite(args.speed) or args.speed <= 0:
        raise ValueError("--n and --speed must be positive; --speed must be finite")
    hw = hardware(args.backend)
    print(json.dumps(hw, indent=2, allow_nan=False))
    mem, engine = MemorySampler(), None
    mem.start()
    try:
        proc = psutil.Process()
        settings = Settings()
        if args.audio_only:
            settings.use_text = settings.use_vision = False
        else:
            settings.require_multimodal = True
        engine = (Engine(settings) if args.backend == "mlx" else
                  create_engine(settings, backend=args.backend, cpu_threads=args.cpu_threads))
        t = time.perf_counter()
        engine.load()
        require_mode(engine, args.audio_only)
        load_s = round(time.perf_counter() - t, 1)
        rss_after_load = proc.memory_info().rss
        clips = pick_clips(args.n, args.seed)
        rows, outcomes = [], []
        proc.cpu_percent(None)
        t_run = time.perf_counter()
        for k, c in clips.iterrows():
            wav, frames = np.zeros(0, dtype=np.float32), []
            stats = {"frames_sent": 0, "frames_dropped": 0, "timed_out": False}
            try:
                wav = load_audio(MELD_DIR / c.wav_path)
                frames = clip_frames(c.id) if engine.vision else []
                events, stats = await run_clip(engine, wav, args.speed, frames)
                if not args.audio_only and not frames:
                    events.append({"type": "error", "where": "video", "message": "No video frames available"})
            except Exception as ex:
                events = [{"type": "error", "where": "benchmark_input", "message": repr(ex)}]
            outcome = clip_outcome(c.id, events, stats)
            outcomes.append(outcome)
            faces = [e for e in events if e["type"] == "face_state"]
            finals = [e for e in events if e["type"] == "final_state"]
            dones = {e["utterance_id"]: e for e in events if e["type"] == "response_done"}
            for f in finals:
                d = dones.get(f["utterance_id"], {})
                partials = [e for e in events if e["type"] == "partial_state" and e.get("utterance_id") == f["utterance_id"]]
                rows.append({
                    "clip": c.id, "true": c.emotion, "pred": f["emotion"], "clip_s": round(wav.size / SAMPLE_RATE, 2),
                    "utt_s": f["duration_s"], "n_utts_in_clip": len(finals), "n_partials": len(partials),
                    "partial_emotion_ms": mean_ms(partials, "emotion"),
                    "final_emotion_ms": f["latency_ms"]["emotion"], "asr_ms": f["latency_ms"]["asr"],
                    "eos_to_state_ms": f["latency_ms"]["end_of_speech_to_state"],
                    "eos_to_first_token_ms": d.get("latency_ms", {}).get("end_of_speech_to_first_token"),
                    "eos_to_last_token_ms": d.get("latency_ms", {}).get("end_of_speech_to_last_token"),
                    "tokens": d.get("tokens"), "tokens_per_s": d.get("tokens_per_s"),
                    "token_count_source": d.get("token_count_source", "visible_stream_deltas"),
                    "fused": f["emotion"], "voice": f["modalities"]["voice"]["emotion"],
                    "words": (f["modalities"].get("words") or {}).get("emotion"),
                    "face": (f["modalities"].get("face") or {}).get("emotion"),
                    "agreement": (f.get("agreement") or {}).get("kind"),
                    "text_emotion_ms": f["latency_ms"].get("text_emotion"),
                    "face_frame_ms": mean_ms(faces, "vision"),
                    # Repeated on utterance rows for context; aggregate frame counts from outcomes only.
                    "frames_sent": stats["frames_sent"], "frames_dropped": stats["frames_dropped"],
                    "transcript": f["transcript"], "reply": d.get("text"),
                })
            print(f"[{k + 1}/{len(clips)}] {c.id:<14} true={c.emotion:<8} status={outcome['status']} "
                  f"utterances={len(finals)} completed_replies={outcome['n_completed_replies']}")
        cpu_avg = proc.cpu_percent(None)
        wall = time.perf_counter() - t_run
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["clip", "true", "pred"] + LATENCY_METRICS)
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = output_suffix(args.audio_only, args.backend)
        df.to_csv(output_dir / f"latency{suffix}.csv", index=False)
        summary = summarize_latency(df)
        result = {
            "hardware": hw, "n_clips_requested": args.n, "n_clips": int(len(clips)), "n_utterances": int(len(df)),
            "speed": args.speed, "end_of_turn_silence_ms": settings.end_silence_ms,
            "model_load_s_total": load_s, "model_load_s": engine.load_times,
            "peak_rss_gb": round(mem.peak_rss / 2 ** 30, 2), "rss_after_load_gb": round(rss_after_load / 2 ** 30, 2),
            "mlx_peak_gb": mlx_peak_gb(args.backend), "torch_mps_driver_gb": mps_driver_gb(args.backend),
            "backend": args.backend, "cpu_threads": args.cpu_threads if args.backend == "cpu" else None,
            "avg_process_cpu_percent": round(cpu_avg, 1), "wall_s": round(wall, 1),
            "latency_ms": summary, "engine": engine.info(), "mode": "audio_only" if args.audio_only else "multimodal",
            "measurement_definition": MEASUREMENT_DEFINITION, "clip_outcomes": outcomes,
            "n_clips_no_speech": sum(o["status"] == "no_speech" for o in outcomes),
            "n_clips_with_errors": sum(bool(o["errors"]) for o in outcomes),
            "n_clips_reply_complete": sum(o["reply_complete"] for o in outcomes),
            "n_utterances_all_modalities": sum(o["n_all_modalities"] for o in outcomes),
            "n_clips_all_modalities": sum(o["n_all_modalities"] > 0 for o in outcomes),
            "frames_sent_total": sum(o["frames_sent"] for o in outcomes),
            "frames_dropped_total": sum(o["frames_dropped"] for o in outcomes),
        }
        (output_dir / f"benchmark{suffix}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        names = {"partial_emotion_ms": "provisional CNN (per-utterance mean compute)",
                 "final_emotion_ms": "final emotion (CNN)", "face_frame_ms": "face analysis (per-clip mean compute)",
                 "text_emotion_ms": "words emotion (after Whisper)", "asr_ms": "Whisper transcript",
                 "eos_to_state_ms": "end of speech → final_state", "eos_to_first_token_ms": "end of speech → first visible reply delta",
                 "eos_to_last_token_ms": "end of speech → full reply", "tokens_per_s": "visible reply deltas / s"}
        mode = "audio only" if args.audio_only else "voice + words + face"
        lines = [f"[{mode}; backend={args.backend}] {len(df)} utterances from {len(clips)} MELD test clips, "
                 f"streamed at {args.speed:g}x. Latencies exclude {settings.end_silence_ms} ms of end silence.",
                 "", MEASUREMENT_DEFINITION["transport"] + ".", "", "| stage | p50 | p95 |", "|---|---|---|"]
        lines += [f"| {names[m]} | {summary[m]['p50']} | {summary[m]['p95']} |" for m in LATENCY_METRICS
                  if summary[m]["p50"] is not None]
        lines += ["", f"Clips with no detected speech: {result['n_clips_no_speech']}; "
                  f"with errors: {result['n_clips_with_errors']}; completed replies: {result['n_clips_reply_complete']}.",
                  f"Frames sent/dropped: {result['frames_sent_total']}/{result['frames_dropped_total']}.",
                  f"Hardware: {hw.get('cpu')} · {hw.get('ram_gb')} GB RAM · macOS {hw.get('macos')}",
                  f"Model load {load_s} s · peak RSS {result['peak_rss_gb']} GB · MLX peak {result['mlx_peak_gb']} GB · "
                  f"MPS driver {result['torch_mps_driver_gb']} GB · CPU {result['avg_process_cpu_percent']}% (100% = one core).",
                  MEASUREMENT_DEFINITION["resources"] + "."]
        (output_dir / f"benchmark{suffix}.md").write_text("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
    finally:
        mem.stop_flag = True
        mem.join(timeout=1)
        if engine is not None:
            engine.close()


# Purpose: Name benchmark outputs according to modality mode and backend.
# Input: audio_only: reduced-mode flag; backend: mlx or cpu.
# Output: Filename suffix identifying the measured configuration.
def output_suffix(audio_only, backend="mlx"):
    base = "_audio_only" if audio_only else "_multimodal"
    return base + ("_cpu" if backend == "cpu" else "")


# Purpose: Parse and validate benchmark counts, timing, backend, and thread limits.
# Input: argv: optional CLI argument list; None uses process arguments.
# Output: Validated argparse Namespace; invalid settings terminate with a parser error.
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--backend", choices=("mlx", "cpu"), default="mlx", help="explicit offline CPU or default MLX backend")
    ap.add_argument("--cpu-threads", type=int, choices=range(1, 5), default=4)
    ap.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--audio-only", action="store_true", help="benchmark the audio-only fallback")
    args = ap.parse_args(argv)
    if args.n <= 0 or not np.isfinite(args.speed) or args.speed <= 0:
        ap.error("--n and --speed must be positive; --speed must be finite")
    return args


# Purpose: Launch the asynchronous benchmark from the command line.
# Input: None explicitly; reads process arguments.
# Output: None; runs the benchmark event loop and saves its reports.
def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
