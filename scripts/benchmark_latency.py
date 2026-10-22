"""Latency + resource benchmark: replay MELD test clips through the real pipeline in real time.

    python -m scripts.benchmark_latency --n 35          # 5 clips per emotion
    python -m scripts.benchmark_latency --n 35 --speed 4  # faster (post-utterance latencies unaffected)

Runs in-process (no WebSocket) with every model loaded exactly as the server loads them.
Writes results/latency.csv, results/benchmark.json and results/benchmark.md.
"""
import argparse
import asyncio
import json
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

CHUNK = SAMPLE_RATE // 10


class MemorySampler(threading.Thread):
    def __init__(self, every=0.1):
        super().__init__(daemon=True)
        self.proc, self.every, self.peak_rss, self.stop_flag = psutil.Process(), every, 0, False

    def run(self):
        while not self.stop_flag:
            self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            time.sleep(self.every)


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def hardware():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "macos": platform.mac_ver()[0] or None, "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1), "cpu_cores": psutil.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["mps_available"] = torch.backends.mps.is_available()
    except Exception:
        pass
    try:
        import mlx.core as mx
        info["mlx"] = mx.__version__
    except Exception:
        pass
    return info


def mlx_peak_gb():
    try:
        import mlx.core as mx
        fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        return round(fn() / 2 ** 30, 2)
    except Exception:
        return None


def mps_driver_gb():
    try:
        import torch
        return round(torch.mps.driver_allocated_memory() / 2 ** 30, 2)
    except Exception:
        return None


def pick_clips(n, seed):
    df = pd.read_csv(MELD_DIR / "test.csv")
    df = df[(df.status == "ok") & df.duration_s.between(1.5, 8.0)]
    per = max(1, n // len(EMOTIONS))
    return pd.concat([df[df.emotion == e].sample(min(per, (df.emotion == e).sum()), random_state=seed)
                      for e in EMOTIONS]).reset_index(drop=True)


async def run_clip(engine, wav, speed):
    session = Session(engine)
    stream = np.concatenate([wav, np.zeros(SAMPLE_RATE, dtype=np.float32)])  # 1 s tail closes the turn
    t0 = time.perf_counter()
    for i in range(0, stream.size, CHUNK):
        session.feed(stream[i:i + CHUNK])
        await asyncio.sleep(max(0.0, t0 + (i + CHUNK) / SAMPLE_RATE / speed - time.perf_counter()))
    session.flush()
    await session.drain(120)
    events = []
    while not session.out.empty():
        events.append(session.out.get_nowait())
    return events


def pct(xs, q):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return round(float(np.percentile(xs, q)), 1) if xs else None


async def main_async(args):
    hw = hardware()
    print(json.dumps(hw, indent=2))
    mem = MemorySampler()
    mem.start()
    proc = psutil.Process()
    engine = Engine(Settings())
    t = time.perf_counter()
    engine.load()
    load_s = round(time.perf_counter() - t, 1)
    rss_after_load = proc.memory_info().rss
    clips = pick_clips(args.n, args.seed)
    rows = []
    proc.cpu_percent(None)
    t_run = time.perf_counter()
    for k, c in clips.iterrows():
        wav = load_audio(MELD_DIR / c.wav_path)
        events = await run_clip(engine, wav, args.speed)
        finals = [e for e in events if e["type"] == "final_state"]
        dones = {e["utterance_id"]: e for e in events if e["type"] == "response_done"}
        partials = [e for e in events if e["type"] == "partial_state"]
        for f in finals:
            d = dones.get(f["utterance_id"], {})
            rows.append({
                "clip": c.id, "true": c.emotion, "pred": f["emotion"], "clip_s": round(wav.size / SAMPLE_RATE, 2),
                "utt_s": f["duration_s"], "n_utts_in_clip": len(finals), "n_partials": len(partials),
                "partial_emotion_ms": round(float(np.mean([p["latency_ms"]["emotion"] for p in partials])), 1) if partials else None,
                "final_emotion_ms": f["latency_ms"]["emotion"], "asr_ms": f["latency_ms"]["asr"],
                "eos_to_state_ms": f["latency_ms"]["end_of_speech_to_state"],
                "eos_to_first_token_ms": d.get("latency_ms", {}).get("end_of_speech_to_first_token"),
                "eos_to_last_token_ms": d.get("latency_ms", {}).get("end_of_speech_to_last_token"),
                "tokens": d.get("tokens"), "tokens_per_s": d.get("tokens_per_s"),
                "transcript": f["transcript"], "reply": d.get("text"),
            })
        last = rows[-1] if rows and rows[-1]["clip"] == c.id else None
        print(f"[{k + 1}/{len(clips)}] {c.id:<14} true={c.emotion:<8} "
              + (f"pred={last['pred']:<8} state={last['eos_to_state_ms']} ms first_tok={last['eos_to_first_token_ms']} ms"
                 if last else "no utterance detected"))
    cpu_avg = proc.cpu_percent(None)
    wall = time.perf_counter() - t_run
    mem.stop_flag = True
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "latency.csv", index=False)

    metrics = ["partial_emotion_ms", "final_emotion_ms", "asr_ms", "eos_to_state_ms",
               "eos_to_first_token_ms", "eos_to_last_token_ms", "tokens_per_s"]
    summary = {m: {"p50": pct(df[m].tolist(), 50), "p95": pct(df[m].tolist(), 95)} for m in metrics}
    s = engine.settings
    result = {
        "hardware": hw, "n_clips": int(len(clips)), "n_utterances": int(len(df)), "speed": args.speed,
        "end_of_turn_silence_ms": s.end_silence_ms,
        "model_load_s_total": load_s, "model_load_s": engine.load_times,
        "peak_rss_gb": round(mem.peak_rss / 2 ** 30, 2), "rss_after_load_gb": round(rss_after_load / 2 ** 30, 2),
        "mlx_peak_gb": mlx_peak_gb(), "torch_mps_driver_gb": mps_driver_gb(),
        "avg_process_cpu_percent": round(cpu_avg, 1), "wall_s": round(wall, 1),
        "latency_ms": summary, "engine": engine.info(),
    }
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(result, indent=2, default=str))
    names = {"partial_emotion_ms": "provisional emotion (CNN, 3 s window)", "final_emotion_ms": "final emotion (CNN)",
             "asr_ms": "Whisper transcript", "eos_to_state_ms": "end of speech → final_state",
             "eos_to_first_token_ms": "end of speech → first reply token",
             "eos_to_last_token_ms": "end of speech → full reply", "tokens_per_s": "LLM tokens / s"}
    lines = [f"{len(df)} utterances from {len(clips)} MELD test clips, streamed in real time"
             f"{'' if args.speed == 1 else f' (x{args.speed})'}. End-of-speech is declared after "
             f"{s.end_silence_ms} ms of silence; the latencies below are measured *after* that point.", "",
             "| stage | p50 | p95 |", "|---|---|---|"]
    lines += [f"| {names[m]} | {summary[m]['p50']} | {summary[m]['p95']} |" for m in metrics]
    lines += ["", f"Hardware: {hw.get('cpu')} · {hw.get('ram_gb')} GB RAM · macOS {hw.get('macos')}  ",
              f"Model load: {load_s} s · peak process RSS {result['peak_rss_gb']} GB · "
              f"MLX peak {result['mlx_peak_gb']} GB · torch-MPS {result['torch_mps_driver_gb']} GB · "
              f"avg CPU {result['avg_process_cpu_percent']}% (100% = one core)"]
    (RESULTS_DIR / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    engine.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
