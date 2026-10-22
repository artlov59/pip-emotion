"""Stream an audio/video file through the running server in real time and print what comes back.

    python -m app.replay data/meld/wav/test/dia3_utt2.wav
    python -m app.replay --clip dia3_utt2                 # MELD test clip by id (shows ground truth)
    python -m app.replay my_recording.m4a --speed 2

This is the reproducible "one input travelling through the system to both outputs" demo.
The full event trace is saved to results/traces/<name>.json.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

from app.config import MELD_DIR, RESULTS_DIR, SAMPLE_RATE
from app.features import load_audio

CHUNK = SAMPLE_RATE // 10  # 100 ms


def fmt_probs(probs: dict, k: int = 3) -> str:
    return "  ".join(f"{e} {p:.2f}" for e, p in sorted(probs.items(), key=lambda kv: -kv[1])[:k])


async def run(path: Path, url: str, speed: float, tail_s: float, quiet: bool, truth: dict = None):
    audio = load_audio(path)
    duration = audio.size / SAMPLE_RATE
    print(f"input: {path}  ({duration:.2f} s)")
    if truth:
        print(f"MELD ground truth: {truth['emotion']}  |  \"{truth['utterance']}\"")
    events, t_start = [], None
    finals, dones = set(), set()

    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready

        async def receiver():
            async for raw in ws:
                ev = json.loads(raw)
                ev["client_t_s"] = round(time.perf_counter() - t_start, 3)
                events.append(ev)
                t, typ = ev["client_t_s"], ev["type"]
                if typ == "vad":
                    print(f"[{t:6.2f}s] vad       {ev['state']}" + (f" (utterance {ev.get('utterance_id')})" if ev.get("utterance_id") else ""))
                elif typ == "partial_state" and not quiet:
                    print(f"[{t:6.2f}s] partial   {ev['emotion']:<8} {ev['confidence']:.2f}   [{fmt_probs(ev['probs'])}]  "
                          f"window {ev['window_s']} s, {ev['latency_ms']['emotion']} ms")
                elif typ == "final_state":
                    finals.add(ev["utterance_id"])
                    print(f"[{t:6.2f}s] FINAL     {ev['emotion']:<8} {ev['confidence']:.2f}   [{fmt_probs(ev['probs'])}]")
                    print(f"           transcript: \"{ev['transcript']}\"")
                    print(f"           tags: {ev['tags']}   delivery: {ev['delivery']}")
                    print(f"           latency: {ev['latency_ms']}")
                    print(f"[{t:6.2f}s] REPLY     ", end="", flush=True)
                elif typ == "response_delta":
                    print(ev["text"], end="", flush=True)
                elif typ == "response_done":
                    dones.add(ev["utterance_id"])
                    print(f"\n           reply latency: {ev['latency_ms']}  ({ev['tokens']} tokens, {ev['tokens_per_s']} tok/s)")
                elif typ == "error":
                    print(f"[{t:6.2f}s] ERROR {ev}")

        t_start = time.perf_counter()
        recv_task = asyncio.create_task(receiver())
        pad = np.zeros(int(tail_s * SAMPLE_RATE), dtype=np.float32)
        stream = np.concatenate([audio, pad])
        for i in range(0, stream.size, CHUNK):
            chunk = np.clip(stream[i:i + CHUNK], -1, 1)
            await ws.send((chunk * 32767).astype("<i2").tobytes())
            target = t_start + (i + CHUNK) / SAMPLE_RATE / speed
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
        await ws.send(json.dumps({"type": "flush"}))
        sent_at = time.perf_counter()
        # wait until every final_state has its response_done (or nothing was detected)
        while time.perf_counter() - sent_at < 60:
            await asyncio.sleep(0.2)
            if finals and dones >= finals:
                break
            if not finals and time.perf_counter() - sent_at > 3:
                print("no speech detected")
                break
        recv_task.cancel()
    return events, ready


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="audio or video file")
    ap.add_argument("--clip", help="MELD test clip id, e.g. dia3_utt2")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 streams faster than real time")
    ap.add_argument("--tail", type=float, default=1.0, help="seconds of silence appended (lets VAD close the turn)")
    ap.add_argument("--quiet", action="store_true", help="hide partial updates")
    args = ap.parse_args()

    truth = None
    if args.clip:
        df = pd.read_csv(MELD_DIR / "test.csv").set_index("id")
        row = df.loc[args.clip]
        truth = {"id": args.clip, "emotion": row.emotion, "utterance": row.utterance}
        path = MELD_DIR / row.wav_path
    elif args.path:
        path = Path(args.path)
    else:
        ap.error("give a file path or --clip")

    events, ready = asyncio.run(run(path, args.url, args.speed, args.tail, args.quiet, truth))
    out = RESULTS_DIR / "traces" / f"{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"input": str(path), "ground_truth": truth, "ready": ready, "events": events}, indent=2))
    print(f"\ntrace saved: {out}")


if __name__ == "__main__":
    main()
