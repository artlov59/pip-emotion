"""Stream an audio/video file through the running server in real time and print what comes back.

    python -m app.replay --clip dia279_utt2               # MELD test clip: audio + video frames, shows ground truth
    python -m app.replay --clip dia279_utt2 --no-video    # audio only
    python -m app.replay my_recording.mov                 # any audio/video file

This is the reproducible "one input travelling through the system to both outputs" demo.
The full event trace is saved to results/traces/<name>.json.
"""
import argparse
import asyncio
import json
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

from app.config import MELD_DIR, RESULTS_DIR, SAMPLE_RATE
from app.features import load_audio

CHUNK = SAMPLE_RATE // 10  # 100 ms
KIND_AUDIO, KIND_JPEG = b"\x01", b"\x02"


# Purpose: Sample and encode video frames for a browser-like replay.
# Input: path: media Path; fps: sample rate; max_side: maximum image dimension in pixels.
# Output: List of (timestamp_seconds, jpeg_bytes) pairs; [] for media without usable video.
def video_frames(path: Path, fps: float = 5.0, max_side: int = 480):
    """[(t_s, jpeg_bytes)] sampled at ~fps from a video file; [] for audio-only files."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    if not src_fps or src_fps <= 0:
        cap.release()
        return []
    step, out, i = max(1, int(round(src_fps / fps))), [], 0
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                k = min(1.0, max_side / max(h, w))
                if k < 1:
                    frame = cv2.resize(frame, (int(w * k), int(h * k)))
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    out.append((i / src_fps, jpg.tobytes()))
        i += 1
    cap.release()
    return out


# Purpose: Format the highest emotion probabilities for terminal output.
# Input: probs: emotion-to-probability dict; k: number to display.
# Output: Space-separated emotion/probability string.
def fmt_probs(probs: dict, k: int = 3) -> str:
    return "  ".join(f"{e} {p:.2f}" for e, p in sorted(probs.items(), key=lambda kv: -kv[1])[:k])


# Purpose: Format one modality's emotion result for terminal output.
# Input: r: emotion reading dict or None.
# Output: Readable label/confidence/probability string, or 'n/a'.
def fmt_reading(r):
    return "n/a" if not r else f"{r['emotion']:<8} {r['confidence']:.2f}  [{fmt_probs(r['probs'])}]"


class ReplayProgress:
    """Track accepted turns, including ones still waiting for their final state."""

    # Purpose: Track accepted turns, completion, failures, and flush acknowledgement.
    # Input: None.
    # Output: None; initializes turn-ID sets and acknowledgement state.
    def __init__(self):
        self.expected, self.dones, self.failed = set(), set(), set()
        self.flushed = False

    # Purpose: Update replay completion tracking from a server event.
    # Input: event: decoded WebSocket event dict.
    # Output: None; records accepted/final/completed/failed turns and flush acknowledgement.
    def observe(self, event):
        kind, uid = event["type"], event.get("utterance_id")
        if kind == "vad" and event.get("state") == "speech_end":
            self.expected.add(uid)
        elif kind == "final_state":
            self.expected.add(uid)
        elif kind == "response_done":
            self.dones.add(uid)
        elif kind == "error" and event.get("where") == "final" and uid is not None:
            self.failed.add(uid)
        elif kind == "flushed":
            self.flushed = True
            # New replay connections start with turn 1; discarded blips use no ID.
            self.expected.update(range(1, int(uid or 0) + 1))

    # Purpose: Check whether every accepted turn has reached a terminal outcome.
    # Input: None explicitly; reads tracked turn sets and flush acknowledgement.
    # Output: Boolean; true only after flush acknowledgement and all expected turns finish or fail.
    @property
    def complete(self):
        return self.flushed and self.expected <= self.dones | self.failed


# Purpose: Replay one media input over WebSocket and collect its interaction events.
# Input: path: media file; url: WebSocket endpoint; speed: playback multiplier; tail_s: silence tail; quiet:
#          output flag; truth: optional labels; use_video: frame flag; response_timeout: seconds.
# Output: Awaited tuple (events, ready_event); streams timed audio/video and waits for terminal turn
#          outcomes.
async def run(path: Path, url: str, speed: float, tail_s: float, quiet: bool, truth: dict = None,
              use_video: bool = True, response_timeout: float = 60.0):
    audio = load_audio(path)
    frames = video_frames(path) if use_video else []
    duration = audio.size / SAMPLE_RATE
    print(f"input: {path}  ({duration:.2f} s, {len(frames)} video frames)")
    if truth:
        print(f"MELD ground truth: {truth['emotion']}  |  \"{truth['utterance']}\"")
    events, t_start = [], None
    progress = ReplayProgress()
    completed = asyncio.Event()

    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready

        # Purpose: Receive, record, and optionally display events during replay.
        # Input: None explicitly; captures the WebSocket, event list, progress tracker, and display
        #          options.
        # Output: None; updates replay events/readiness until reception ends or is cancelled.
        async def receiver():
            async for raw in ws:
                ev = json.loads(raw)
                ev["client_t_s"] = round(time.perf_counter() - t_start, 3)
                events.append(ev)
                progress.observe(ev)
                if progress.complete:
                    completed.set()
                t, typ = ev["client_t_s"], ev["type"]
                if typ == "vad":
                    print(f"[{t:6.2f}s] vad       {ev['state']}" + (f" (utterance {ev.get('utterance_id')})" if ev.get("utterance_id") else ""))
                elif typ == "partial_state" and not quiet:
                    print(f"[{t:6.2f}s] partial   {ev['emotion']:<8} {ev['confidence']:.2f}   [{fmt_probs(ev['probs'])}]  "
                          f"window {ev['window_s']} s, {ev['latency_ms']['emotion']} ms"
                          + (f"  | face {ev['modalities']['face']['emotion']}" if (ev.get("modalities") or {}).get("face") else ""))
                elif typ == "final_state":
                    mods = ev.get("modalities", {})
                    print(f"[{t:6.2f}s] FINAL     {fmt_reading(ev)}   (fused: {', '.join(ev.get('fusion', {}).get('used', ['audio']))})")
                    print(f"           transcript: \"{ev['transcript']}\"")
                    for name in ("voice", "words", "face"):
                        if name in mods:
                            extra = f"  ({mods[name]['frames']} frames)" if mods[name] and "frames" in mods[name] else ""
                            print(f"           {name:<6} {fmt_reading(mods[name])}{extra}")
                    if ev.get("agreement"):
                        print(f"           agreement: {ev['agreement']['kind']}: {ev['agreement'].get('note', '')}")
                    print(f"           tags: {ev['tags']}   delivery: {ev['delivery']}")
                    print(f"           latency: {ev['latency_ms']}")
                    print(f"[{t:6.2f}s] REPLY     ", end="", flush=True)
                elif typ == "response_delta":
                    print(ev["text"], end="", flush=True)
                elif typ == "response_done":
                    print(f"\n           reply latency: {ev['latency_ms']}  ({ev['tokens']} stream deltas, {ev['tokens_per_s']} deltas/s)")
                elif typ == "error":
                    print(f"[{t:6.2f}s] ERROR {ev}")

        t_start = time.perf_counter()
        recv_task = asyncio.create_task(receiver())
        pad = np.zeros(int(tail_s * SAMPLE_RATE), dtype=np.float32)
        stream = np.concatenate([audio, pad])
        fi = 0
        for i in range(0, stream.size, CHUNK):
            while fi < len(frames) and frames[fi][0] <= (i + CHUNK) / SAMPLE_RATE:  # frames due by this chunk
                await ws.send(KIND_JPEG + frames[fi][1])
                fi += 1
            chunk = np.clip(stream[i:i + CHUNK], -1, 1)
            await ws.send(KIND_AUDIO + (chunk * 32767).astype("<i2").tobytes())
            target = t_start + (i + CHUNK) / SAMPLE_RATE / speed
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
        await ws.send(json.dumps({"type": "flush"}))
        completion_task = asyncio.create_task(completed.wait())
        try:
            finished, _ = await asyncio.wait(
                {completion_task, recv_task}, timeout=response_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not finished:
                pending = sorted(progress.expected - progress.dones - progress.failed)
                message = (f"Replay timed out after {response_timeout:g}s; pending turns: {pending}. "
                           f"Flush acknowledged: {progress.flushed}. Restart an older server if needed.")
                print(message)
                events.append({"type": "error", "where": "replay", "message": message})
            elif recv_task in finished and not progress.complete:
                # Surface network/receiver exceptions instead of reporting a successful replay.
                recv_task.result()
                events.append({"type": "error", "where": "replay",
                               "message": "Connection closed before all accepted turns completed"})
            elif not progress.expected:
                print("no speech detected (server acknowledged flush)")
        finally:
            for task in (completion_task, recv_task):
                task.cancel()
            for task in (completion_task, recv_task):
                with suppress(asyncio.CancelledError):
                    await task
    return events, ready


# Purpose: Parse replay options and launch one media replay.
# Input: None explicitly; reads command-line media/server/timing options.
# Output: None; prints interaction results and optionally saves its trace.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="audio or video file")
    ap.add_argument("--clip", help="MELD test clip id, e.g. dia3_utt2")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 streams faster than real time")
    ap.add_argument("--tail", type=float, default=1.0, help="seconds of silence appended (lets VAD close the turn)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for accepted turns after input")
    ap.add_argument("--quiet", action="store_true", help="hide partial updates")
    ap.add_argument("--no-video", action="store_true", help="send audio only, even for a video file")
    args = ap.parse_args()

    if args.speed <= 0 or args.tail < 0 or args.timeout <= 0:
        ap.error("speed and timeout must be positive; tail must be nonnegative")

    truth = None
    if args.clip:
        df = pd.read_csv(MELD_DIR / "test.csv").set_index("id")
        row = df.loc[args.clip]
        truth = {"id": args.clip, "emotion": row.emotion, "utterance": row.utterance}
        path = MELD_DIR / row.wav_path
        if not args.no_video:
            from app.config import DATA_DIR
            from training.prepare_meld import find_video_dir
            vdir = find_video_dir(DATA_DIR / "MELD.Raw", "test") if (DATA_DIR / "MELD.Raw").exists() else None
            if vdir and (vdir / f"{args.clip}.mp4").exists():
                path = vdir / f"{args.clip}.mp4"
    elif args.path:
        path = Path(args.path)
    else:
        ap.error("give a file path or --clip")

    events, ready = asyncio.run(run(path, args.url, args.speed, args.tail, args.quiet, truth, not args.no_video, args.timeout))
    out = RESULTS_DIR / "traces" / f"{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"input": str(path), "ground_truth": truth, "ready": ready, "events": events}, indent=2))
    print(f"\ntrace saved: {out}")
    if any(e["type"] == "error" for e in events):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
