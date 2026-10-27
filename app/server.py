"""Local server: static UI + WebSocket streaming endpoint.

    python -m app.server                 # http://127.0.0.1:8000
    python -m app.server --no-llm        # template replies (debug)

WebSocket protocol (/ws)
  client -> server : binary frames, first byte = kind:
                       0x01 + little-endian int16 mono PCM @ 16 kHz (100 ms chunks recommended)
                       0x02 + one JPEG image (webcam / video frame, ~5 per second)
                     text frames = {"type": "flush"} end current utterance now
                                   {"type": "reset"} clear conversation history
  server -> client : JSON events: ready, vad, partial_state, face_state, final_state,
                     response_delta, response_done, flushed, error   (see README "Output format")
"""
import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import DATA_DIR, EMOTIONS, MELD_DIR, ROOT, Settings
from app.backends import create_engine
from app.pipeline import Engine, Session
from app.protocol import handle_message

STATIC = ROOT / "app" / "static"
engine: Engine = None
_video_dirs = {}


# Purpose: Locate a prepared MELD clip's original video.
# Input: split: MELD split name; clip_id: clip identifier.
# Output: Existing MP4 Path, or None when the source video is absent.
def video_path(split: str, clip_id: str):
    """MELD source .mp4 for a clip (None if MELD.Raw was not extracted)."""
    if split not in _video_dirs:
        from training.prepare_meld import find_video_dir
        root = DATA_DIR / "MELD.Raw"
        _video_dirs[split] = find_video_dir(root, split) if root.exists() else None
    d = _video_dirs[split]
    p = d / f"{clip_id}.mp4" if d else None
    return p if p and p.exists() else None


# Purpose: Load shared inference models at startup and release workers at shutdown.
# Input: _app: FastAPI application; uses configured Engine/backend and readiness requirements.
# Output: Async context-manager yield of None; startup failures propagate after cleanup.
@asynccontextmanager
async def lifespan(_app):
    global engine
    if engine is None:
        engine = Engine(Settings())
    try:
        await asyncio.to_thread(engine.load)
        if engine.settings.require_multimodal and not engine.info()["multimodal_ready"]:
            raise RuntimeError(
                "Required multimodal mode needs audio, text, vision and the local LLM. "
                f"Model errors: {engine.errors}. See README model setup instructions."
            )
        print("[server] ready", flush=True)
        yield
    finally:
        engine.close()


app = FastAPI(title="Pip: real-time multimodal emotion prototype", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# Purpose: Serve the browser interaction page.
# Input: None.
# Output: FileResponse containing static/index.html.
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


# Purpose: Expose current local inference readiness and model information.
# Input: None explicitly; uses the loaded shared Engine.
# Output: Dict containing ok and Engine.info() fields.
@app.get("/api/health")
def health():
    return {"ok": True, **engine.info()}


# Purpose: Choose a reproducible emotion-balanced MELD test demo sample.
# Input: per_emotion: clips per label; seed: sample seed.
# Output: Dict containing clip metadata/media URLs, or an empty list with a preparation note.
@app.get("/api/clips")
def clips(per_emotion: int = 5, seed: int = 7):
    """A fixed, emotion-balanced sample of MELD *test* clips for the replay demo."""
    csv = MELD_DIR / "test.csv"
    if not csv.exists():
        return {"clips": [], "note": "MELD not prepared (run training.prepare_meld)"}
    df = pd.read_csv(csv)
    df = df[(df.status == "ok") & df.duration_s.between(1.5, 8.0)]
    picks = []
    for emo in EMOTIONS:
        sub = df[df.emotion == emo]
        picks.append(sub.sample(min(per_emotion, len(sub)), random_state=seed))
    out = pd.concat(picks)
    return {"clips": [{"id": r.id, "emotion": r.emotion, "utterance": r.utterance, "speaker": r.Speaker,
                       "duration_s": round(float(r.duration_s), 2), "url": f"/api/clips/test/{r.id}.wav",
                       "video_url": f"/api/clips/test/{r.id}.mp4" if video_path("test", r.id) else None}
                      for r in out.itertuples()]}


# Purpose: Serve a validated MELD source video.
# Input: split: train/dev/test; clip_id: allowed clip identifier.
# Output: MP4 FileResponse; raises an HTTP error for invalid or unavailable media.
@app.get("/api/clips/{split}/{clip_id}.mp4")
def clip_video(split: str, clip_id: str):
    if split not in {"train", "dev", "test"} or not clip_id.replace("_", "").isalnum():
        raise HTTPException(404)
    path = video_path(split, clip_id)
    if path is None:
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")


# Purpose: Serve a validated prepared MELD audio clip.
# Input: split: train/dev/test; clip_id: allowed clip identifier.
# Output: WAV FileResponse; raises an HTTP error for invalid or unavailable media.
@app.get("/api/clips/{split}/{clip_id}.wav")
def clip_audio(split: str, clip_id: str):
    if split not in {"train", "dev", "test"} or not clip_id.replace("_", "").isalnum():
        raise HTTPException(404)
    path = MELD_DIR / "wav" / split / f"{clip_id}.wav"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


# Purpose: Run one conversation over a media/event WebSocket.
# Input: ws: FastAPI WebSocket connection.
# Output: None; accepts input packets, sends session events, and cleans up the session on disconnect.
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(engine)

    # Purpose: Forward queued session events to the connected browser.
    # Input: None explicitly; captures the Session and WebSocket.
    # Output: None; sends JSON events until cancelled or disconnected.
    async def pump():
        while True:
            event = await session.out.get()
            await ws.send_text(json.dumps(event))

    sender = asyncio.create_task(pump())
    session.emit(session.ready_event())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            handle_message(session, data=msg.get("bytes"), text=msg.get("text"))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        session.close()
        sender.cancel()
        try:
            await sender
        except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            pass


# Purpose: Configure and launch the local FastAPI server.
# Input: None explicitly; reads CLI host/port/backend/model/readiness options.
# Output: None; starts Uvicorn using the requested settings.
def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", choices=["mlx", "cpu"], default="mlx",
                    help="local inference backend; CPU uses existing cached weights with FP32 computation")
    ap.add_argument("--cpu-threads", type=int, choices=range(1, 5), default=4)
    ap.add_argument("--checkpoint", help="emotion CNN checkpoint (default checkpoints/emotion_cnn.pt)")
    ap.add_argument("--no-llm", action="store_true", help="template replies instead of the LLM (debug)")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--audio-only", action="store_true", help="disable text and vision (diagnostic fallback)")
    modes.add_argument("--require-multimodal", action="store_true",
                       help="fail startup unless audio, text, vision and local LLM are available")
    args = ap.parse_args()
    settings = Settings()
    if args.checkpoint:
        settings.emotion_checkpoint = Path(args.checkpoint)
    if args.no_llm:
        settings.use_llm = False
    if args.audio_only:
        settings.use_text = settings.use_vision = False
        settings.require_multimodal = False
    if args.require_multimodal:
        settings.require_multimodal = True
    engine = create_engine(settings, args.backend, args.cpu_threads)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws_max_size=2 ** 22)


if __name__ == "__main__":
    main()
