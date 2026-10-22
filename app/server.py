"""Local server: static UI + WebSocket streaming endpoint + offline /api/analyze.

    python -m app.server                 # http://127.0.0.1:8000
    python -m app.server --no-llm        # template replies (debug)

WebSocket protocol (/ws)
  client -> server : binary frames = little-endian int16 mono PCM @ 16 kHz (100 ms recommended)
                     text frames   = {"type": "flush"} end current utterance now
                                     {"type": "reset"} clear conversation history
  server -> client : JSON events: ready, vad, partial_state, final_state,
                     response_delta, response_done, error   (see README "Output format")
"""
import argparse
import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import EMOTIONS, MELD_DIR, ROOT, Settings
from app.features import load_audio
from app.pipeline import Engine, Session

STATIC = ROOT / "app" / "static"
engine: Engine = None


@asynccontextmanager
async def lifespan(_app):
    global engine
    if engine is None:
        engine = Engine(Settings())
    await asyncio.to_thread(engine.load)
    print("[server] ready", flush=True)
    yield
    engine.close()


app = FastAPI(title="Pip: real-time audio emotion prototype", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, **engine.info()}


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
                       "duration_s": round(float(r.duration_s), 2), "url": f"/api/clips/test/{r.id}.wav"}
                      for r in out.itertuples()]}


@app.get("/api/clips/{split}/{clip_id}.wav")
def clip_audio(split: str, clip_id: str):
    if split not in {"train", "dev", "test"} or not clip_id.replace("_", "").isalnum():
        raise HTTPException(404)
    path = MELD_DIR / "wav" / split / f"{clip_id}.wav"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Non-streaming debug path: whole file = one utterance. Returns final_state + full reply."""
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
    try:
        audio = await asyncio.to_thread(load_audio, tmp.name)
    except Exception as ex:
        raise HTTPException(400, f"could not decode audio: {ex}")
    finally:
        os.unlink(tmp.name)
    session = Session(engine)
    await session.process_utterance(1, audio)
    events = []
    while not session.out.empty():
        events.append(session.out.get_nowait())
    final = next((e for e in events if e["type"] == "final_state"), None)
    done = next((e for e in events if e["type"] == "response_done"), None)
    return {"final_state": final, "response": done, "errors": [e for e in events if e["type"] == "error"]}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(engine)

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
            if msg.get("bytes") is not None:
                pcm = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                session.feed(pcm)
            elif msg.get("text"):
                cmd = json.loads(msg["text"])
                if cmd.get("type") == "flush":
                    session.flush()
                elif cmd.get("type") == "reset":
                    session.reset()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        session.close()
        sender.cancel()


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint", help="emotion CNN checkpoint (default checkpoints/emotion_cnn.pt)")
    ap.add_argument("--no-llm", action="store_true", help="template replies instead of the LLM (debug)")
    args = ap.parse_args()
    settings = Settings()
    if args.checkpoint:
        settings.emotion_checkpoint = Path(args.checkpoint)
    if args.no_llm:
        settings.use_llm = False
    engine = Engine(settings)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws_max_size=2 ** 22)


if __name__ == "__main__":
    main()
