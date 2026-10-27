---
paths:
  - "app/**"
  - "model.py"
  - "tests/test_pipeline.py"
  - "tests/test_session_regressions.py"
  - "tests/test_protocol.py"
---

# Runtime rules

## Threads and latency

- The asyncio loop only runs VAD and bookkeeping. Blocking work goes through the Engine pools:
  `run_torch` (AudioCNN, text model), `run_vision` (faces), `run_mlx` (Whisper and Qwen; MLX objects
  must be created and used on that one thread).
- Every async result checks the session `epoch` (and `speech_generation` for partials) before it is
  emitted, so a reset never leaks stale events. Keep that pattern in new code.
- Video frames are dropped, never queued, while the vision worker is busy.
- Latency targets after end-of-speech: state ~0.5 s, first reply words ~1 s. Measured on MLX
  (M5 Pro): 0.11 s and 0.23 s p50. Re-benchmark after any change on the hot path.

## Protocol

- Binary frames: `0x01` + little-endian int16 mono PCM at 16 kHz; `0x02` + JPEG.
  Control: `{"type": "flush"}`, `{"type": "reset"}`.
- Server events: `ready`, `vad`, `partial_state`, `face_state`, `final_state`, `response_delta`,
  `response_done`, `flushed`, `error`. Add fields freely; never rename or remove existing ones,
  because `app/static/app.js`, `app/replay.py` and the benchmark read them.

## Models and fusion

- `fusion.fuse` is a weighted sum of log-probabilities; weights come from
  `results/fusion_weights.json` per available-input subset. Changing them changes locked results.
- `fusion.agreement` uses dev-fitted temperatures and margin from `results/agreement_calibration.json`.
  It must never change the fused emotion.
- Audio features must come from `app/features.py`; checkpoints store and verify `FEATURE_CONFIG`.
- Optional inputs degrade gracefully (no face, empty transcript), except with
  `--require-multimodal`, which must fail loudly at startup.

## Reply prompt (`app/llm.py`)

- Pip speaks directly to the person ("you"), one or two short sentences, no labels or percentages.
- Readings are hints. The words-vs-delivery note appears only when the delivery leads.
- Prompt changes can't be scored on MELD here; check them by talking to the app and say so.
