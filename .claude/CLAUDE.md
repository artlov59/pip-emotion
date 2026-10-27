# Pip: multimodal emotion prototype (MELD)

Local, real-time prototype for a character robot: microphone (+ optional camera) in, a structured
MELD emotion state plus a short streamed reply out. Primary track is **audio**; words (Whisper ->
DistilRoBERTa) and face (YuNet -> HSEmotion) are the extension. Built for a take-home challenge;
the brief and a requirement-to-evidence map are in `.claude/specs/challenge-brief.md`.

<!-- Maintainers: keep this file under ~150 lines. Procedures belong in .claude/skills/,
     path-specific rules in .claude/rules/, background in .claude/specs/. -->

## Commands

- Setup: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements-verified.txt`
- Models not in git: `python -m scripts.download_models` then `python -m scripts.cache_models --output results/model_cache_manifest.json`
- Run: `python -m app.server --require-multimodal` (MLX on the Apple GPU), open http://127.0.0.1:8000
- CPU fallback: `python -m app.server --backend cpu --require-multimodal`
- Replay a file through a running server: `python -m app.replay path/to/clip.mp4`
- All tests: `python -m unittest discover -s tests && python -m tests.test_pipeline && node tests/test_frontend.js`
- Latency: `python -m scripts.benchmark_latency --n 35 --output-dir results/benchmark_mlx`
- Parameter budget: `python -m scripts.count_params_offline --require-multimodal`
- Re-score saved predictions: `python -m scripts.report_baselines`

## Layout

- `app/` runtime. `pipeline.py` owns sessions and worker threads; `fusion.py` holds fusion and the
  words-vs-delivery check; one module per model (`emotion.py`, `text_emotion.py`, `vision.py`,
  `asr.py`, `llm.py`); `cpu_asr.py`/`cpu_llm.py` are the CPU backend; `static/` is the browser UI.
- `model.py` AudioCNN. `training/` data prep, training, locked evaluation. `scripts/` tooling.
- `results/` is evidence: every number in README.md must come from a file here.
- `data/` (MELD, ~30 GB) and `checkpoints/` (except `emotion_cnn.pt`) are gitignored.

## Hard rules

- **Select on dev, report on test once.** Models, weights, thresholds and prompts are chosen on
  MELD dev only. Never read test data, test caches or test predictions to make a decision. Test
  evaluation runs only after the recipe is locked (hashes in `results/submission_model_selection.json`).
  Details: `.claude/rules/evaluation.md`; procedure: the `dev-experiment` and `locked-test-report` skills.
- **Never edit evidence.** Files under `results/retraining_20261023/final_test/`, the selection
  receipt, and benchmark outputs are records. New runs write to new, dated directories.
- **No hand-typed numbers.** README figures are copied from `results/*.md|json`. If a number changes,
  rerun the script that produces it and update README from its output.
- **Stay inside the 6B learned-parameter budget** (currently 3,434,439,901). Any new model or head must
  be added to `scripts/count_params_offline.py` and the README table.
- **Keep training and serving identical.** Audio features come only from `app/features.py`
  (`FEATURE_CONFIG` is stored in checkpoints and checked at load). Fusion offline and online both go
  through `app.fusion.fuse` + `runtime_weights`.
- **Don't block the asyncio loop.** Model calls run on the Engine's worker pools (`run_torch`,
  `run_mlx`, `run_vision`); MLX objects must stay on the MLX thread.

## Conventions

- Python 3.12, standard library + existing dependencies only; pin new packages in both
  `requirements.txt` and `requirements-verified.txt` (avoid adding any).
- Existing modules use `# Purpose / # Input / # Output` header comments; follow the local style of
  the file you edit.
- Emotion labels are always `app.config.EMOTIONS` order (alphabetical, 7 classes).
- WebSocket events are a public interface: add fields, don't rename or remove them
  (`app/protocol.py`, `app/static/app.js`, `tests/test_protocol.py`).
- Frontend is plain HTML/CSS/JS with no build step; keep `node tests/test_frontend.js` passing.
- Commit in small, logical commits; the user pushes.

## Definition of done

1. The three test commands above pass.
2. If behaviour or numbers changed: results regenerated into `results/`, README updated from them.
3. If the runtime changed, the user has run it on the Mac (MLX needs the Apple GPU; a Linux
   container or VM cannot run the default backend).

## Known gotchas

- Whisper can loop on noise or long clips ("No, no, no, ..." or "Sssss..."). About 0.5% of dev and
  0.8% of test transcripts exceed Whisper's own compression-ratio threshold (2.4). The emotion path
  keeps them (dropping them did not help on dev); the reply prompt treats them as unclear speech.
- HSEmotion's `contempt` is merged into `disgust` (`app/vision.py`), so a neutral webcam face often
  reads as disgust. The face has fusion weight 0.3 and adds no reliable gain on test.
- MELD clip boundaries often include the next speaker, so transcripts can run past the labelled line.
- ~96% of dev/test lines come from speakers seen in training (mostly the six main Friends
  characters). MELD scores are speaker-dependent; expect worse results on a new live voice.
- The CPU backend unpacks 4-bit Qwen to float32: ~14.6 GiB peak memory vs ~3.9 GiB on MLX.
- `training.evaluate` refuses to overwrite an existing `--output-dir`; pick a new dated path.

## Where to look

- ML practice checklist and how this repo meets it: `.claude/specs/ml-practices.md`
- Challenge brief and requirement map: `.claude/specs/challenge-brief.md`
- Runtime rules (threads, protocol, latency): `.claude/rules/runtime.md`
