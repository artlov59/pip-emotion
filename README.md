# Pip: real-time emotion state and replies from voice, words and face

Pip is a prototype for a character robot. It listens to one person through a microphone (and,
optionally, a camera), tracks how they sound while they talk, and when they finish it returns:

1. a **structured state**: one of MELD's seven emotions with probabilities, per-input readings,
   a transcript, prosody and tags such as `voice:anger`, `words:surprise`, `agreement:conflict`;
2. a **short reply**, streamed word by word and grounded in that state.

Everything runs locally on a Mac. The full inference path has 3.43 B learned parameters (limit: 6 B).

<!-- Demo video: add the link here -->

## Track and scope

- **Primary track: Audio.** The voice model is my own ResNet-style CNN on log-mel spectrograms,
  trained on MELD. Voice drives turn-taking, the live emotion updates and the final state.
- **Extension: voice + words + face.** Whisper transcribes each turn and a fine-tuned DistilRoBERTa
  classifies the words; a face detector and expression model read the camera. The three are
  fused, and a calibrated check flags when the words and the delivery disagree, for example
  "great" said angrily.
- **Interaction model:** one English speaker, taking turns with Pip. Pip doesn't interrupt and
  can't be interrupted.

## What "real-time" means here

For a robot in a face-to-face conversation, real-time means keeping pace with the conversation,
not frame-by-frame output. My assumptions:

- **Input is processed as it arrives.** The browser sends 100 ms chunks of 16 kHz audio and about
  5 camera frames per second. Nothing waits for a whole file.
- **While the person talks**, a provisional emotion (`partial_state`) updates every 0.5 s from the
  last 3 s of voice plus face. One update costs ~15 ms of CNN compute, so it never falls behind.
  Face readings stream continuously (~12 ms per frame; frames are dropped, not queued, if busy).
- **A turn ends after 500 ms of silence** (voice activity detection, Silero VAD). From that moment
  my targets are: structured state within ~0.5 s and the first reply words within ~1 s, with the
  rest of the reply streamed as it's generated.
- **Why these numbers:** people normally answer each other within a fraction of a second. A robot
  that starts answering about a second after you stop feels attentive; two seconds or more feels
  laggy. The 500 ms silence window is the main fixed cost: shorter cuts people off when they
  pause mid-sentence, longer feels sluggish.

Measured latency (35 MELD test clips streamed at 1x, all three inputs, times after the end of
speech is detected, so add the 500 ms silence window for the user's wait):

| stage | target | CPU backend p50 | CPU backend p95 |
|---|---:|---:|---:|
| structured state (`final_state`) | ~0.5 s | 0.48 s | 0.82 s |
| first reply words | ~1 s | 1.46 s | 2.45 s |
| full reply | streamed | 3.40 s | 4.82 s |

<!-- MLX backend row: python -m scripts.benchmark_latency --n 35 --output-dir results/benchmark_mlx -->

The CPU backend meets the state target at the median and misses the first-words target, mostly
because the 3B reply model generates ~6 visible chunks per second on CPU. Details:
[`results/retraining_20261023/benchmark_multimodal_cpu.md`](results/retraining_20261023/benchmark_multimodal_cpu.md).
The benchmark runs in-process, so it excludes browser and WebSocket overhead.

## How it works

```text
PCM 100 ms -> VAD -> utterance ----> AudioCNN ---------------------- voice --+
                 |             +---> Whisper-small -> DistilRoBERTa - words --+-> fusion -> final_state
                 |             +---> prosody (pitch, loudness, pace) - tags   |       + agreement + tags
                 +-- every 0.5 s while speaking: AudioCNN(last 3 s) + face -> partial_state
JPEG ~5 fps -> YuNet face detector -> HSEmotion ------------------ face ---+               |
                                                                                           v
                                                         Qwen2.5-3B-Instruct (4-bit) -> streamed reply
```

- **Fusion** is a weighted product of the three probability distributions:
  `score(c) = sum_m w_m * log p_m(c)`, then softmax. The weights were tuned on MELD dev
  (audio 1.0, words 0.9, face 0.3), with a separate set for each combination of available inputs,
  so a missing camera or empty transcript just drops that input.
- **Words vs. delivery.** Each input is first temperature-calibrated on dev so the confidences
  are comparable. If the words and voice+face disagree, the delivery is flagged only when it is
  the more confident side. Only then does the reply prompt tell Pip to respond to how the person
  sounded rather than the literal words.
- **Reply.** Qwen sees the transcript, the top readings from each input, the fused read, a
  plain-language delivery description ("speaking fast, animated") and the agreement note, plus the
  last three turns.
- **Concurrency.** The asyncio loop only runs VAD and bookkeeping. The CNN and text model share one
  worker thread, face analysis has its own, and Whisper and Qwen share a third, so live updates keep
  flowing while a reply is being generated. A reset cancels in-flight work and drops stale results.

Example `final_state` from a real run (MELD clip `dia17_utt8`, labelled anger; probabilities
trimmed, full trace in [`results/example_trace_multimodal.json`](results/example_trace_multimodal.json)):

```json
{
  "type": "final_state",
  "emotion": "anger",
  "confidence": 0.6608,
  "modalities": {
    "voice": {"emotion": "anger", "confidence": 0.3422},
    "words": {"emotion": "surprise", "confidence": 0.3799},
    "face":  {"emotion": "anger", "confidence": 0.3397, "frames": 20}
  },
  "fusion": {"used": ["audio", "text", "vision"], "weights": {"audio": 1.0, "text": 0.9, "vision": 0.3}},
  "transcript": "I know! At 4.99 for a pillow-top queen said who cares about the divorce?",
  "tags": ["emotion:anger", "voice:anger", "words:surprise", "face:anger", "pitch_variation:animated", "pace:fast"],
  "delivery": "animated, varied intonation, speaking fast",
  "latency_ms": {"emotion": 26.0, "asr": 589.5, "end_of_speech_to_state": 599.1}
}
```

Streamed reply: *"It seems like a lot of stress though. Maybe you're feeling upset about something."*
The transcript runs past the labelled line ("I know!") because MELD clip boundaries often include
the next speaker. This trace was recorded before the words-vs-delivery calibration; re-scored with
it, the turn is flagged as delivery-led (words: surprise 0.32 vs voice+face: anger 0.66), which
matches the label.

## Run it

Requirements: an Apple Silicon Mac, Python 3.12 and `ffmpeg` on PATH (`brew install ffmpeg`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-verified.txt
python -m scripts.download_models            # text model (GitHub release) + face detector, checksummed
python -m scripts.cache_models --output results/model_cache_manifest.json   # Whisper + Qwen
python -m app.server --require-multimodal    # MLX backend (default) on the Apple GPU
```

Open http://127.0.0.1:8000, start the microphone and camera, and talk. You can also stream an
audio or video file from the page. `--require-multimodal` refuses to start if any model is
missing instead of silently dropping an input. `python -m app.server --backend cpu` runs
without the GPU (slower; see the trade-offs below).

Replaying a clip through the running server prints every event:

```bash
python -m app.replay path/to/clip.mp4
python -m app.replay --clip dia17_utt8          # needs prepared MELD data
```

**Protocol** (`/ws`): binary messages start with `0x01` (little-endian int16 mono PCM at 16 kHz) or `0x02` (JPEG). Control
messages are `{"type": "flush"}` and `{"type": "reset"}`. The server sends `ready`, `vad`,
`partial_state`, `face_state`, `final_state`, `response_delta`, `response_done`, `flushed` and
`error` events.

## Results

MELD, 7 emotions, official splits (9,877 train / 1,092 dev / 2,576 test clips with usable audio).
Weighted F1 is MELD's usual metric; macro F1 shows how rare classes like fear and disgust do.
I picked every model, weight and threshold on dev, then opened test once after locking them (with
file hashes). Scores use MELD's clip boundaries, so they don't include VAD segmentation errors.

**Test set, 2,576 clips:**

| model | weighted F1 | macro F1 | accuracy |
|---|---:|---:|---:|
| always predict "neutral" (majority baseline) | 0.312 | 0.093 | 0.481 |
| voice only: AudioCNN (primary track) | 0.421 | 0.225 | 0.458 |
| words only: Whisper -> DistilRoBERTa | 0.509 | 0.350 | 0.509 |
| voice + words | 0.532 | 0.363 | 0.545 |
| **voice + words + face (deployed)** | **0.537** | **0.365** | **0.554** |

What I take from this:

- **Voice adds a real gain on top of words:** +2.3 weighted-F1 points (95% dialogue-bootstrap
  interval +1.1 to +3.6). **Face adds +0.5 points, which isn't reliable** (-0.5 to +1.6). Likely reasons:
  the face model was never trained on MELD, and TV shots often show a listener, not the speaker.
- **The voice model is the weakest part.** It beats the majority baseline on weighted F1, but its
  accuracy (45.8%) is below always predicting "neutral" (48.1%), and it never predicts disgust
  or fear. For context, the emotion2vec paper reports MELD audio-only weighted F1 of 35.2 for
  WavLM-base, 43.6 for data2vec and 48.7 for emotion2vec ([Ma et al., 2023](https://arxiv.org/abs/2312.15185)).
  Their setup differs from mine, so this is only a rough reference. A from-scratch 21M-parameter
  CNN at 42.1 sits in that range but well below the best pretrained encoder.
- **Speech recognition costs more than the camera adds.** On dev, the text model scores 0.564
  weighted F1 on MELD's own transcripts but 0.490 on Whisper's (TV audio, overlapping speakers,
  clip boundaries).

Dev scores and per-class F1 are in [`results/baselines.md`](results/baselines.md). The text model
was fine-tuned again on a mix of gold and Whisper transcripts (dev weighted F1 0.477 -> 0.490).
Two attempts to retrain the audio CNN and one learned-fusion experiment didn't beat the existing
models on dev, so I kept those. That evidence is in [`results/retraining_20261023/`](results/retraining_20261023/).

**Words-vs-delivery check.** The first version compared raw confidences with a fixed 0.4
threshold. The audio CNN is under-confident and the text and face models are over-confident, so
it flagged a mismatch on 37% of turns, and in those turns the words were right far more often
than the delivery the prompt told Pip to follow. After per-input temperature calibration and a
"delivery must be the more confident side" rule, both tuned on dev only
([`scripts/calibrate_agreement.py`](scripts/calibrate_agreement.py)):

| test set (2,576 clips) | turns flagged | delivery right when flagged | words right when flagged |
|---|---:|---:|---:|
| before: raw confidences, threshold 0.4 | 941 (36.5%) | 25% | 45% |
| after: calibrated, delivery must lead | 564 (21.9%) | 38% | 27% |

The gap (delivery minus words) is +11.5 points, with a 95% dialogue-bootstrap interval of +4.9 to
+18.5. Neither side is right in many flagged turns, so the prompt treats the flag as a hint, not
a fact. The fused emotion itself is unchanged by this.

## Hardware, resources and parameter budget

Measured on an Apple Silicon Mac with 48 GB unified memory and 18 logical CPU cores (macOS 26.6.2,
Python 3.12.15, PyTorch 2.14.1). CPU backend, 4 threads: models load in 8.0 s, peak resident
memory is 14.6 GiB and mean process CPU is ~96% of one core. The CPU backend expands the 4-bit
Qwen weights to 32-bit floats, which accounts for most of that memory; the MLX backend keeps them
in 4-bit.

| component (all required at inference) | learned parameters |
|---|---:|
| Silero VAD | 462,594 |
| AudioCNN (mine, voice emotion) | 21,281,991 |
| Whisper-small (speech to text) | 240,582,912 |
| DistilRoBERTa (fine-tuned by me, words emotion) | 82,123,783 |
| YuNet face detector + HSEmotion EfficientNet-B0 | 4,049,910 |
| Qwen2.5-3B-Instruct, 4-bit (reply) | 3,085,938,688 |
| fusion weights (all stored tables) + agreement calibration | 17 + 6 |
| **total** | **3,434,439,901** |

Quantized weights are counted as their unpacked values, frozen models are counted in full, and
ONNX files are counted conservatively (every initializer). Re-check with
`python -m scripts.count_params_offline --require-multimodal`.

## Decisions and trade-offs

- **Late fusion instead of a learned joint model.** Each input stays a separate, testable model,
  any of them can be missing at runtime, the weights are 17 numbers I can explain, and they were
  tuned on dev. A learned fusion layer didn't beat it on dev. The cost: no cross-modal
  interactions beyond the agreement check.
- **A from-scratch audio CNN instead of a pretrained speech encoder.** I adapted a CNN I'd built
  earlier for environmental sounds (ESC-50): I know it end to end and it runs in ~15 ms. It
  is clearly weaker than pretrained encoders. Since Pip already runs Whisper, the cheapest upgrade
  is a linear head on Whisper's encoder states (5,383 extra parameters).
  [`training/probe_whisper_encoder.py`](training/probe_whisper_encoder.py) tests that on dev.
- **Turn-based VAD with a 500 ms silence window.** It's simple and robust, and it gives the reply
  model a complete thought. The cost is a fixed 0.5 s wait and no barge-in.
- **Qwen2.5-3B in 4-bit for replies.** A balance between reply quality and speed: big enough to
  follow the persona and length rules, small enough to fit the budget with ~2.5 B parameters to
  spare. It is still the slowest stage.
- **Two backends.** MLX on the Apple GPU is the default. The CPU backend loads the same cached
  Whisper and Qwen weights into PyTorch so the whole system can run and be benchmarked without a
  GPU. It is slower and uses much more memory, and its outputs aren't guaranteed to match MLX
  exactly.
- **Evaluation discipline over extra experiments.** Selection on dev, hash-locked test,
  dialogue-level bootstrap intervals, and every claim in this README traceable to a file in
  `results/`.

## Limitations

- The face model picks the largest face, which in a TV shot is often a listener, not the speaker.
- Emotion confidences in the state are uncalibrated (only the agreement check uses calibrated
  values).
- MELD is acted TV dialogue with laugh tracks. A live user at a laptop sounds and looks different,
  and live microphone and webcam accuracy hasn't been measured.
- Reply quality hasn't been formally evaluated.
- The words model classifies each utterance on its own, without the previous turns, which usually
  helps on MELD.

## Completed and left out

Completed:

- Audio-track pipeline end to end: microphone -> VAD -> live and final voice emotion -> structured
  state -> streamed reply, in a browser UI and over a WebSocket protocol.
- Extension with all three inputs working together, including a dev-calibrated words-vs-delivery
  check, with evidence that voice adds measurable value over words.
- Evaluation on official MELD splits with baselines, locked test, latency benchmark, parameter audit
  and tests.

Left out on purpose:

- Reinforcement learning (optional in the brief).
- Speech output (TTS), barge-in, active-speaker detection, diarization, multiple users.
- Fine-tuning the face model on MELD, dialogue context for the words model, and swapping in a
  pretrained speech encoder (the probe script is the first step).
- A formal reply-quality study and a live-user study.

## Repository map

```text
app/          runtime: pipeline.py (sessions, workers), fusion.py (fusion + agreement), emotion.py,
              text_emotion.py, vision.py, asr.py, llm.py, prosody.py, vad.py, server.py, static/ (UI);
              cpu_asr.py / cpu_llm.py are the CPU backend
model.py      AudioCNN
training/     MELD preparation, audio and text training, transcription, locked evaluation
scripts/      latency benchmark, parameter audit, model download and cache, calibration, baselines
tests/        unit, session-regression and frontend tests
results/      every number in this README
```

Training and evaluation commands are documented in each script's `--help`. The main ones:
`python -m training.prepare_meld`, `python -m training.train`, `python -m training.train_text`,
`python -m training.evaluate {audio,compare,test}`, `python -m scripts.calibrate_agreement {fit,test}`
and `python -m scripts.report_baselines`.

Tests: `python -m unittest discover -s tests`, `python -m tests.test_pipeline` and
`node tests/test_frontend.js`.

## External components and AI assistance

External: MELD (data), Silero VAD, Whisper-small, DistilRoBERTa-base (which I fine-tuned), YuNet,
HSEmotion EfficientNet-B0 and Qwen2.5-3B-Instruct. Model revisions and file hashes are recorded by
`scripts/cache_models.py` and `results/submission_model_selection.json`. The audio CNN and its
training, the text fine-tuning, the streaming pipeline, fusion, calibration, prompts and tools are
this project's own components.

I used an AI coding assistant for parts of the implementation, review, experiments and
documentation. I'm responsible for the code, the results and the decisions described here.
