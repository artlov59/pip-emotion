# Pip: local multimodal emotion state and streamed reply

Pip accepts speech and video over time. It combines voice, recognized words and facial
expression into a seven-category MELD emotion state, then streams a short spoken-style
text reply. Models run locally. The target is one English speaker, sequential turns,
and an Apple-Silicon Mac.

## Run

Use Python 3.12, an Apple-Silicon Mac, and `ffmpeg` on PATH. The submission includes the
selected audio and text classifiers and YuNet detector. MELD media is not bundled.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-verified.txt
python -m scripts.cache_models --output results/model_cache_manifest.json
python -m app.server --backend cpu --cpu-threads 4 --require-multimodal
```

Open http://127.0.0.1:8000 and start microphone and camera, or stream an audio/video file.
The MELD clip library requires prepared dataset files. Microphone input does not.
Whisper and Qwen are acquired during setup; HSEmotion downloads its weights on first
load. Acquisition can use the network; inference runs locally.

The default backend is MLX. On a Metal-enabled session use:

```bash
python -m app.server --require-multimodal
```

For exact MLX reproduction, set `ASR_REPO` and `LLM_REPO` to the local snapshot paths
written by `scripts.cache_models`. CPU mode resolves the pinned cached revisions itself
and never downloads Whisper/Qwen. CPU decoding is not claimed to match MLX numerically.
The CPU Qwen adapter reconstructs cached affine four-bit values into FP32 memory. Its
process memory guard is 20 billion bytes; measured peak RSS was 14.64 GiB.

Useful configuration: `TEXT_MODEL_DIR`, `EMOTION_CKPT`, `TORCH_DEVICE`, `USE_TEXT`,
`USE_VISION`, `USE_LLM`. `--audio-only` and `--no-llm` are diagnostic modes. Strict
`--require-multimodal` startup fails if the required models cannot load. A three-input
interaction also needs recognized words and video containing a face.

## Interaction and outputs

The browser sends 16 kHz mono PCM in 100 ms chunks and JPEG video at about 5 fps over
`/ws`. Audio drives Silero VAD; 500 ms silence closes a turn. While speaking, the CNN
analyzes the last three seconds every 0.5 seconds. Final voice emotion uses up to the
last ten seconds. Face readings are matched to the captured audio interval.

```text
PCM -> VAD -> utterance -> AudioCNN ---------------- voice --+
                       -> Whisper -> DistilRoBERTa - words --+-> fusion -> final_state
JPEG -> YuNet -> HSEmotion ------------------------- face ---+                |
                       utterance -> prosody ---------------- tags            v
                                                                  Qwen -> streamed reply
```

Events include `ready`, `vad`, `partial_state`, `face_state`, `final_state`,
`response_delta`, `response_done`, `flushed` and `error`. Control messages are
`{"type":"flush"}` and `{"type":"reset"}`. Binary messages start with `0x01` for PCM
or `0x02` for JPEG. Reset cancels pending work and suppresses stale results.

A final state contains an emotion from anger/disgust/fear/joy/neutral/sadness/surprise,
probabilities, transcript, per-modality readings, prosody, agreement and latency fields.
`fusion.used` identifies the inputs actually used. Reply deltas reconstruct the final
response text. Legacy `tokens`/`tokens_per_s` fields count visible stream deltas;
`stream_deltas`/`stream_deltas_per_s` make that unit explicit.

```bash
# Keep the server running in another terminal. Prepared MELD supplies the clip/video.
python -m app.replay --clip dia17_utt8
python -m app.replay path/to/clip.mp4
```

`results/example_trace_multimodal.json` records a real selected-model CPU interaction.
The full chain produced an anger state using audio/text/vision and streamed a reply.
ASR also recognized speech beyond the labelled "I know!" line, illustrating MELD clip
boundary noise. This is a functional demonstration, not a perfect recognition example.

## Results and constraints

The selected recipe keeps the original AudioCNN and frozen fusion/vision weights, and
uses text checkpoint epoch 2 after mixed gold/CPU-ASR fine-tuning. Development selection
required higher weighted F1 without lower macro F1. Model hashes were locked before the
final comparison. Existing historical test reports predated this experiment; no new test
result was used to select models or weights.

| evaluation | baseline weighted F1 | selected weighted F1 | baseline macro F1 | selected macro F1 |
|---|---:|---:|---:|---:|
| text, dev | 0.4773 | 0.4902 | 0.3394 | 0.3568 |
| audio + text + vision, dev | 0.5201 | 0.5321 | 0.3719 | 0.3907 |
| text, test | 0.5068 | 0.5089 | 0.3395 | 0.3498 |
| audio + text, test | 0.5135 | 0.5321 | 0.3420 | 0.3629 |
| audio + text + vision, test | 0.5311 | 0.5372 | 0.3591 | 0.3654 |

Test uses 2,576 usable labelled clips, singleton CPU audio/text wrappers, frozen MLX
transcripts and cached vision. It does not measure VAD segmentation. The all-three gain
is 0.62 weighted-F1 points; its paired dialogue-bootstrap 95% interval is [-0.39,+1.62]
points. Overall accuracy falls 0.62 points and disgust F1 falls from 0.1846 to 0.1417.
This is a small observed trade-off, not strong evidence of general improvement.
Two audio adaptation runs and a learned-fusion experiment failed their safeguards and
were not selected. Their obsolete training/experiment tools have been retired.

Full CPU benchmark: 35 clips at 1x, 36 detected utterances using all three modalities,
32 clips with completed replies, three clips with no detected speech, zero errors,
and 527 video frames with no drops.

| stage after VAD declares end-of-speech | median | p95 |
|---|---:|---:|
| structured state | 483.2 ms | 822.6 ms |
| first visible reply text | 1461.0 ms | 2447.7 ms |
| completed reply | 3403.4 ms | 4816.8 ms |

Measurements exclude browser/WebSocket overhead and the configured 500 ms silence wait.
Including silence, median first reply is about 1.96 seconds. The design target was about
one second, so CPU misses it. Full MLX multimodal performance remains unverified because
the execution environment exposes no Metal device. Perception updates continue during
speech and replies; the user waits before the next turn. Barge-in is out of scope.

Observed hardware: arm64 Mac, 48 GiB memory, 18 logical CPUs, macOS 26.6.2, Python
3.12.15, Torch 2.14.1 with four intra-op threads. The sandbox hides the chip brand; the
earlier hardware report identified Apple M5 Pro. Load took 8.0 s, peak RSS 14.64 GiB,
and mean process CPU was 95.7% (100% is one core). JSON `*_gb` fields use bytes/2^30.

| required component | learned parameters / conservative bound |
|---|---:|
| Silero VAD | 462,594 |
| AudioCNN | 21,281,991 |
| Whisper small | 240,582,912 |
| DistilRoBERTa text emotion | 82,123,783 |
| YuNet + HSEmotion | 4,049,910 |
| Qwen2.5-3B-Instruct | 3,085,938,688 |
| all stored fusion coefficients | 17 |
| **total** | **3,434,439,895 < 6 billion** |

Quantized matrices count logical unpacked weights. Fixed alignment-head indices,
sinusoidal positions and packing metadata are not learned weights. ONNX initializers
are included conservatively. Frozen encoders are counted in full.

Detailed evidence lives in `results/retraining_20261023/`: development comparison,
locked test metrics/predictions, text evaluation, audio experiment histories, benchmark
CSV/JSON and the selected parameter audit. `results/submission_model_selection.json`
records current checkpoint locations and hashes; it references the original immutable
freeze receipt. The original text baseline is retained for paired evaluation. Duplicate
copies and inactive training checkpoints are outside the maintained project.

## Training and evaluation

Prepared data uses official MELD splits: 9,877 audio-usable train / 1,092 dev / 2,576 test
clips. Text training retains all 9,989 train utterances and falls back to gold text where
ASR/audio is missing or ASR hit its token cap. New checkpoints use distinct output paths.

```bash
# Prepare an existing data/MELD.Raw.tar.gz archive.
python -m training.prepare_meld
# Audio training; exact pooling and corrected variable-length mixup for new models.
python -m training.train --run-name audio_new
python -m training.evaluate audio --checkpoint checkpoints/audio_new/best.pt --output-dir results/audio_new_dev
# Create a separate CPU training transcript cache, or reuse the existing verified cache.
python -m training.transcribe_cpu --output data/meld/whisper_cpu_train_new.csv
# MLX development/test transcript regeneration is available through training.transcribe_meld.
python -m training.train_text --train-source mixed --selection-source whisper \
  --init-model-dir checkpoints/text_emotion_baseline_20261023 --local-files-only \
  --train-asr-cache data/meld/whisper_cpu_train.csv --preserve-baseline-macro \
  --epochs 2 --lr 1e-5 --device cpu --output-dir checkpoints/text_new
```

All scoring entry points are under `python -m training.evaluate`: `audio` is development
only; `compare` compares two fixed recipes on development caches without fitting;
`test` requires a dev-selection lock and verifies hashes before opening test inputs.
See each subcommand's `--help`. No command promotes a model from its test score.
Training text evaluates dev only by default; test requires explicit `--eval-only`.
Face cache regeneration uses `training.extract_faces_meld`; CPU shard merging uses
`training.merge_asr_caches`. Frozen fusion weights are supplied in results/fusion_weights.json.

## Verification and handoff

```bash
python -m unittest discover -s tests
python -m tests.test_pipeline
node tests/test_frontend.js
python -m scripts.count_params_offline --require-multimodal --output results/parameter_check.json
python -m scripts.benchmark_latency --backend cpu --cpu-threads 4 --n 35
python -m scripts.create_submission_bundle --repo . \
  --audio-checkpoint checkpoints/emotion_cnn.pt --text-model-dir checkpoints/text_emotion \
  --selection-receipt results/submission_model_selection.json \
  --face-detector checkpoints/face_detection_yunet_2023mar.onnx --output /path/to/pip_submission.tar.gz
```

The archive includes selected models, source, tests, evidence and per-file checksums.
It excludes MELD media, environments, caches and unselected training checkpoints.
`requirements.txt` lists direct dependencies; `requirements-verified.txt` pins the tested
Python 3.12 environment. The baseline text model is needed only for paired comparison
or warm-start reproduction and is supplied separately from the selected-model archive.

The app has separate modality adapters, one engine owning inference workers, a per-session
state machine, a small protocol validator and a browser client. Tensor conversion is
confined to the CPU adapters. Evaluation and runtime share fusion-weight lookup.

External components are MELD, Silero VAD, Whisper-small, DistilRoBERTa-base, YuNet,
HSEmotion EfficientNet-B0 and Qwen2.5-3B-Instruct. Whisper/Qwen pinned revisions and source
hashes are recorded by `scripts.cache_models`; learned-parameter evidence names all models.
The audio classifier, MELD text fine-tuning, streaming pipeline, fusion, prompts and tools
are project components. An AI coding assistant assisted implementation, review, experiments
and documentation; the repository author remains responsible for the submitted results.

Limitations: largest-face selection can choose a non-speaker; emotion confidences are
uncalibrated; TV dialogue differs from live users. Response quality and live microphone/
webcam hardware have not been formally evaluated. RL, active-speaker detection, face
fine-tuning, diarization, TTS, barge-in and production multi-user operation were left out.
