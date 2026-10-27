# Frozen-recipe held-out comparison

Labelled utterance boundaries and frozen vision/ASR caches; not a streaming VAD, generation-quality or end-to-end interaction benchmark.

| mode | baseline wF1 | selected wF1 | delta | baseline macro | selected macro |
|---|---:|---:|---:|---:|---:|
| audio | 0.4215 | 0.4215 | +0.0000 | 0.2250 | 0.2250 |
| text | 0.5068 | 0.5089 | +0.0021 | 0.3395 | 0.3498 |
| audio_text | 0.5135 | 0.5321 | +0.0186 | 0.3420 | 0.3629 |
| audio_text_vision | 0.5311 | 0.5372 | +0.0062 | 0.3591 | 0.3654 |

The recipe was locked using development data before this test comparison. No fitting, tuning or recommendation is performed here.
