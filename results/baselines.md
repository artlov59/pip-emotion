# Baselines and per-modality scores

Re-scored from saved predictions on labelled MELD clips (not streaming VAD). Dev picked every model and weight; test was opened once after that.

## dev (1092 clips)

| model | weighted F1 | macro F1 | accuracy | never predicts |
|---|---:|---:|---:|---|
| always neutral | 0.253 | 0.085 | 0.424 | all but neutral |
| audio CNN | 0.406 | 0.257 | 0.429 | fear |
| text (Whisper transcript) | 0.490 | 0.357 | 0.495 | - |
| text (MELD gold transcript) | 0.564 | 0.418 | 0.563 | - |
| audio + text + face (deployed fusion) | 0.532 | 0.391 | 0.548 | - |

## test (2576 clips)

| model | weighted F1 | macro F1 | accuracy | never predicts |
|---|---:|---:|---:|---|
| always neutral | 0.312 | 0.093 | 0.481 | all but neutral |
| audio CNN | 0.421 | 0.225 | 0.458 | disgust, fear |
| text (Whisper transcript) | 0.509 | 0.350 | 0.509 | - |
| audio + text | 0.532 | 0.363 | 0.545 | - |
| audio + text + face (deployed fusion) | 0.537 | 0.365 | 0.554 | - |

## Per-class F1 on test

| model | anger | disgust | fear | joy | neutral | sadness | surprise |
|---|---:|---:|---:|---:|---:|---:|---:|
| audio CNN | 0.29 | 0.00 | 0.00 | 0.20 | 0.64 | 0.16 | 0.29 |
| text (Whisper transcript) | 0.33 | 0.16 | 0.07 | 0.43 | 0.66 | 0.33 | 0.46 |
| audio + text + face (deployed fusion) | 0.40 | 0.14 | 0.08 | 0.43 | 0.70 | 0.34 | 0.47 |

## What each input adds on test (paired, dialogue bootstrap)

| change | weighted F1 change | 95% interval |
|---|---:|---:|
| voice on top of words | +2.3 pts | +1.1 to +3.6 pts |
| face on top of words + voice | +0.5 pts | -0.5 to +1.6 pts |
