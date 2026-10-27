# Words-vs-delivery calibration (dev)

Fitted on MELD dev predictions from the deployed audio CNN, the selected text model on Whisper transcripts, and cached face readings. Fusion and the reported emotion are unchanged.

| modality | temperature | ECE before | ECE after |
|---|---:|---:|---:|
| audio | 0.649 | 0.137 | 0.074 |
| text | 1.362 | 0.122 | 0.027 |
| vision | 2.767 | 0.168 | 0.049 |

T < 1 means the model was under-confident (sharpened); T > 1 over-confident (softened).

| rule | flagged turns | delivery right when flagged | words right when flagged | net |
|---|---:|---:|---:|---:|
| previous: raw probabilities, voice:face 1:1, both confidences >= 0.4 | 413 (37.8%) | 27% | 45% | -77 |
| selected: calibrated, voice:face 1:1, margin +0.00 | 254 (23.3%) | 35% | 27% | +21 |

Full sweep (vision weight x margin) is in agreement_calibration.json.
