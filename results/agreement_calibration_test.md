# Words-vs-delivery check on held-out test

Rule locked on dev (`results/agreement_calibration.json`, sha256 93370ad9459b...). 2576 test utterances; labelled clip boundaries and cached face readings.

| rule | flagged turns | delivery right when flagged | words right when flagged | net |
|---|---:|---:|---:|---:|
| previous | 941 (36.5%) | 25% | 45% | -187 |
| calibrated | 564 (21.9%) | 38% | 27% | +65 |

Calibrated rule, delivery-right minus words-right rate among flagged turns: +11.5% (95% dialogue-bootstrap interval +4.9% to +18.5%).

Neither side is right in many flagged turns, so the reply prompt treats the flag as a hint.
