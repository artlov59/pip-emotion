# Four-recipe development comparison

Checkpoint/fusion selection and this recipe comparison use dev. Bootstrap intervals are conditional, not selection-adjusted or held-out gains. Scores use labelled clips rather than end-to-end streaming VAD.

Vision and the current fusion weights remain fixed for the primary comparison.

| recipe | weighted F1 | delta | macro F1 | accuracy | macro guard |
|---|---:|---:|---:|---:|---|
| audio0_text0 | 0.5201 | +0.0000 | 0.3719 | 0.5440 | True |
| audio1_text0 | 0.5201 | +0.0000 | 0.3719 | 0.5440 | True |
| audio0_text1 | 0.5321 | +0.0120 | 0.3907 | 0.5476 | True |
| audio1_text1 | 0.5321 | +0.0120 | 0.3907 | 0.5476 | True |

audio0/text0 are the original models; audio1/text1 are the candidates.
Development recommendation: **audio0_text1**. This is a review snapshot; runtime defaults are unchanged and no test recipe is locked.

| emotion | baseline F1 | recommended F1 | delta |
|---|---:|---:|---:|
| anger | 0.4672 | 0.4714 | +0.0042 |
| disgust | 0.0513 | 0.0500 | -0.0013 |
| fear | 0.1364 | 0.2373 | +0.1009 |
| joy | 0.4542 | 0.4714 | +0.0172 |
| neutral | 0.6667 | 0.6801 | +0.0134 |
| sadness | 0.3243 | 0.3249 | +0.0005 |
| surprise | 0.5032 | 0.5000 | -0.0032 |

Recommended recipe's conditional paired dialogue-bootstrap weighted-F1 delta interval: [-0.0034, +0.0285].
All four recipes' per-class metrics, changes, missing-channel counts, guards and intervals are in JSON.
