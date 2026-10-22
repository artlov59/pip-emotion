MELD **dev** split, audio only, n=1092

| model | weighted F1 | macro F1 | accuracy | F1 anger | F1 disgust | F1 fear | F1 joy | F1 neutral | F1 sadness | F1 surprise |
|---|---|---|---|---|---|---|---|---|---|---|
| majority class (neutral) | 0.252 | 0.085 | 0.424 | 0.00 | 0.00 | 0.00 | 0.00 | 0.60 | 0.00 | 0.00 |
| ce_plain | 0.390 | 0.218 | 0.465 | 0.44 | 0.00 | 0.00 | 0.15 | 0.63 | 0.00 | 0.30 |
| wce_sqrt | 0.393 | 0.227 | 0.445 | 0.43 | 0.00 | 0.00 | 0.18 | 0.61 | 0.03 | 0.33 |
| wce_sqrt_mixup | 0.410 | 0.254 | 0.443 | 0.42 | 0.00 | 0.00 | 0.22 | 0.60 | 0.21 | 0.32 |
