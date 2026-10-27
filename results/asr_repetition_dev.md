# Whisper repetition loops (dev check)

5 of 1092 dev transcripts exceed Whisper's compression-ratio threshold (2.4); 21 of 2576 test transcripts do (counted after the decision).

| dev recipe | weighted F1 | macro F1 | accuracy |
|---|---:|---:|---:|
| deployed | 0.5321 | 0.3907 | 0.5476 |
| words of looping transcripts dropped | 0.5308 | 0.3906 | 0.5467 |

Adopt for emotion: no. Decision: keep words for emotion; mark transcript as garbled in the reply prompt only.
