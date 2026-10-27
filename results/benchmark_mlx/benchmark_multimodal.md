[voice + words + face; backend=mlx] 36 utterances from 35 MELD test clips, streamed at 1x. Latencies exclude 500 ms of end silence.

in-process Session; excludes WebSocket, network and browser capture overhead.

| stage | p50 | p95 |
|---|---|---|
| provisional CNN (per-utterance mean compute) | 25.4 | 43.8 |
| face analysis (per-clip mean compute) | 19.3 | 20.5 |
| final emotion (CNN) | 54.5 | 83.3 |
| Whisper transcript | 94.8 | 129.2 |
| words emotion (after Whisper) | 18.9 | 30.7 |
| end of speech → final_state | 113.0 | 139.7 |
| end of speech → first visible reply delta | 228.4 | 268.9 |
| end of speech → full reply | 365.6 | 425.3 |
| visible reply deltas / s | 67.5 | 77.5 |

Clips with no detected speech: 3; with errors: 0; completed replies: 32.
Frames sent/dropped: 527/0.
Hardware: Apple M5 Pro · 48.0 GB RAM · macOS 26.6.2
Model load 12.2 s · peak RSS 3.86 GB · MLX peak 2.96 GB · MPS driver 3.21 GB · CPU 71.2% (100% = one core).
RSS, MLX peak and MPS driver allocations can overlap in unified memory; do not add them.
