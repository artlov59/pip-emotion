[voice + words + face; backend=cpu] 36 utterances from 35 MELD test clips, streamed at 1x. Latencies exclude 500 ms of end silence.

in-process Session; excludes WebSocket, network and browser capture overhead.

| stage | p50 | p95 |
|---|---|---|
| provisional CNN (per-utterance mean compute) | 14.9 | 17.7 |
| face analysis (per-clip mean compute) | 11.9 | 12.8 |
| final emotion (CNN) | 17.9 | 24.3 |
| Whisper transcript | 465.9 | 574.3 |
| words emotion (after Whisper) | 5.4 | 5.8 |
| end of speech → final_state | 483.2 | 822.6 |
| end of speech → first visible reply delta | 1461.0 | 2447.7 |
| end of speech → full reply | 3403.4 | 4816.8 |
| visible reply deltas / s | 6.2 | 6.8 |

Clips with no detected speech: 3; with errors: 0; completed replies: 32.
Frames sent/dropped: 527/0.
Hardware: None · 48.0 GB RAM · macOS 26.6.2
Model load 8.0 s · peak RSS 14.64 GB · MLX peak None GB · MPS driver None GB · CPU 95.7% (100% = one core).
RSS, MLX peak and MPS driver allocations can overlap in unified memory; do not add them.
