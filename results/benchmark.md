36 utterances from 35 MELD test clips, streamed in real time. End-of-speech is declared after 500 ms of silence; the latencies below are measured *after* that point.

| stage | p50 | p95 |
|---|---|---|
| provisional emotion (CNN, 3 s window) | nan | nan |
| final emotion (CNN) | 54.2 | 80.9 |
| Whisper transcript | 89.6 | 128.0 |
| end of speech → final_state | 91.0 | 129.3 |
| end of speech → first reply token | 186.3 | 239.6 |
| end of speech → full reply | 310.4 | 372.3 |
| LLM tokens / s | 69.9 | 78.7 |

Hardware: Apple M5 Pro · 48.0 GB RAM · macOS 26.6.2  
Model load: 7.1 s · peak process RSS 3.52 GB · MLX peak 2.96 GB · torch-MPS 2.2 GB · avg CPU 10.6% (100% = one core)
