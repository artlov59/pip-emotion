---
name: benchmark-and-document
description: Measures end-to-end latency and memory on the user's Mac and updates the README latency and hardware tables from the benchmark output. Use after runtime changes, model swaps, or when asked how fast the app is.
---

# Benchmark and update the README

The default backend is MLX and needs the Apple GPU. A Linux container or VM cannot run it, so the
user runs these commands on the Mac and pastes the output.

1. Ask the user to run, from the repo with `.venv` active:
   ```bash
   sysctl -n machdep.cpu.brand_string
   python -m unittest discover -s tests && python -m tests.test_pipeline && node tests/test_frontend.js
   python -m scripts.benchmark_latency --n 35 --output-dir results/benchmark_mlx_<YYYYMMDD>
   ```
   Use a new dated directory so earlier benchmarks stay as evidence. CPU fallback:
   add `--backend cpu --cpu-threads 4`.
2. Read `benchmark_multimodal.md` (and `.json` for load time, peak RSS, MLX peak, CPU %).
3. Update README sections "What real-time means here" (latency table: state, first reply words,
   full reply, Whisper; p50 and p95) and "Hardware, resources and parameter budget". Copy numbers,
   round consistently (seconds to 2 decimals, GiB to 1), and link the new results file.
4. State what the benchmark excludes: browser and WebSocket overhead, and the 500 ms end-of-turn
   silence (add it when describing the user's total wait).
5. Commit the results directory and README together.
