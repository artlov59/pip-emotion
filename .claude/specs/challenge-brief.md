# Challenge brief and requirement map

The assignment as received, then where each requirement is met. Keep the map current when files move.

## Brief (verbatim)

> **Real-Time Multimodal ML Challenge**
>
> **Context.** We are exploring emotion-aware interactions for a character robot. Tone, expression,
> and timing can change how words should be understood. Build a small prototype that turns
> multimodal signals into real-time output. Use MELD (Multimodal EmotionLines Dataset) as the shared
> foundation. It provides conversational text, audio, video, and emotion labels.
>
> **Assignment.**
> - Choose one primary track: Text and Audio and Vision
> - Produce structured state or tags containing a MELD emotion category, plus short response text
>   grounded in the multimodal input.
> - Accept input over time and return results suitable for interaction.
> - Deliver one coherent end-to-end prototype.
>
> **Scope and freedom.**
> - Define "real-time" for your intended interaction and explain your assumptions.
> - Choose the problem framing, models, architecture, interfaces, and tools.
> - Prefer a focused, understandable solution over a large unfinished one.
>
> **Optional standout extension.**
> - After completing the core task, optionally combine vision, voice, and text in one real-time
>   system producing structured state or tags and streaming text.
> - Show an interaction where all three inputs work together meaningfully.
> - Optionally apply reinforcement learning to one component and demonstrate measured value.
>
> **Demonstration and handoff.**
> - Submit a runnable repository with clear guidance.
> - Demonstrate one input travelling through the system to both outputs.
> - Explain the decisions, evidence, trade-offs, and limitations.
>
> **Constraints.**
> - The complete local inference path may contain at most 6 billion total learned parameters,
>   counting every required encoder, backbone, adapter, projector, head, and auxiliary generator.
>   Use total, not active, effective, or trainable, parameters.
> - Report the hardware and observed resource requirements.
> - Pretrained models, AI tools, and open-source tools are allowed. Remote inference does not
>   satisfy the core task.
> - Identify important external or generated components and own their results.
> - State what you completed and what you intentionally left out.
>
> **Timebox.** Spend no more than two to three days. A small, correct, well-explained prototype is
> preferred over additional incomplete experiments.
>
> **Use of AI tools.** AI-assisted development tools are allowed. You remain responsible for the
> code, experiments, results, and technical decisions, and should be able to explain and modify the work.

## Requirement map

| requirement | status | where |
|---|---|---|
| MELD as foundation | done | `training/prepare_meld.py`; official splits; `results/` |
| one primary track | done | Audio (README "Track and scope"); AudioCNN in `model.py` |
| structured state with a MELD emotion + tags | done | `final_state` in `app/pipeline.py`; example in README; `tests/test_pipeline.py` |
| short reply grounded in the input | done | `app/llm.py` (`build_messages`); `response_delta` / `response_done` events |
| input over time, interactive output | done | streaming VAD, `partial_state` every 0.5 s; `results/benchmark_mlx/` |
| one coherent end-to-end prototype | done | `app/server.py` + `app/static/` UI |
| define real-time + assumptions | done | README "What real-time means here" |
| focused and understandable | partial | large tooling surface (~9k lines incl. tests); README leads with the essentials |
| all three inputs together (optional) | done | fusion + words-vs-delivery check; README results and calibration tables |
| interaction where all three matter (optional) | partial | example trace; the demo video is still to record |
| reinforcement learning (optional) | left out | stated in README |
| runnable repo with guidance | done | README "Run it"; `scripts/download_models.py`; fresh-clone check |
| one input through to both outputs | done | `results/example_trace_multimodal.json`; README example |
| decisions, evidence, trade-offs, limitations | done | README sections of the same names |
| <= 6B total learned parameters | done | 3,434,439,901; `scripts/count_params_offline.py` |
| hardware and observed resources | done | README "Hardware, resources and parameter budget" |
| local inference only | done | all models run locally; `--require-multimodal` refuses missing models |
| external components identified | done | README "External components and AI assistance" |
| completed vs left out | done | README "Completed and left out" |
| own the AI-assisted work | ongoing | the author must be able to explain every module; see `.claude/CLAUDE.md` |
