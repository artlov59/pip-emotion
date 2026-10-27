"""Explicit local inference backends; existing MLX construction remains the default."""
from dataclasses import replace

from app.pipeline import Engine


# Purpose: Construct an Engine with the requested local inference backend.
# Input: settings: runtime Settings; backend: mlx or cpu; cpu_threads: Torch thread count.
# Output: Engine instance; models are not loaded yet. Invalid backends/limits raise errors.
def create_engine(settings, backend="mlx", cpu_threads=4):
    if backend == "mlx":
        return Engine(settings)
    if backend != "cpu":
        raise ValueError(f"Unknown local backend: {backend}")
    if not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 4:
        raise ValueError("CPU runtime threads must be between 1 and 4")
    from app.cpu_asr import CPUWhisperASR
    from app.cpu_llm import CPUResponder
    from app.llm import TemplateResponder

    settings = replace(settings, torch_device="cpu")
    return Engine(
        settings,
        asr=CPUWhisperASR(repo=settings.asr_repo, threads=cpu_threads),
        responder=CPUResponder(settings.llm_repo, settings.max_response_tokens, threads=cpu_threads)
        if settings.use_llm else TemplateResponder(),
        backend="cpu", cpu_threads=cpu_threads,
    )
