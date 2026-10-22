"""Count the total learned parameters of every model on the local inference path.

    python -m scripts.count_params

Loads each component exactly as the server does and counts real tensors (4-bit LLM
weights are counted as unpacked parameters, see app/mlx_params.py). Writes
results/params.json. Works before training too: the CNN is then counted from its
architecture (the parameter count does not depend on the weights).
"""
import json

from app.config import EMOTIONS, RESULTS_DIR, Settings

LIMIT = 6_000_000_000


def main():
    s = Settings()
    rows = []

    from app.vad import SileroVAD
    vad = SileroVAD()
    rows.append(("Silero VAD (voice activity)", "silero-vad pip package",
                 sum(p.numel() for p in vad.model.parameters())))

    from model import AudioCNN
    if s.emotion_checkpoint.exists():
        from app.emotion import EmotionClassifier
        clf = EmotionClassifier(s.emotion_checkpoint, "cpu")
        clf.load()
        n_cnn, src = clf.num_parameters(), str(s.emotion_checkpoint)
    else:
        n_cnn = sum(p.numel() for p in AudioCNN(num_classes=len(EMOTIONS)).parameters())
        src = "architecture only (no checkpoint yet)"
    rows.append(("AudioCNN emotion classifier (ours)", src, n_cnn))

    from app.asr import WhisperASR
    asr = WhisperASR(s.asr_repo)
    rows.append(("Whisper small (speech-to-text)", s.asr_repo, asr.num_parameters()))

    from app.llm import Responder
    llm = Responder(s.llm_repo)
    from mlx_lm import load
    llm.model, llm.tokenizer = load(s.llm_repo)
    rows.append(("Qwen2.5-3B-Instruct (reply generator)", s.llm_repo, llm.num_parameters()))

    rows.append(("Prosody features (librosa YIN/RMS)", "hand-written DSP", 0))
    total = sum(r[2] for r in rows)

    width = max(len(r[0]) for r in rows)
    for name, src, n in rows:
        print(f"{name:<{width}}  {n:>15,}   {src}")
    print(f"{'TOTAL':<{width}}  {total:>15,}   ({total / 1e9:.2f} B of the 6 B limit)")
    assert total <= LIMIT, "over the 6 B parameter budget!"

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"components": [{"name": n, "source": src, "parameters": c} for n, src, c in rows],
           "total_parameters": total, "limit": LIMIT}
    (RESULTS_DIR / "params.json").write_text(json.dumps(out, indent=2))
    print(f"saved {RESULTS_DIR / 'params.json'}")


if __name__ == "__main__":
    main()
