"""Optional experiment: does the Whisper encoder we already run carry more emotion than the CNN?

The live system already runs Whisper-small for transcripts. If its encoder states predict emotion
better than the from-scratch AudioCNN, the voice model could become a 5,383-parameter linear head
on states that are computed anyway, instead of a separate 21M-parameter network.

This is a frozen-encoder linear probe, dev only:
  1. mean-pool selected Whisper-small encoder layers over the speech frames of each clip
     (train + dev, cached to data/meld/whisper_encoder_<split>.npz);
  2. fit a standardised logistic-regression head on train for each layer, C and class weighting;
  3. report the dev scores next to the deployed CNN's dev scores. Test is never read.

    python -m training.probe_whisper_encoder --limit 64   # smoke test, about a minute
    python -m training.probe_whisper_encoder              # full run; feature extraction dominates

Requires the prepared MELD wavs (training.prepare_meld) and downloads openai/whisper-small
(same architecture and parameter count as the MLX copy used at runtime) on first use.
Not wired into the runtime: adopting it would mean exposing encoder states from the MLX
transcription call and re-running the locked dev -> test protocol.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.config import EMOTIONS, MELD_DIR, RESULTS_DIR, SAMPLE_RATE
from app.features import load_audio

MODEL_ID = "openai/whisper-small"
LAYERS = (4, 6, 8, 10, 12)          # hidden_states index; 12 = final encoder layer
MAX_SECONDS = 30                    # Whisper's fixed input window; longer clips keep their last 30 s
FRAMES_PER_SECOND = 50              # encoder output rate (100 mel frames/s, stride-2 conv)
C_GRID = (0.01, 0.1, 1.0)
CLASS_WEIGHTS = (None, "balanced")
CNN_DEV = RESULTS_DIR / "runtime_dev" / "dev_runtime_audio_metrics.json"
OUT_JSON, OUT_MD = RESULTS_DIR / "whisper_probe_dev.json", RESULTS_DIR / "whisper_probe_dev.md"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def manifest(split: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(MELD_DIR / f"{split}.csv")
    df = df[df["status"] == "ok"].reset_index(drop=True)
    return df.head(limit) if limit else df


@torch.inference_mode()
def extract(df: pd.DataFrame, model, extractor, device, batch_size: int) -> np.ndarray:
    """(n_clips, len(LAYERS), 768) mean-pooled encoder states over each clip's speech frames."""
    out = np.zeros((len(df), len(LAYERS), model.config.d_model), dtype=np.float32)
    t0 = time.perf_counter()
    for start in range(0, len(df), batch_size):
        rows = df.iloc[start:start + batch_size]
        wavs = [load_audio(MELD_DIR / p)[-MAX_SECONDS * SAMPLE_RATE:] for p in rows["wav_path"]]
        feats = extractor(wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
        hidden = model.encoder(feats, output_hidden_states=True).hidden_states
        n_valid = [max(1, min(1500, int(np.ceil(w.size / SAMPLE_RATE * FRAMES_PER_SECOND)))) for w in wavs]
        for li, layer in enumerate(LAYERS):
            h = hidden[layer].float().cpu().numpy()
            for bi, n in enumerate(n_valid):
                out[start + bi, li] = h[bi, :n].mean(axis=0)
        done = start + len(rows)
        if done % (batch_size * 25) == 0 or done == len(df):
            rate = done / (time.perf_counter() - t0)
            print(f"  {done}/{len(df)} clips, {rate:.1f} clips/s, ~{(len(df) - done) / rate / 60:.1f} min left", flush=True)
    return out


def features(split, df, args, model_cache):
    cache = MELD_DIR / f"whisper_encoder_{split}{'_limit' + str(args.limit) if args.limit else ''}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == list(df["id"]) and tuple(z["layers"]) == LAYERS:
            print(f"[probe] using cached {cache.name}")
            return z["x"]
    if not model_cache:
        from transformers import WhisperFeatureExtractor, WhisperModel
        device = pick_device(args.device)
        print(f"[probe] loading {MODEL_ID} encoder on {device}")
        model_cache["extractor"] = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
        model_cache["model"] = WhisperModel.from_pretrained(MODEL_ID).to(device).eval()
        model_cache["device"] = device
    print(f"[probe] extracting {split} ({len(df)} clips)")
    x = extract(df, model_cache["model"], model_cache["extractor"], model_cache["device"], args.batch_size)
    np.savez(cache, x=x, ids=df["id"].to_numpy(dtype=str), layers=np.array(LAYERS))
    return x


def main():
    ap = argparse.ArgumentParser(description="Frozen Whisper-small encoder linear probe on MELD dev (no test).")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train and dev clips (smoke test)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto | mps | cpu")
    args = ap.parse_args()

    tr, dv = manifest("train", args.limit), manifest("dev", args.limit)
    cache = {}
    x_tr, x_dv = features("train", tr, args, cache), features("dev", dv, args, cache)
    y_tr, y_dv = tr["emotion"].map(EMOTIONS.index).to_numpy(), dv["emotion"].map(EMOTIONS.index).to_numpy()

    runs = []
    for li, layer in enumerate(LAYERS):
        for c in C_GRID:
            for cw in CLASS_WEIGHTS:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight=cw, max_iter=3000))
                clf.fit(x_tr[:, li], y_tr)
                pred = clf.predict(x_dv[:, li])
                runs.append({"layer": layer, "C": c, "class_weight": cw or "none",
                             "dev_weighted_f1": round(float(f1_score(y_dv, pred, average="weighted")), 4),
                             "dev_macro_f1": round(float(f1_score(y_dv, pred, average="macro")), 4),
                             "dev_accuracy": round(float(accuracy_score(y_dv, pred)), 4)})
                print(f"  layer {layer:2d} C={c:<5} weights={cw or 'none':8s} dev wF1 {runs[-1]['dev_weighted_f1']:.4f} "
                      f"macro {runs[-1]['dev_macro_f1']:.4f}", flush=True)
    best = max(runs, key=lambda r: (r["dev_weighted_f1"], r["dev_macro_f1"]))
    cnn = json.loads(CNN_DEV.read_text()) if CNN_DEV.exists() else {}
    report = {"model": MODEL_ID, "layers": LAYERS, "train_clips": len(tr), "dev_clips": len(dv), "limit": args.limit,
              "head_parameters": 768 * len(EMOTIONS) + len(EMOTIONS), "best_on_dev": best, "runs": runs,
              "cnn_dev": {k: cnn.get(k) for k in ("weighted_f1", "macro_f1", "accuracy")},
              "note": "Dev only. Selection over layer/C/class weighting on dev makes this optimistic; "
                      "an adopted probe still needs the locked dev -> test comparison."}
    lines = ["# Frozen Whisper-small encoder probe vs AudioCNN (dev)", "",
             f"{len(tr)} train / {len(dv)} dev clips{' (smoke test)' if args.limit else ''}. Test not read.", "",
             "| voice model | learned params added | dev weighted F1 | dev macro F1 | dev accuracy |",
             "|---|---:|---:|---:|---:|",
             f"| AudioCNN (deployed) | 21,281,991 | {cnn.get('weighted_f1', float('nan')):.3f} | "
             f"{cnn.get('macro_f1', float('nan')):.3f} | {cnn.get('accuracy', float('nan')):.3f} |",
             f"| Whisper encoder layer {best['layer']} + linear head (C={best['C']}, {best['class_weight']}) | "
             f"{report['head_parameters']:,} | {best['dev_weighted_f1']:.3f} | {best['dev_macro_f1']:.3f} | "
             f"{best['dev_accuracy']:.3f} |", "",
             "The probe's row is the best of the grid on dev, so it is optimistic.",
             "All runs are in whisper_probe_dev.json."]
    print("\n".join(lines))
    if args.limit:
        print("[probe] smoke test only; results/ not written")
    else:
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        OUT_MD.write_text("\n".join(lines) + "\n")
        print(f"[probe] wrote {OUT_MD.name} and {OUT_JSON.name}")


if __name__ == "__main__":
    main()
