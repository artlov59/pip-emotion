"""Put every model next to the trivial baseline, using predictions already saved in results/.

    python -m scripts.report_baselines   # -> results/baselines.{md,json}

No model is run and nothing is selected here; this only re-scores saved prediction files.
"Always neutral" is the majority-class baseline: it predicts the most common MELD label for every clip.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from app.config import EMOTIONS, RESULTS_DIR

R = RESULTS_DIR / "retraining_20261023"
DEV = {
    "audio CNN": (RESULTS_DIR / "runtime_dev" / "dev_runtime_audio_predictions.csv", "pred"),
    "text (Whisper transcript)": (R / "text_selected" / "text_dev_whisper_predictions.csv", "pred"),
    "text (MELD gold transcript)": (R / "text_selected" / "text_dev_gold_predictions.csv", "pred"),
    "audio + text + face (deployed fusion)": (R / "dev_comparison" / "dev_multimodal_predictions.csv", "audio0_text1"),
}
TEST_FILE = R / "final_test" / "test_frozen_predictions.csv"
TEST = {"audio CNN": "selected_audio", "text (Whisper transcript)": "selected_text",
        "audio + text": "selected_audio_text", "audio + text + face (deployed fusion)": "selected_audio_text_vision"}
OUT_MD, OUT_JSON = RESULTS_DIR / "baselines.md", RESULTS_DIR / "baselines.json"
# What each extra input adds on test, as paired comparisons of saved predictions.
INCREMENTS = {"voice on top of words": ("selected_text", "selected_audio_text"),
              "face on top of words + voice": ("selected_audio_text", "selected_audio_text_vision")}


def score(gold: pd.Series, pred: pd.Series) -> dict:
    gold, pred = gold.astype(str), pred.fillna("neutral").astype(str)  # empty transcript -> neutral
    per_class = f1_score(gold, pred, labels=EMOTIONS, average=None, zero_division=0)
    return {"weighted_f1": round(float(f1_score(gold, pred, labels=EMOTIONS, average="weighted", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(gold, pred, labels=EMOTIONS, average="macro", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(gold, pred)), 4),
            "per_class_f1": {e: round(float(v), 4) for e, v in zip(EMOTIONS, per_class)},
            "never_predicted": [e for e in EMOTIONS if not (pred == e).any()], "n": int(len(gold))}


def paired_bootstrap(df, before, after, n_boot=2000, seed=0):
    """95% interval of the weighted-F1 change, resampling whole dialogues (utterances in a dialogue are correlated)."""
    gold = df["emotion"].astype(str).to_numpy()
    a, b = (df[c].fillna("neutral").astype(str).to_numpy() for c in (before, after))
    groups = [np.flatnonzero(df["Dialogue_ID"].to_numpy() == d) for d in df["Dialogue_ID"].unique()]
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        wf1 = lambda p: f1_score(gold[idx], p[idx], labels=EMOTIONS, average="weighted", zero_division=0)
        deltas.append(wf1(b) - wf1(a))
    point = (f1_score(gold, b, labels=EMOTIONS, average="weighted", zero_division=0)
             - f1_score(gold, a, labels=EMOTIONS, average="weighted", zero_division=0))
    return {"delta_weighted_f1": round(float(point), 4),
            "ci95": [round(float(np.percentile(deltas, q)), 4) for q in (2.5, 97.5)]}


def main():
    report = {"dev": {}, "test": {}}
    for name, (path, col) in DEV.items():
        df = pd.read_csv(path)
        if not report["dev"]:
            report["dev"]["always neutral"] = score(df["emotion"], pd.Series("neutral", index=df.index))
        report["dev"][name] = score(df["emotion"], df[col])
    test = pd.read_csv(TEST_FILE)
    report["test"]["always neutral"] = score(test["emotion"], pd.Series("neutral", index=test.index))
    for name, col in TEST.items():
        report["test"][name] = score(test["emotion"], test[col])
    report["test_increments"] = {name: paired_bootstrap(test, *cols) for name, cols in INCREMENTS.items()}
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Baselines and per-modality scores", "",
             "Re-scored from saved predictions on labelled MELD clips (not streaming VAD). Dev picked every "
             "model and weight; test was opened once after that.", ""]
    for split in ("dev", "test"):
        lines += [f"## {split} ({next(iter(report[split].values()))['n']} clips)", "",
                  "| model | weighted F1 | macro F1 | accuracy | never predicts |", "|---|---:|---:|---:|---|"]
        for name, s in report[split].items():
            never = ", ".join(s["never_predicted"]) if len(s["never_predicted"]) < 6 else "all but neutral"
            lines.append(f"| {name} | {s['weighted_f1']:.3f} | {s['macro_f1']:.3f} | {s['accuracy']:.3f} | {never or '-'} |")
        lines.append("")
    lines += ["## Per-class F1 on test", "", "| model | " + " | ".join(EMOTIONS) + " |",
              "|---|" + "---:|" * len(EMOTIONS)]
    for name in ("audio CNN", "text (Whisper transcript)", "audio + text + face (deployed fusion)"):
        lines.append(f"| {name} | " + " | ".join(f"{report['test'][name]['per_class_f1'][e]:.2f}" for e in EMOTIONS) + " |")
    lines += ["", "## What each input adds on test (paired, dialogue bootstrap)", "",
              "| change | weighted F1 change | 95% interval |", "|---|---:|---:|"]
    for name, r in report["test_increments"].items():
        lines.append(f"| {name} | {r['delta_weighted_f1'] * 100:+.1f} pts | "
                     f"{r['ci95'][0] * 100:+.1f} to {r['ci95'][1] * 100:+.1f} pts |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
