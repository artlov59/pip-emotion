"""Calibrate the words-vs-delivery check on MELD dev, then report the locked rule on test.

    python -m scripts.calibrate_agreement fit    # dev only -> results/agreement_calibration.{json,md}
    python -m scripts.calibrate_agreement test   # one held-out report -> results/agreement_calibration_test.{json,md}

Why: the original check compared raw confidences against a fixed 0.4 threshold. The audio CNN
is under-confident, the text model and face model are over-confident, so the check flagged a
mismatch on ~37% of turns and, when it did, the words were usually right, not the delivery
the reply prompt was told to follow.

What `fit` does, using only dev predictions the deployed models already produced:
  1. fits one temperature per modality (audio, text, vision) by minimising negative log-likelihood;
  2. sweeps the voice/face weighting and the margin by which the delivery must out-score the
     words before it is flagged;
  3. keeps the setting with the largest net benefit of following the delivery when flagged:
     (# flagged turns where the delivery matches the label) - (# where the words match it).
Every candidate is scored by calling app.fusion.agreement itself, so offline numbers and the
runtime use the same code. The fused emotion in final_state is not affected.

`test` re-applies the saved rule unchanged to the frozen test predictions and adds a paired
dialogue-bootstrap interval. It never writes the calibration file.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app import fusion
from app.config import EMOTIONS, MELD_DIR, RESULTS_DIR, ROOT

P = [f"p_{e}" for e in EMOTIONS]
SOURCES = {
    "dev": {"audio": RESULTS_DIR / "runtime_dev" / "dev_runtime_audio_predictions.csv",
            "text": RESULTS_DIR / "retraining_20261023" / "text_selected" / "text_dev_whisper_predictions.csv",
            "vision": MELD_DIR / "vision_dev.csv"},
    "test": {"audio_text": RESULTS_DIR / "retraining_20261023" / "final_test" / "test_frozen_predictions.csv",
             "vision": MELD_DIR / "vision_test.csv"},
}
OUT_JSON = fusion.AGREEMENT_FILE
OUT_MD = RESULTS_DIR / "agreement_calibration.md"
TEST_JSON = RESULTS_DIR / "agreement_calibration_test.json"
TEST_MD = RESULTS_DIR / "agreement_calibration_test.md"
VISION_WEIGHTS = (1.0, 0.2)  # 1.0 = the previous runtime; 0.2 = dev-tuned audio+vision fusion weight
MARGINS = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
T_GRID = np.exp(np.linspace(np.log(0.05), np.log(10.0), 801))
PREVIOUS_RULE = "raw probabilities, voice:face 1:1, both confidences >= 0.4"
FLAGGED = ("conflict", "tone_only", "words_only", "differ")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load(split: str) -> pd.DataFrame:
    """One row per utterance: id, dialogue, gold label, and 7 probabilities per modality (NaN = missing)."""
    src = SOURCES[split]
    if split == "dev":
        a = pd.read_csv(src["audio"]).set_index("id")
        t = pd.read_csv(src["text"]).set_index("id")[P]
        df = a[["Dialogue_ID", "emotion"] + P].rename(columns={c: f"audio_{c}" for c in P})
        df = df.join(t.rename(columns={c: f"text_{c}" for c in P}))
    else:
        f = pd.read_csv(src["audio_text"]).set_index("id")
        df = f[["Dialogue_ID", "emotion"]].copy()
        for m in ("audio", "text"):
            for c in P:
                df[f"{m}_{c}"] = f[f"selected_{m}_{c}"]
    v = pd.read_csv(src["vision"]).set_index("id")[P]
    df = df.join(v.rename(columns={c: f"vision_{c}" for c in P}))
    return df


def probs(df: pd.DataFrame, mode: str) -> np.ndarray:
    return df[[f"{mode}_{c}" for c in P]].to_numpy(dtype=np.float64)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Grid-search the temperature that minimises dev NLL (rows with missing input are skipped)."""
    ok = ~np.isnan(p).any(1)
    p, y = p[ok], y[ok]
    nll = [-np.log(np.clip(fusion.temperature_scale(p, t)[np.arange(len(y)), y], 1e-12, 1)).mean() for t in T_GRID]
    return float(T_GRID[int(np.argmin(nll))])


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    ok = ~np.isnan(p).any(1)
    p, y = p[ok], y[ok]
    conf, hit = p.max(1), p.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(abs(hit[m].mean() - conf[m].mean()) * m.mean()
                     for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf >= lo) & (conf < hi)).any()))


def reading(row: np.ndarray):
    if np.isnan(row).any():
        return None
    i = int(row.argmax())
    return {"emotion": EMOTIONS[i], "confidence": float(row[i]), "probs": dict(zip(EMOTIONS, row.tolist()))}


def previous_rule(words, voice, face):
    """The rule shipped before calibration, kept here only to measure what changed."""
    nonverbal = fusion.fuse({"audio": voice, "vision": face}, {"audio": 1.0, "vision": 1.0})
    if words is None or nonverbal is None:
        return {"kind": "unknown", "words": None, "nonverbal": None}
    w, n = words["emotion"], nonverbal["emotion"]
    if w == n:
        return {"kind": "agree", "words": w, "nonverbal": n}
    if words["confidence"] < 0.4 or nonverbal["confidence"] < 0.4:
        return {"kind": "unknown", "words": w, "nonverbal": n}
    return {"kind": "flagged", "words": w, "nonverbal": n}


def run_rule(df: pd.DataFrame, rule) -> pd.DataFrame:
    rows = []
    a, t, v = probs(df, "audio"), probs(df, "text"), probs(df, "vision")
    for i in range(len(df)):
        r = rule(reading(t[i]), reading(a[i]), reading(v[i]))
        rows.append((r["kind"], r.get("words"), r.get("nonverbal")))
    out = pd.DataFrame(rows, columns=["kind", "words", "nonverbal"], index=df.index)
    out["gold"], out["dialogue"] = df["emotion"], df["Dialogue_ID"]
    return out


def summarize(out: pd.DataFrame) -> dict:
    flagged = out["kind"].isin(FLAGGED + ("flagged",))
    f = out[flagged]
    deliv, words = int((f["nonverbal"] == f["gold"]).sum()), int((f["words"] == f["gold"]).sum())
    return {"n": int(len(out)), "kinds": {k: int(v) for k, v in out["kind"].value_counts().items()},
            "flagged": int(flagged.sum()), "flagged_rate": round(float(flagged.mean()), 4),
            "flagged_delivery_correct": deliv, "flagged_words_correct": words,
            "flagged_delivery_correct_rate": round(deliv / max(len(f), 1), 4),
            "flagged_words_correct_rate": round(words / max(len(f), 1), 4),
            "net_benefit_of_following_delivery": deliv - words}


def dialogue_bootstrap(out: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> list:
    """95% interval for (delivery-correct rate - words-correct rate) among flagged turns, resampling dialogues."""
    flagged = out[out["kind"].isin(FLAGGED)]
    groups = {d: g for d, g in flagged.groupby("dialogue")}
    keys = list(out["dialogue"].unique())
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        parts = [groups[k] for k in rng.choice(keys, size=len(keys), replace=True) if k in groups]
        if not parts:
            continue
        s = pd.concat(parts)
        stats.append(((s["nonverbal"] == s["gold"]).mean() - (s["words"] == s["gold"]).mean()))
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def fit(_args):
    df = load("dev")
    y = df["emotion"].map(EMOTIONS.index).to_numpy()
    temps, calib_report = {}, {}
    for mode in ("audio", "text", "vision"):
        p = probs(df, mode)
        temps[mode] = round(fit_temperature(p, y), 4)
        calib_report[mode] = {"temperature": temps[mode], "ece_before": round(ece(p, y), 4),
                              "ece_after": round(ece(fusion.temperature_scale(p, temps[mode]), y), 4),
                              "rows": int((~np.isnan(p).any(1)).sum())}
    sweep = []
    for wv in VISION_WEIGHTS:
        for margin in MARGINS:
            cal = {"temperatures": temps, "nonverbal_weights": {"audio": 1.0, "vision": wv}, "margin": margin}
            s = summarize(run_rule(df, lambda w, a, v, cal=cal: fusion.agreement(w, a, v, cal)))
            sweep.append({"vision_weight": wv, "margin": margin, **s})
    # Largest net benefit; ties go to the larger margin (fewer interventions), then the 1:1 weighting.
    best = max(sweep, key=lambda r: (r["net_benefit_of_following_delivery"], r["margin"], r["vision_weight"]))
    before = summarize(run_rule(df, previous_rule))
    spec = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "dev", "temperatures": temps,
        "nonverbal_weights": {"audio": 1.0, "vision": best["vision_weight"]}, "margin": best["margin"],
        "rule": "calibrate each modality with its temperature; delivery = log-linear fusion of calibrated voice "
                "and face; when words and delivery disagree, flag the delivery only if its confidence >= words "
                "confidence + margin",
        "selection": "max over dev of (# flagged where delivery matches gold) - (# flagged where words match gold); "
                     "ties -> larger margin, then vision weight 1.0",
        "calibration": calib_report, "dev_selected": best, "dev_previous_rule": {"rule": PREVIOUS_RULE, **before},
        "sweep": sweep, "sources": {rel(p): sha256(p) for p in SOURCES["dev"].values()},
        "note": "Does not change fusion.fuse or the reported emotion; only the agreement note/tags and the prompt.",
    }
    OUT_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    lines = [
        "# Words-vs-delivery calibration (dev)",
        "",
        "Fitted on MELD dev predictions from the deployed audio CNN, the selected text model on Whisper "
        "transcripts, and cached face readings. Fusion and the reported emotion are unchanged.",
        "",
        "| modality | temperature | ECE before | ECE after |",
        "|---|---:|---:|---:|",
        *[f"| {m} | {r['temperature']:.3f} | {r['ece_before']:.3f} | {r['ece_after']:.3f} |" for m, r in calib_report.items()],
        "",
        "T < 1 means the model was under-confident (sharpened); T > 1 over-confident (softened).",
        "",
        "| rule | flagged turns | delivery right when flagged | words right when flagged | net |",
        "|---|---:|---:|---:|---:|",
        f"| previous: {PREVIOUS_RULE} | {before['flagged']} ({before['flagged_rate']:.1%}) | "
        f"{before['flagged_delivery_correct_rate']:.0%} | {before['flagged_words_correct_rate']:.0%} | "
        f"{before['net_benefit_of_following_delivery']:+d} |",
        f"| selected: calibrated, voice:face 1:{best['vision_weight']:g}, margin {best['margin']:+.2f} | "
        f"{best['flagged']} ({best['flagged_rate']:.1%}) | {best['flagged_delivery_correct_rate']:.0%} | "
        f"{best['flagged_words_correct_rate']:.0%} | {best['net_benefit_of_following_delivery']:+d} |",
        "",
        "Full sweep (vision weight x margin) is in agreement_calibration.json.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def test(_args):
    if not OUT_JSON.exists():
        raise SystemExit("Run `python -m scripts.calibrate_agreement fit` first.")
    cal = fusion.load_agreement_calibration()
    df = load("test")
    after = run_rule(df, lambda w, a, v: fusion.agreement(w, a, v, cal))
    before = run_rule(df, previous_rule)
    s_after, s_before = summarize(after), summarize(before)
    ci = dialogue_bootstrap(after)
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": "test",
              "calibration_file": rel(OUT_JSON), "calibration_sha256": sha256(OUT_JSON),
              "selected": s_after, "previous_rule": {"rule": PREVIOUS_RULE, **s_before},
              "delivery_minus_words_rate_ci95_dialogue_bootstrap": ci,
              "sources": {rel(p): sha256(p) for p in SOURCES["test"].values()},
              "note": "Rule locked on dev before this report; nothing here is fitted on test."}
    TEST_JSON.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Words-vs-delivery check on held-out test",
        "",
        f"Rule locked on dev (`{rel(OUT_JSON)}`, sha256 {report['calibration_sha256'][:12]}...). "
        f"{s_after['n']} test utterances; labelled clip boundaries and cached face readings.",
        "",
        "| rule | flagged turns | delivery right when flagged | words right when flagged | net |",
        "|---|---:|---:|---:|---:|",
        f"| previous | {s_before['flagged']} ({s_before['flagged_rate']:.1%}) | "
        f"{s_before['flagged_delivery_correct_rate']:.0%} | {s_before['flagged_words_correct_rate']:.0%} | "
        f"{s_before['net_benefit_of_following_delivery']:+d} |",
        f"| calibrated | {s_after['flagged']} ({s_after['flagged_rate']:.1%}) | "
        f"{s_after['flagged_delivery_correct_rate']:.0%} | {s_after['flagged_words_correct_rate']:.0%} | "
        f"{s_after['net_benefit_of_following_delivery']:+d} |",
        "",
        f"Calibrated rule, delivery-right minus words-right rate among flagged turns: "
        f"{s_after['flagged_delivery_correct_rate'] - s_after['flagged_words_correct_rate']:+.1%} "
        f"(95% dialogue-bootstrap interval {ci[0]:+.1%} to {ci[1]:+.1%}).",
        "",
        "Neither side is right in many flagged turns, so the reply prompt treats the flag as a hint.",
    ]
    TEST_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit", help="fit on dev and write results/agreement_calibration.json").set_defaults(fn=fit)
    sub.add_parser("test", help="report the locked rule once on test").set_defaults(fn=test)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
