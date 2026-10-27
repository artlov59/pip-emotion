---
paths:
  - "training/**/*.py"
  - "scripts/calibrate_agreement.py"
  - "scripts/report_baselines.py"
  - "scripts/benchmark_latency.py"
  - "app/fusion.py"
  - "results/**"
---

# Evaluation and data rules

These follow the leakage taxonomy of Kapoor & Narayanan (2023) and the REFORMS checklist; see
`.claude/specs/ml-practices.md` for sources and the repo's status on each item.

## Splits

- Use MELD's official train/dev/test splits from `data/meld/{train,dev,test}.csv`, rows with
  `status == "ok"` (9,877 / 1,092 / 2,576 clips). Never re-split or move clips between splits.
- Anything fitted (weights, normalisation, temperatures, thresholds, class weights, early stopping,
  checkpoint choice, prompt wording judged by scores) uses train or dev only.
- Test inputs: `data/meld/test.csv`, `data/meld/wav/test/`, `data/meld/whisper_test.csv`,
  `data/meld/vision_test.csv`, `data/meld/features/test.pkl`, and anything under
  `results/**/final_test/`. Do not open them while choosing between options.

## Selecting a change

- Compare a candidate with the current recipe on dev, paired on the same clips.
- Report weighted F1 (primary, MELD convention), macro F1 and accuracy, plus per-class F1.
- Add a 95% interval from a paired bootstrap that resamples whole dialogues (utterances in a
  dialogue are correlated). `scripts/report_baselines.py` has `paired_bootstrap` to reuse.
- Adopt only if dev weighted F1 improves and macro F1 does not drop. Small changes whose interval
  spans zero are "no evidence of improvement", not wins. Say so in writing.
- Always show the majority-class ("always neutral") baseline next to new numbers.

## Locking and test

- Lock = record sha256 of every checkpoint, weight file and calibration file in a receipt before
  any test read. `training.evaluate test` verifies hashes before opening test inputs.
- Run test once per locked recipe and write to a new directory. If a test number disappoints,
  it is still the reported number; do not iterate on test.

## Parity and evidence

- Offline scoring must call the runtime code (`app.fusion.fuse`, `fusion.runtime_weights`,
  `fusion.agreement`, `app.features`) instead of re-implementing it.
- Every evaluation output records its inputs' sha256 and the command that produced it.
- Never overwrite `results/` evidence; `training.evaluate` refuses existing `--output-dir`.
- Latency claims state backend, hardware, clip count, and that the 500 ms end-of-turn silence is
  excluded from the measured stages.
