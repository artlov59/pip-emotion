---
name: dev-experiment
description: Runs one model, fusion, calibration or threshold change through the dev-only comparison protocol and records the decision. Use when trying to improve accuracy, retraining the audio CNN or text model, changing fusion weights, or asking "does X help?".
---

# Dev-only experiment

One change per experiment. Test data is never read here (see `.claude/rules/evaluation.md`).

## 1. Write the plan first

Create `results/<YYYYMMDD>_<short-name>/plan.md` with: the hypothesis, the single change, the
baseline (current recipe from `results/submission_model_selection.json`), and the adoption rule
written before seeing results: dev weighted F1 up, macro F1 not down.

## 2. Produce the candidate (train/dev only, new output paths)

- Audio CNN: `python -m training.train --run-name <name>`, then
  `python -m training.evaluate audio --checkpoint checkpoints/<name>/best.pt --output-dir results/<dir>/audio_dev`
- Text model: `python -m training.train_text ... --output-dir checkpoints/<name>` (dev is evaluated
  by default; never pass `--eval-splits test` here).
- Fusion, calibration, thresholds: re-score saved dev predictions through the runtime functions
  (`app.fusion.fuse` with `fusion.runtime_weights`, `fusion.agreement`). Inputs:
  `results/runtime_dev/dev_runtime_audio_predictions.csv`,
  `results/retraining_20261023/text_selected/text_dev_whisper_predictions.csv`,
  `data/meld/vision_dev.csv`. First reproduce the deployed dev score (weighted F1 0.5321) to
  prove the offline path matches the runtime, then apply the change.
- Prompt changes: cannot be scored on MELD; describe them and ask the user to try them live.

## 3. Compare on dev

- Whole recipe: `python -m training.evaluate compare --baseline-audio ... --candidate-audio ...
  --baseline-text ... --candidate-text ... --output-dir results/<dir>/compare_dev`
- Otherwise report weighted F1, macro F1, accuracy, per-class F1, the always-neutral baseline,
  and a paired dialogue-bootstrap 95% interval (`paired_bootstrap` in `scripts/report_baselines.py`).

## 4. Decide and record

Write `results/<dir>/summary.md`: numbers, interval, decision (adopt / reject / no evidence) and
why. Rejected experiments are kept; they are evidence too. If adopted, stop and hand over to the
`locked-test-report` skill; do not update README numbers from dev alone.
