---
name: locked-test-report
description: Locks an adopted dev recipe by hash and runs the one-time MELD test evaluation, then updates the README from the outputs. Run only when the user explicitly asks for a final test number.
disable-model-invocation: true
---

# Locked test report

Only after a `dev-experiment` ended in "adopt". Confirm with the user before step 3: this spends
the test set for this recipe.

1. **Lock.** Write a new receipt next to `results/submission_model_selection.json` (do not edit the
   old one) listing sha256 for the audio checkpoint, every text-model file, `results/fusion_weights.json`
   and `results/agreement_calibration.json`, plus `"selection_split": "dev"` and
   `"test_used_for_selection": false`. Commit the receipt before any test read.
2. **Check the budget.** `python -m scripts.count_params_offline --require-multimodal` must report
   within the 6B limit.
3. **Run test once.**
   `python -m training.evaluate test --receipt <new receipt> --output-dir results/<YYYYMMDD>_final_test`
   If the agreement calibration changed: `python -m scripts.calibrate_agreement test`.
4. **Report.** `python -m scripts.report_baselines`, then update README tables by copying from the
   generated `.md` files. Keep the previous results directory; note what changed and why.
5. **If the number is worse than before,** it is still the number. Do not tune and re-run test.
