# ML practice checklist for this repo

Condensed from published checklists, each item mapped to what this repo does today.
Status: **done**, **partial** (done with a known caveat), **open** (not done; say so if asked).

Sources:
- Kapoor & Narayanan, *Leakage and the reproducibility crisis in ML-based science* (Patterns, 2023):
  https://reproducible.cs.princeton.edu/ (leakage taxonomy L1-L3)
- Kapoor et al., *REFORMS: consensus-based recommendations for ML-based science* (2024):
  https://reforms.cs.princeton.edu/ (8 modules: design, reproducibility, data quality,
  preprocessing, modeling, leakage, metrics and uncertainty, generalizability)
- Zinkevich, *Rules of Machine Learning*: https://developers.google.com/machine-learning/guides/rules-of-ml
- Breck et al., *The ML Test Score* (Google, 2017):
  https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
- Mitchell et al., *Model Cards for Model Reporting* (2019): https://arxiv.org/abs/1810.03993

## 1. Leakage (Kapoor & Narayanan)

| item | status | how / caveat |
|---|---|---|
| L1.1 a held-out test set exists | done | MELD official test split, opened once after a hash lock |
| L1.2 no preprocessing fitted on all data | done | per-utterance log-mel normalisation; class weights from train labels; temperatures, margins and fusion weights from dev |
| L1.3 no selection using test | done | every model, weight, threshold chosen on dev; `results/submission_model_selection.json` records `test_used_for_selection: false` |
| L1.4 duplicates across splits | done (checked) | 15% of dev/test lines have identical text in train, almost all interjections ("hey", "yeah", "okay"); the train majority label matches only ~60% of them, and the audio differs. Not label leakage |
| L2 only inputs available at prediction time | done | runtime uses Whisper transcripts, never MELD gold text; gold text only for training mix and the ASR-gap analysis |
| L3.2 non-independence | partial | dialogues are correlated: intervals use a dialogue bootstrap. Speakers are shared: ~96% of dev/test lines come from speakers seen in training (82-86% from the six main characters), so scores are speaker-dependent |
| L3.3 test distribution = target use | open | MELD is acted TV dialogue; the target is one live user at a laptop. Not measured |

## 2. Metrics and uncertainty (REFORMS 7)

| item | status | how / caveat |
|---|---|---|
| metric matches the task convention | done | weighted F1 (MELD standard) + macro F1 + accuracy |
| trivial baseline reported | done | always-neutral in `results/baselines.md` (ML Test Score: a simpler model is not better) |
| per-slice results | partial | per-class F1 and per-input increments; no per-speaker or live-user slices |
| uncertainty on comparisons | done | paired dialogue-bootstrap 95% intervals for every claimed gain |
| calibrated confidences | partial | agreement check uses dev-fitted temperatures (ECE reported); fused state confidences are raw |

## 3. Training/serving parity (Rules of ML #29, #32; ML Test Score)

| item | status | how / caveat |
|---|---|---|
| same feature code in training and serving | done | `app/features.py`; `FEATURE_CONFIG` stored in checkpoints and checked at load |
| offline scoring calls runtime code | done | rebuilding fusion from saved dev predictions reproduces the deployed dev predictions exactly (`tests/test_fusion_parity.py`, skips without MELD caches) |
| text model trained on serving-like input | done | mixed gold + Whisper transcripts |
| backend parity | partial | CPU backend reuses MLX weights but decodes differently; numerical parity not claimed |
| serving-time input checks | partial | Whisper repetition loops (compression ratio > 2.4) are tagged `asr:repetitive` and kept out of the reply prompt; dropping them from emotion fusion didn't help on dev (`results/asr_repetition_dev.md`) |

## 4. Reproducibility (REFORMS 2)

| item | status | how / caveat |
|---|---|---|
| pinned environment | done | `requirements-verified.txt` (Python 3.12) |
| pinned model versions + hashes | done | `scripts/cache_models.py` revisions; checksummed `scripts/download_models.py` |
| seeds | done | seed 0 in training scripts; GPU kernels may still be non-deterministic |
| data provenance | done | MELD not redistributed; `training.prepare_meld` rebuilds manifests; prep report in `data/meld/prep_report.json` |
| every number traceable | done | README figures come from `results/` files that record input hashes |

## 5. Testing (Rules of ML #5; ML Test Score)

| item | status | how / caveat |
|---|---|---|
| infrastructure tested without real models | done | fake VAD/ASR/models drive `tests/test_pipeline.py` and session regressions |
| protocol and UI contract tested | done | `tests/test_protocol.py`, `tests/test_frontend.js` |
| model integrity checks | done | label order, feature config and hashes verified at load |
| live quality monitoring | open | out of scope for a prototype |

## 6. Simplicity and reporting (Rules of ML #4; Model Cards)

| item | status | how / caveat |
|---|---|---|
| simple, explainable first model | done | late fusion with 17 dev-tuned coefficients; learned fusion tried and rejected on dev |
| intended use, metrics, limitations documented | done | README: track, real-time definition, results, limitations, left-out work |
| subgroup and live-user evaluation | open | no demographic or live-recording evaluation set |
| reply quality evaluated | open | prompt changes are checked by hand only |
