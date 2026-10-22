#!/usr/bin/env bash
# Full reproduction of the emotion model: data prep -> smoke test -> 3 training runs -> dev selection -> test.
# Usage:  bash scripts/run_training.sh        (logs to results/logs/)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
# use the project venv even if it isn't activated in this shell
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
python() { "$PY" "$@"; }
export PYTHONUNBUFFERED=1   # keep log lines in order
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a results/logs/run_training.log; }

log "1/6 prepare MELD (idempotent)"
python -m training.prepare_meld 2>&1 | tee results/logs/prepare_meld.log

log "2/6 smoke test (1 epoch on 256 clips)"
python -m training.train --run-name _smoke --epochs 1 --limit 256 2>&1 | tee results/logs/smoke.log
rm -rf checkpoints/_smoke runs/_smoke

for cfg in "ce_plain --class-weighting none" \
           "wce_sqrt --class-weighting sqrt" \
           "wce_sqrt_mixup --class-weighting sqrt --mixup-prob 0.5"; do
  set -- $cfg
  name=$1; shift
  log "3/6 train $name $*"
  python -m training.train --run-name "$name" "$@" 2>&1 | tee "results/logs/train_$name.log"
done

log "4/6 select on dev + promote to checkpoints/emotion_cnn.pt"
python -m training.evaluate --split dev --checkpoints checkpoints/ce_plain/best.pt \
  checkpoints/wce_sqrt/best.pt checkpoints/wce_sqrt_mixup/best.pt --promote 2>&1 | tee results/logs/eval_dev.log

log "5/6 final test evaluation (promoted model only)"
python -m training.evaluate --split test --checkpoints checkpoints/emotion_cnn.pt 2>&1 | tee results/logs/eval_test.log

log "6/6 done"
