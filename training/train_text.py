"""Train an isolated MELD text-emotion candidate, then evaluate dev only by default.

    python -m training.train_text --train-source mixed --selection-source whisper \
        --init-model-dir checkpoints/text_emotion --device cpu --threads 4
    python -m training.train_text --eval-only --output-dir checkpoints/text_emotion_candidate \
        --eval-splits test --results-dir results/text_candidate_test

Mixed training picks one gold/ASR representation per utterance per epoch; empty ASR falls
back to gold. Capped ASR generations are excluded without changing raw caches or coverage.
--preserve-baseline-macro saves the explicit warm-start as epoch 0 and only replaces it
with higher dev weighted F1 while preserving its macro F1. All training/selection inputs
are train/dev. Test evaluation is an explicit,
evaluation-only operation. Existing model/output directories are never overwritten by training.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from app.config import CHECKPOINT_DIR, EMOTIONS, MELD_DIR, RESULTS_DIR
from app.text_emotion import MAX_LEN
from training.metrics_utils import metrics
from training.train import class_weights, pick_device

BASE = "distilbert/distilroberta-base"
OUT_DIR = CHECKPOINT_DIR / "text_emotion_candidate"


# Purpose: Record an input's exact file identity.
# Input: path: local file.
# Output: Dict containing resolved path and SHA-256 digest.
def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


# Purpose: Load a split and validate unique IDs and MELD labels.
# Input: meld_dir: prepared dataset directory; split: manifest split name.
# Output: Tuple (validated frame with gold text, source fingerprint).
def read_manifest(meld_dir, split):
    path = Path(meld_dir) / f"{split}.csv"
    frame = pd.read_csv(path)
    required = {"id", "utterance", "label", "emotion", "status"}
    if not required.issubset(frame.columns) or frame.id.isna().any() or not frame.id.is_unique:
        raise ValueError(f"Invalid or duplicate ids in {split} manifest")
    labels = pd.to_numeric(frame.label, errors="raise")
    if not labels.isin(range(len(EMOTIONS))).all():
        raise ValueError(f"Invalid labels in {split} manifest")
    frame["label"] = labels.astype(int)
    frame["gold"] = frame.utterance.fillna("").astype(str).str.strip()
    return frame, fingerprint(path)


# Purpose: Join transcripts while checking coverage, provenance, and generation quality.
# Input: frame: manifest rows; cache_path: ASR CSV; split: expected split; allow_partial/min_coverage:
#          explicit coverage policy.
# Output: Tuple (joined frame, provenance dict); capped transcripts become empty without modifying the raw
#          cache.
def attach_asr(frame, cache_path, split, allow_partial=False, min_coverage=1.0):
    """Coverage is cached IDs / audio-usable IDs; an empty recognized transcript is covered."""
    frame = frame.reset_index(drop=True)
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"ASR cache missing for {split}: {path}. Generate real Whisper transcripts first.")
    cache = pd.read_csv(path)
    if not {"id", "whisper"}.issubset(cache.columns) or cache.id.isna().any() or not cache.id.is_unique:
        raise ValueError(f"Invalid or duplicate ids in ASR cache: {path}")
    if "split" in cache and not cache["split"].eq(split).all():
        raise ValueError(f"ASR cache split disagrees with requested {split}")
    unknown = set(cache.id) - set(frame.id)
    if unknown:
        raise ValueError(f"ASR cache contains {len(unknown)} ids outside the {split} manifest")
    eligible = frame.status.eq("ok")
    expected = int(eligible.sum())
    covered = int(frame.loc[eligible, "id"].isin(cache.id).sum())
    coverage = covered / expected if expected else 1.0
    required = min_coverage if allow_partial else 1.0
    if coverage + 1e-12 < required:
        raise ValueError(f"{split} ASR cache coverage {covered}/{expected} ({coverage:.1%}) is below {required:.1%}. "
                         "Partial training requires --allow-partial-asr and an explicit --min-asr-coverage.")
    # A generation that ran into its token cap can be a truncated/hallucinated transcript.
    # Keep its processed ID for coverage, but use the existing empty-ASR fallback policy.
    capped = pd.Series(False, index=cache.index)
    if "hit_token_limit" in cache:
        flags = cache.hit_token_limit.fillna(False).astype(str).str.strip().str.lower()
        if not flags.isin(["true", "false", "1", "0", "1.0", "0.0", ""]).all():
            raise ValueError(f"Invalid hit_token_limit flag in ASR cache: {path}")
        capped = flags.isin(["true", "1", "1.0"])
    nonempty_before = cache.whisper.fillna("").astype(str).str.strip().ne("")
    cache = cache.copy()
    cache.loc[capped, "whisper"] = ""
    joined = frame.merge(cache[["id", "whisper"]], on="id", how="left", validate="one_to_one", sort=False)
    joined["whisper"] = joined.whisper.fillna("").astype(str).str.strip()
    provenance = {**fingerprint(path), "split": split, "cache_rows": len(cache), "expected_audio_usable": expected,
                  "covered_audio_usable": covered, "coverage": coverage, "allow_partial": allow_partial,
                  "required_coverage": required, "nonempty_audio_usable": int(joined.loc[eligible, "whisper"].ne("").sum()),
                  "excluded_hit_token_limit": int(capped.sum()),
                  "excluded_nonempty_hit_token_limit": int((capped & nonempty_before).sum()),
                  "excluded_hit_token_limit_audio_usable": int((capped & cache.id.isin(frame.loc[eligible, "id"])).sum()),
                  "quality_filter": "discard hit_token_limit transcripts; keep processed IDs for coverage; raw cache unchanged"}
    # Existing caches predate sidecars. Hash available generation metadata without inventing provenance.
    sidecar = path.with_suffix(".meta.json")
    if not sidecar.exists():
        sidecar = Path(str(path) + ".meta.json")
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        if metadata.get("split", split) != split:
            raise ValueError(f"ASR cache metadata split disagrees with requested {split}")
        if metadata.get("cache_sha256") is not None and metadata["cache_sha256"] != provenance["sha256"]:
            raise ValueError(f"ASR cache checksum disagrees with generation metadata: {path}")
        if metadata.get("complete") is False and not allow_partial:
            raise ValueError(f"ASR cache metadata is incomplete; finalize generation or explicitly allow partial ASR: {path}")
        provenance["generation_metadata"] = metadata
        provenance["generation_metadata_file"] = fingerprint(sidecar)
    else:
        provenance["generation_metadata"] = None
    return joined, provenance


# Purpose: Prepare gold/Whisper text columns for the chosen training source.
# Input: frame: manifest with text columns; source: gold, whisper, or mixed.
# Output: Prepared DataFrame filtered to nonempty texts for the selected source.
def prepare_training(frame, source):
    frame = frame.copy()
    if "whisper" not in frame:
        frame["whisper"] = ""
    gold, asr = frame.gold.ne(""), frame.whisper.ne("")
    keep = gold if source == "gold" else (asr if source == "whisper" else gold | asr)
    prepared = frame.loc[keep].reset_index(drop=True)
    if prepared.empty:
        raise ValueError("Training source has no nonempty texts")
    return prepared


# Purpose: Choose one text representation per utterance for a training epoch.
# Input: frame: prepared rows; source: gold/whisper/mixed; seed: experiment seed; epoch: epoch index.
# Output: Tuple (text list, integer label list, gold/whisper source counts).
def epoch_examples(frame, source, seed, epoch):
    """Exactly one row per utterance: mixed augmentation preserves the label distribution."""
    gold, asr = frame.gold.tolist(), frame.whisper.tolist()
    rng = np.random.default_rng(seed + epoch)
    if source == "mixed":
        use_asr = [bool(a) and (not g or rng.random() < 0.5) for g, a in zip(gold, asr)]
    else:
        use_asr = [source == "whisper"] * len(frame)
    texts = [a if use else g for g, a, use in zip(gold, asr, use_asr)]
    return texts, frame.label.astype(int).tolist(), {"gold": len(frame) - sum(use_asr), "whisper": sum(use_asr)}


# Purpose: Group aligned texts and labels, optionally shuffling row order.
# Input: texts: strings; labels: class indices; bs: batch size; shuffle: random-order flag.
# Output: Generator yielding (text_list, label_tensor) batches.
def batches(texts, labels, bs, shuffle):
    idx = list(range(len(texts)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield [texts[k] for k in j], torch.tensor([labels[k] for k in j])


# Purpose: Run batched text inference without training updates.
# Input: model: classifier; tok: tokenizer; texts: strings; device: Torch device; bs: batch size.
# Output: NumPy class probabilities shaped (N, 7), including an empty (0, 7) result for no texts.
@torch.no_grad()
def predict_probs(model, tok, texts, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.empty((0, len(EMOTIONS)), dtype=np.float32)


# Purpose: Convert batched classifier probabilities into label indices.
# Input: model: classifier; tok: tokenizer; texts: strings; device: Torch device.
# Output: Predicted label list, with empty texts assigned neutral.
def predict_labels(model, tok, texts, device):
    probs = predict_probs(model, tok, texts, device)
    empty = np.array([not t.strip() for t in texts], dtype=bool)
    labels = probs.argmax(1)
    labels[empty] = EMOTIONS.index("neutral")
    return labels


# Purpose: Evaluate only the explicitly requested splits and text sources.
# Input: model/tok/device: inference components; results_dir: output path; splits: explicit split tuple;
#          meld_dir: dataset root; dev_asr_cache/test_asr_cache: optional paths; allow_partial/min_coverage:
#          coverage policy.
# Output: None; saves source-specific predictions, metrics, and input provenance.
def evaluate_sources(model, tok, device, results_dir, splits=("dev",), meld_dir=MELD_DIR,
                     dev_asr_cache=None, test_asr_cache=None, allow_partial=False, min_coverage=1.0):
    """Only the caller's explicit splits are read. Test is gated by parse_args to eval-only."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in splits:
        frame, _ = read_manifest(meld_dir, split)
        path = (dev_asr_cache if split == "dev" else test_asr_cache) or Path(meld_dir) / f"whisper_{split}.csv"
        sources = {"gold": frame.gold}
        asr_provenance = None
        if Path(path).exists():
            frame, asr_provenance = attach_asr(frame, path, split, allow_partial, min_coverage)
            sources["whisper"] = frame.whisper
        eligible = frame.status.eq("ok")
        usable = frame.loc[eligible].reset_index(drop=True)
        for src, series in sources.items():
            texts = series.loc[eligible].tolist()
            probs = predict_probs(model, tok, texts, device)
            empty = np.array([not t.strip() for t in texts], dtype=bool)
            probs[empty] = np.nan
            pred = np.where(empty, -1, np.nan_to_num(probs, nan=0).argmax(1))
            pred_eval = np.where(pred < 0, EMOTIONS.index("neutral"), pred)
            m = metrics(usable.label.astype(int).to_numpy(), pred_eval)
            m.update({"split": split, "text_source": src, "n": len(usable), "empty": int(empty.sum()),
                      "asr_provenance": asr_provenance if src == "whisper" else None})
            stem = f"text_{split}_{src}"
            (results_dir / f"{stem}_metrics.json").write_text(json.dumps(m, indent=2, allow_nan=False))
            out = usable[["id", "emotion"]].copy()
            out["text"], out["pred"] = texts, [EMOTIONS[p] if p >= 0 else "" for p in pred]
            for i, emotion in enumerate(EMOTIONS):
                out[f"p_{emotion}"] = np.round(probs[:, i], 4)
            out.to_csv(results_dir / f"{stem}_predictions.csv", index=False)
            rows.append((split, src, m))
    lines = ["Text-only emotion candidate", "", "| split | source | n | weighted F1 | macro F1 | accuracy | empty |",
             "|---|---|---|---|---|---|---|"]
    for split, src, m in rows:
        lines.append(f"| {split} | {src} | {m['n']} | {m['weighted_f1']:.3f} | {m['macro_f1']:.3f} | {m['accuracy']:.3f} | {m['empty']} |")
    (results_dir / "text_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# Purpose: Protect existing training outputs from accidental overwrite.
# Input: output_dir: model destination; results_dir: results destination.
# Output: None; raises if either destination already exists.
def validate_training_outputs(output_dir, results_dir):
    for name, path in [("model", output_dir), ("results", results_dir)]:
        if Path(path).exists():
            raise FileExistsError(f"Refusing to overwrite existing {name} directory: {path}. Choose a new candidate path.")


# Purpose: Require the original ordered MELD labels for a warm-start classifier.
# Input: model_config: saved classifier configuration.
# Output: None; raises on incompatible label metadata.
def validate_warm_start(model_config):
    actual = [model_config.id2label.get(i, model_config.id2label.get(str(i))) for i in range(len(EMOTIONS))]
    if model_config.num_labels != len(EMOTIONS) or actual != EMOTIONS:
        raise ValueError("Warm-start checkpoint must have the same ordered MELD emotion labels")


# Purpose: Require a finite weighted-F1 improvement above an optional macro floor.
# Input: weighted_f1/macro_f1: candidate scores; best_weighted_f1: current best; macro_floor: optional
#          epoch-zero limit.
# Output: Boolean selection decision; ties and invalid scores are rejected.
def should_select_epoch(weighted_f1, macro_f1, best_weighted_f1, macro_floor=None):
    """Improve the primary score without crossing an optional fixed epoch-0 macro floor."""
    if not np.isfinite(weighted_f1) or not np.isfinite(macro_f1):
        return False
    return bool(weighted_f1 > best_weighted_f1 + 1e-12 and
                (macro_floor is None or macro_f1 + 1e-12 >= macro_floor))


# Purpose: Save a usable baseline fallback before optimization starts.
# Input: model/tok: warm-start components; texts/labels: selection inputs; device: Torch device; output_dir:
#          checkpoint destination; run: provenance dict.
# Output: Epoch-zero score record; also saves the initial model/tokenizer and input provenance.
def initialize_epoch_zero(model, tok, texts, labels, device, output_dir, run):
    """Save a usable warm-start fallback before optimization, with its exact selection inputs."""
    t0 = time.perf_counter()
    predicted = predict_labels(model, tok, texts, device)
    baseline = metrics(np.asarray(labels), predicted)
    baseline["macro_f1"] = float(f1_score(labels, predicted, labels=range(len(EMOTIONS)),
                                         average="macro", zero_division=0))
    run["selection_macro_floor"] = baseline["macro_f1"]
    run["baseline_selection"] = {
        "epoch": 0, "initialization": run["initialization"], "source": run["selection_source"],
        "metrics": baseline, "selection_n": len(labels),
        "manifest": run["provenance"]["dev_manifest"],
        "asr_cache": run["provenance"].get("selection_asr"),
    }
    row = {"epoch": 0, "train_loss": None, "dev_weighted_f1": baseline["weighted_f1"],
           "dev_macro_f1": baseline["macro_f1"], "epoch_s": time.perf_counter() - t0,
           "source_counts": {"gold": 0, "whisper": 0}, "evaluation_only": True,
           "passes_macro_guard": True, "selected": True}
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    (Path(output_dir) / "meld_meta.json").write_text(json.dumps({**run, **row}, indent=2, allow_nan=False))
    (Path(output_dir) / "history.json").write_text(json.dumps([row], indent=2, allow_nan=False))
    return row


# Purpose: Parse text training/evaluation options and enforce split/output safeguards.
# Input: argv: optional argument list; None uses process arguments.
# Output: Validated argparse Namespace; invalid combinations terminate with a parser error.
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--class-weighting", choices=["none", "sqrt", "balanced"], default="sqrt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--train-source", choices=["gold", "whisper", "mixed"], default="gold")
    ap.add_argument("--selection-source", choices=["gold", "whisper"], default="gold")
    ap.add_argument("--train-asr-cache", type=Path, default=MELD_DIR / "whisper_train.csv")
    ap.add_argument("--dev-asr-cache", "--selection-asr-cache", dest="dev_asr_cache", type=Path, default=MELD_DIR / "whisper_dev.csv")
    ap.add_argument("--test-asr-cache", type=Path, default=MELD_DIR / "whisper_test.csv")
    ap.add_argument("--allow-partial-asr", action="store_true")
    ap.add_argument("--min-asr-coverage", type=float, default=1.0)
    ap.add_argument("--init-model-dir", type=Path, help="explicit local MELD checkpoint warm-start; default uses pretrained base")
    ap.add_argument("--preserve-baseline-macro", action="store_true",
                    help="save explicit warm-start epoch 0; select only higher dev weighted F1 with macro F1 >= epoch 0")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true", help="explicitly train classifier head only; default fine-tunes all weights")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--results-dir", type=Path, help="default: an isolated results/ subdirectory of --output-dir")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-splits", nargs="+", choices=["dev", "test"], default=["dev"])
    args = ap.parse_args(argv)
    if args.preserve_baseline_macro and (args.init_model_dir is None or args.eval_only):
        ap.error("--preserve-baseline-macro requires --init-model-dir and training mode")
    if "test" in args.eval_splits and not args.eval_only:
        ap.error("test evaluation requires --eval-only; training and epoch selection use train/dev only")
    if min(args.epochs, args.batch_size, args.threads) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        ap.error("epochs, batch size, threads and learning rate must be positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        ap.error("weight decay must be finite and nonnegative")
    if not 0 <= args.min_asr_coverage <= 1 or (args.min_asr_coverage < 1 and not args.allow_partial_asr):
        ap.error("lower ASR coverage requires --allow-partial-asr and a threshold in [0, 1]")
    args.results_dir = args.results_dir or args.output_dir / "results"
    if args.results_dir.resolve() == args.output_dir.resolve() or args.results_dir.resolve() in args.output_dir.resolve().parents:
        ap.error("results directory must be separate from or nested inside the model output directory")
    return args


# Purpose: Train and select a text candidate, or run explicit evaluation-only mode.
# Input: argv: optional command-line argument list.
# Output: None; writes model/history/evaluation artifacts according to the validated mode.
def main(argv=None):
    args = parse_args(argv)
    if not args.eval_only:
        validate_training_outputs(args.output_dir, args.results_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.eval_only:
        tok = AutoTokenizer.from_pretrained(args.output_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, local_files_only=True).to(device)
        validate_warm_start(model.config)
        evaluate_sources(model, tok, device, args.results_dir, args.eval_splits, dev_asr_cache=args.dev_asr_cache,
                         test_asr_cache=args.test_asr_cache, allow_partial=args.allow_partial_asr, min_coverage=args.min_asr_coverage)
        return

    tr, train_manifest = read_manifest(MELD_DIR, "train")
    dv, dev_manifest = read_manifest(MELD_DIR, "dev")
    provenance = {"train_manifest": train_manifest, "dev_manifest": dev_manifest}
    if args.train_source != "gold":
        tr, provenance["train_asr"] = attach_asr(tr, args.train_asr_cache, "train", args.allow_partial_asr, args.min_asr_coverage)
    train = prepare_training(tr, args.train_source)
    if args.selection_source == "whisper":
        dv, provenance["selection_asr"] = attach_asr(dv, args.dev_asr_cache, "dev", args.allow_partial_asr, args.min_asr_coverage)
        dv = dv.loc[dv.status == "ok"].reset_index(drop=True)
    selection_texts = dv["gold" if args.selection_source == "gold" else "whisper"].tolist()
    selection_y = dv.label.astype(int).tolist()
    if not selection_texts:
        raise ValueError("No development examples for model selection")
    initial = args.init_model_dir or BASE
    local_only = args.local_files_only or args.init_model_dir is not None
    if args.init_model_dir is not None:
        if not (args.init_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Warm-start checkpoint missing: {args.init_model_dir}")
        provenance["warm_start"] = {p.name: fingerprint(p) for p in args.init_model_dir.iterdir()
                                    if p.name == "config.json" or p.suffix in {".safetensors", ".bin"}}
    tok = AutoTokenizer.from_pretrained(initial, local_files_only=local_only)
    if args.init_model_dir is not None:
        model = AutoModelForSequenceClassification.from_pretrained(initial, local_files_only=True).to(device)
        validate_warm_start(model.config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            initial, num_labels=len(EMOTIONS), id2label=dict(enumerate(EMOTIONS)),
            label2id={e: i for i, e in enumerate(EMOTIONS)}, local_files_only=local_only).to(device)
    if args.freeze_encoder:
        for param in model.base_model.parameters():
            param.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train.label.tolist(), args.class_weighting).to(device))
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.results_dir.mkdir(parents=True, exist_ok=False)
    serial_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run = {"base": BASE, "initialization": str(initial), "warm_start": args.init_model_dir is not None,
           "classes": EMOTIONS, "parameters": n_params, "trainable_parameters": n_trainable,
           "device": str(device), "threads": torch.get_num_threads(),
           "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__},
           "args": serial_args, "provenance": provenance, "selection_source": args.selection_source,
           "selection_n": len(selection_y), "train_n": len(train), "test_read": False,
           "mixed_policy": "one representation per paired utterance per epoch, ASR probability 0.5, gold fallback for empty/capped ASR",
           "selection_policy": "higher dev weighted F1 with macro F1 >= epoch 0" if args.preserve_baseline_macro else "higher dev weighted F1",
           "selection_macro_floor": None, "baseline_selection": None}
    best, history, macro_floor = -1.0, [], None
    if args.preserve_baseline_macro:
        epoch_zero = initialize_epoch_zero(model, tok, selection_texts, selection_y, device, args.output_dir, run)
        best, macro_floor = epoch_zero["dev_weighted_f1"], epoch_zero["dev_macro_f1"]
        history.append(epoch_zero)
        print(f"epoch 0 warm-start dev-{args.selection_source} wF1={best:.4f} macro={macro_floor:.4f}; saved fallback")
    (args.output_dir / "run_config.json").write_text(json.dumps(run, indent=2, allow_nan=False))
    print(f"initial={initial} params={n_params:,} trainable={n_trainable:,} device={device} threads={args.threads} "
          f"train={len(train)} source={args.train_source} selection={args.selection_source}:{len(selection_y)}")
    for epoch in range(args.epochs):
        texts, labels, source_counts = epoch_examples(train, args.train_source, args.seed, epoch)
        model.train()
        if args.freeze_encoder:
            model.base_model.eval()
        t0, total, n = time.time(), 0.0, 0
        for batch_texts, y in tqdm(batches(texts, labels, args.batch_size, True),
                                  total=(len(texts) + args.batch_size - 1) // args.batch_size, desc=f"epoch {epoch + 1}", leave=False):
            enc = tok(batch_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            loss = criterion(model(**enc).logits, y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        pred = predict_labels(model, tok, selection_texts, device)
        wf1 = f1_score(selection_y, pred, labels=range(len(EMOTIONS)), average="weighted", zero_division=0)
        mf1 = f1_score(selection_y, pred, labels=range(len(EMOTIONS)), average="macro", zero_division=0)
        improved = should_select_epoch(wf1, mf1, best, macro_floor)
        row = {"epoch": epoch + 1, "train_loss": total / n, "dev_weighted_f1": wf1,
               "dev_macro_f1": mf1, "epoch_s": time.time() - t0, "source_counts": source_counts,
               "evaluation_only": False, "passes_macro_guard": bool(macro_floor is None or mf1 + 1e-12 >= macro_floor),
               "selected": improved}
        history.append(row)
        print(f"epoch {epoch + 1} train={total / n:.4f} dev-{args.selection_source} wF1={wf1:.4f} "
              f"macro={mf1:.4f} {row['epoch_s']:.1f}s{' *best*' if improved else ''}"
              f"{' (macro guard failed)' if not row['passes_macro_guard'] else ''}")
        if improved:
            best = wf1
            model.save_pretrained(args.output_dir)
            tok.save_pretrained(args.output_dir)
            (args.output_dir / "meld_meta.json").write_text(json.dumps({**run, **row}, indent=2, allow_nan=False))
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False))
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, local_files_only=True).to(device)
    evaluate_sources(model, tok, device, args.results_dir, args.eval_splits, dev_asr_cache=args.dev_asr_cache,
                     test_asr_cache=args.test_asr_cache, allow_partial=args.allow_partial_asr, min_coverage=args.min_asr_coverage)


if __name__ == "__main__":
    main()
