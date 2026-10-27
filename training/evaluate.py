"""Deployment-matched MELD evaluation: audio dev, recipe comparison, and locked test.

python -m training.evaluate audio --output-dir results/audio_dev
python -m training.evaluate compare --help
python -m training.evaluate test --receipt results/submission_model_selection.json --output-dir results/test_check

Development comparison never fits weights or opens test data. Test input access requires
an explicit dev-selection lock and unchanged checkpoint/fusion hashes first.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from app import fusion
from app.config import CHECKPOINT_DIR, EMOTIONS, ROOT, SAMPLE_RATE, Settings
from app.emotion import EmotionClassifier
from app.features import load_audio
from app.text_emotion import MAX_LEN, TextEmotionClassifier
from training.metrics_utils import metrics, bootstrap_comparison, read_probabilities, source_record

TEXT_FILES = ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'meld_meta.json')
MODES = {'audio': ('audio',), 'text': ('text',), 'audio_text': ('audio', 'text'),
         'audio_text_vision': ('audio', 'text', 'vision')}
PROBABILITIES = [f'p_{emotion}' for emotion in EMOTIONS]

# Purpose: Compute a source artifact's checksum.
# Input: path: readable local file.
# Output: Hexadecimal SHA-256 string.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# Purpose: Resolve a receipt path against the repository when it is relative.
# Input: path: relative or absolute path; repo: repository root.
# Output: Resolved absolute Path.
def local_path(path, repo):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


# Purpose: Verify that a recorded artifact exists and matches its locked checksum.
# Input: record: path/hash metadata; repo: repository root.
# Output: Verified artifact Path; raises when the file or checksum is invalid.
def verify_file(record, repo):
    if not isinstance(record, dict) or not isinstance(record.get('sha256'), str) or len(record['sha256']) != 64:
        raise ValueError('Each frozen file requires a path and SHA256')
    path = local_path(record['path'], repo)
    if not path.is_file() or sha256(path) != record['sha256']:
        raise ValueError(f'Frozen file missing or changed: {path}')
    return path


# Purpose: Verify the model-selection lock before any test data is read.
# Input: receipt_path: frozen recipe JSON; repo: repository root.
# Output: Tuple (receipt, model_profiles, fusion_weights); invalid locks or changed artifacts raise errors.
def validate_receipt(receipt_path, repo):
    """Validate the lock and all model artifacts without inspecting any test input."""
    receipt = json.loads(Path(receipt_path).read_text())
    if (receipt.get('schema') != 1 or receipt.get('locked_for_test') is not True or
            receipt.get('selection_split') != 'dev' or receipt.get('test_used_for_selection') is not False):
        raise ValueError('Require an explicit frozen dev-selected recipe with test_used_for_selection=false')
    locked = datetime.fromisoformat(str(receipt.get('locked_utc', '')).replace('Z', '+00:00'))
    if locked.utcoffset() is None:
        raise ValueError('locked_utc must include a timezone')
    if not isinstance(receipt.get('recipe'), str) or not receipt['recipe']:
        raise ValueError('A frozen recipe name is required')
    protocol = receipt.get('protocol', {})
    if (protocol.get('device') != 'cpu' or protocol.get('text_max_tokens') != MAX_LEN or
            not np.isfinite(protocol.get('max_audio_seconds', float('nan'))) or
            not np.isclose(protocol['max_audio_seconds'], Settings().final_max_s, rtol=0, atol=1e-12)):
        raise ValueError('Freeze CPU singleton protocol with the current runtime audio crop and text_max_tokens')
    profiles = {}
    for name in ('baseline', 'selected'):
        profile = receipt[name]
        audio = verify_file(profile['audio'], repo)
        text = profile['text']
        directory = local_path(text['path'], repo)
        if not isinstance(text.get('files'), dict) or not set(TEXT_FILES).issubset(text['files']):
            raise ValueError(f'{name} text checkpoint requires hashes for {TEXT_FILES}')
        for filename, value in text['files'].items():
            if Path(filename).name != filename:
                raise ValueError('Text artifact hashes must use plain filenames')
            digest = value['sha256'] if isinstance(value, dict) else value
            verify_file({'path': str(directory/filename), 'sha256': digest}, repo)
        profiles[name] = {'audio': audio, 'text': directory}
    weights_path = verify_file(receipt['fusion_weights'], repo)
    weights = json.loads(weights_path.read_text())
    if not isinstance(weights.get('live'), dict):
        raise ValueError('Frozen fusion specification must contain live weights')
    return receipt, profiles, weights


# Purpose: Load the fixed test manifest, MLX transcripts, and cached face readings.
# Input: repo: repository root; call only after validating the selection receipt.
# Output: Tuple (joined_test_frame, face_readings, input_path_hash_records).
def load_test_inputs(repo):
    """Called only after validate_receipt. Fixed filenames prevent CPU-ASR cache substitution."""
    paths = {name: repo/'data/meld'/file for name, file in {
        'manifest': 'test.csv', 'asr': 'whisper_test.csv', 'vision': 'vision_test.csv'}.items()}
    frame = pd.read_csv(paths['manifest'])
    frame = frame.loc[frame.status.eq('ok')].reset_index(drop=True)
    if frame.empty or frame.id.isna().any() or not frame.id.is_unique:
        raise ValueError('Test manifest must have unique usable IDs')
    if not frame.label.isin(range(len(EMOTIONS))).all() or not np.array_equal(frame.emotion.to_numpy(), np.asarray(EMOTIONS)[frame.label.to_numpy(dtype=int)]):
        raise ValueError('Test manifest emotion/label order disagrees with runtime labels')
    asr, vision = pd.read_csv(paths['asr']), pd.read_csv(paths['vision'])
    for name, cached in [('asr', asr), ('vision', vision)]:
        if cached.id.isna().any() or not cached.id.is_unique or set(cached.id) != set(frame.id):
            raise ValueError(f'Frozen {name} cache must cover usable test IDs exactly once')
    if 'backend' in asr and asr.backend.astype(str).str.contains('cpu', case=False).any():
        raise ValueError('Require the original frozen MLX whisper_test.csv, not CPU-ASR transcripts')
    aligned_asr = frame[['id']].merge(asr[['id','whisper']], on='id', validate='one_to_one')
    frame['transcript'] = aligned_asr.whisper.fillna('').astype(str)
    aligned_vision = frame[['id','emotion']].merge(vision[['id','emotion']+PROBABILITIES], on='id', validate='one_to_one', suffixes=('','_vision'))
    if not aligned_vision.emotion.equals(aligned_vision.emotion_vision):
        raise ValueError('Frozen vision cache labels disagree with test manifest')
    face_readings = []
    for values in aligned_vision[PROBABILITIES].to_numpy(dtype=float):
        if np.isnan(values).all():
            face_readings.append(None)
            continue
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any() or values.sum() <= 0:
            raise ValueError('Invalid/partly missing vision probabilities')
        index = int(values.argmax())
        face_readings.append({'emotion': EMOTIONS[index], 'confidence': float(values[index]),
                              'probs': dict(zip(EMOTIONS, values.tolist()))})
    return frame, face_readings, {name: {'path': str(path), 'sha256': sha256(path)} for name,path in paths.items()}


# Purpose: Apply runtime fusion to a requested modality subset.
# Input: readings: modality result dicts; modes: included modality names; weights: saved fusion
#          specification.
# Output: Integer MELD label index; neutral if no fused reading is available.
def fused_label(readings, modes, weights):
    available = {mode for mode in modes if readings.get(mode) is not None}
    result = fusion.fuse({mode: readings.get(mode) for mode in modes}, fusion.runtime_weights(available, weights))
    # Text-only reporting follows historical scoring; multimodal fusion always skips missing words.
    return EMOTIONS.index(result['emotion']) if result else EMOTIONS.index('neutral')


# Purpose: Compare baseline/selected predictions with dialogue bootstrap intervals.
# Input: truth: label indices; dialogue_ids: grouping IDs; predictions: recipe/mode predictions; bootstrap:
#          resamples; seed: RNG seed.
# Output: Metrics/comparison dict with point scores, changed predictions, and bootstrap intervals.
def summarize(truth, dialogue_ids, predictions, bootstrap=2000, seed=0):
    groups, group_index = np.unique(dialogue_ids, return_inverse=True)
    counts = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=bootstrap)
    result = {}
    for mode in MODES:
        before, after = np.asarray(predictions['baseline'][mode]), np.asarray(predictions['selected'][mode])
        base, selected = metrics(truth, before), metrics(truth, after)
        result[mode] = {'baseline': base, 'selected': selected,
                        'delta': {key: selected[key]-base[key] for key in ('weighted_f1','macro_f1','accuracy')},
                        'paired_dialogue_bootstrap_delta_95ci': bootstrap_comparison(truth,before,after,group_index,counts),
                        'changed_predictions': int((before!=after).sum()),
                        'fixes': int(((before!=truth)&(after==truth)).sum()),
                        'breaks': int(((before==truth)&(after!=truth)).sum())}
    return result



# Purpose: Evaluate the locked recipe and baseline on held-out inputs without selection.
# Input: args: parsed receipt/output/device/thread/bootstrap options.
# Output: None; writes test predictions, metrics, comparison report, and receipt evidence.
def evaluate_test(args):
    repo = args.repo.resolve()
    receipt_hash = sha256(args.receipt)
    receipt, profiles, weights = validate_receipt(args.receipt, repo)
    # The first test-data access happens only after the explicit lock and model hashes pass.
    frame, faces, input_records = load_test_inputs(repo)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(args.threads)
    unique_audio, unique_text, models = {}, {}, {}
    for name in ('baseline','selected'):
        audio_key = receipt[name]['audio']['sha256']
        text_key = tuple(sorted((k, v['sha256'] if isinstance(v,dict) else v) for k,v in receipt[name]['text']['files'].items()))
        if audio_key not in unique_audio:
            model = EmotionClassifier(profiles[name]['audio'],device='cpu'); model.load()
            if model.classes != EMOTIONS:
                raise ValueError('Audio checkpoint label order differs from MELD')
            unique_audio[audio_key] = model
        if text_key not in unique_text:
            model = TextEmotionClassifier(profiles[name]['text'],device='cpu'); model.load()
            if model.classes != EMOTIONS or [model.model.config.id2label[i] for i in range(7)] != EMOTIONS:
                raise ValueError('Text checkpoint label order differs from MELD')
            unique_text[text_key] = model
        models[name] = (audio_key,text_key)
    predictions = {name: {mode: [] for mode in MODES} for name in models}
    rows=[]; started=time.perf_counter(); crop=int(receipt['protocol']['max_audio_seconds']*SAMPLE_RATE)
    for index, clip in enumerate(frame.itertuples()):
        waveform = load_audio(repo/'data/meld'/clip.wav_path)[-crop:]
        voice = {key:model.predict(waveform) for key,model in unique_audio.items()}
        words = {key:model.predict(clip.transcript) for key,model in unique_text.items()}
        row={'id':clip.id,'Dialogue_ID':clip.Dialogue_ID,'emotion':clip.emotion,'label':int(clip.label),'transcript':clip.transcript}
        for name,(audio_key,text_key) in models.items():
            readings={'audio':voice[audio_key],'text':words[text_key],'vision':faces[index]}
            for mode,modes in MODES.items():
                label=fused_label(readings,modes,weights)
                predictions[name][mode].append(label); row[f'{name}_{mode}']=EMOTIONS[label]
            for modality in ('audio','text'):
                reading=readings[modality]
                for emotion in EMOTIONS:
                    row[f'{name}_{modality}_p_{emotion}']=reading['probs'][emotion] if reading else np.nan
        rows.append(row)
        if (index+1)%100==0 or index+1==len(frame):
            print(json.dumps({'evaluated':index+1,'n':len(frame),'elapsed_s':round(time.perf_counter()-started,1)}),flush=True)
    results=summarize(frame.label.to_numpy(dtype=int),frame.Dialogue_ID.to_numpy(),predictions,args.bootstrap,args.seed)
    if sha256(args.receipt) != receipt_hash:
        raise ValueError('Selection receipt changed during evaluation')
    validate_receipt(args.receipt, repo)  # reject checkpoints/weights modified during this run
    report={'schema':1,'split':'test','n':len(frame),'locked_recipe':receipt['recipe'],
            'receipt':{'path':str(args.receipt.resolve()),'sha256':receipt_hash},
            'sources':input_records,'checkpoint_profiles':{name:receipt[name] for name in models},
            'fusion_weights':receipt['fusion_weights'],'comparisons':results,
            'method':{'test_used_for_fitting':False,'recipe_locked_before_test':True,'runtime_defaults_modified':False,
                      'device':'cpu','batch_size':1,'audio_max_seconds':receipt['protocol']['max_audio_seconds'],
                      'text_max_tokens':MAX_LEN,'asr':'existing frozen MLX whisper_test.csv; no ASR inference',
                      'vision':'existing frozen vision_test.csv; no live face extraction',
                      'fusion':'app.fusion.fuse with deployed per-availability subset lookup',
                      'vad_segmentation':False,'text_only_missing_policy':'neutral; missing words skipped in fused modes',
                      'bootstrap_unit':'paired dialogue clusters','bootstrap_replicates':args.bootstrap,'seed':args.seed,
                      'fitting_or_tuning':False,'recipe_recommendation':False},
            'limitations':'Labelled utterance boundaries and frozen vision/ASR caches; not a streaming VAD, generation-quality or end-to-end interaction benchmark.',
            'inference_wall_s':time.perf_counter()-started,'torch_version':str(torch.__version__)}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'frozen_selection_receipt.json').write_bytes(args.receipt.read_bytes())
    (args.output_dir/'test_frozen_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(rows).to_csv(args.output_dir/'test_frozen_predictions.csv',index=False)
    lines=['# Frozen-recipe held-out comparison','',report['limitations'],'',
           '| mode | baseline wF1 | selected wF1 | delta | baseline macro | selected macro |',
           '|---|---:|---:|---:|---:|---:|']
    for mode,value in results.items():
        lines.append(f"| {mode} | {value['baseline']['weighted_f1']:.4f} | {value['selected']['weighted_f1']:.4f} | {value['delta']['weighted_f1']:+.4f} | {value['baseline']['macro_f1']:.4f} | {value['selected']['macro_f1']:.4f} |")
    lines += ['', 'The recipe was locked using development data before this test comparison. No fitting, tuning or recommendation is performed here.']
    (args.output_dir/'test_frozen_comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


# Purpose: Score a checkpoint on dev using the live singleton crop/feature path.
# Input: args: parsed audio checkpoint, crop, threads, and output options.
# Output: None; writes development audio probabilities/metrics; invalid settings or existing output raise
#          errors.
def evaluate_audio(args):
    """Development only; crop the waveform before features, as the live session does."""
    torch.set_num_threads(args.threads)
    manifest_path = args.repo / 'data/meld/dev.csv'
    manifest = pd.read_csv(manifest_path).query("status == 'ok'").reset_index(drop=True)
    if manifest.empty or not manifest.id.is_unique:
        raise ValueError('Development manifest requires unique usable clips')
    classifier = EmotionClassifier(args.checkpoint, device='cpu')
    started = time.perf_counter()
    classifier.load()
    rows = []
    limit = int(args.max_seconds * SAMPLE_RATE)
    for clip in tqdm(manifest.itertuples(), total=len(manifest), desc='Audio development'):
        wav = load_audio(args.repo / 'data/meld' / clip.wav_path)
        reading = classifier.predict(wav[-limit:])
        rows.append({'id': clip.id, 'Dialogue_ID': clip.Dialogue_ID, 'emotion': clip.emotion,
                     'label': int(clip.label), 'pred': reading['emotion'], 'ms': reading['ms'],
                     'source_duration_s': wav.size / SAMPLE_RATE,
                     **{f'p_{e}': reading['probs'][e] for e in EMOTIONS}})
    frame = pd.DataFrame(rows)
    report = {**metrics(frame.label, frame.pred.map(EMOTIONS.index)),
              'split': 'dev', 'n': len(frame), 'checkpoint': source_record(args.checkpoint),
              'manifest': source_record(manifest_path), 'device': 'cpu', 'batch_size': 1,
              'max_audio_seconds': args.max_seconds, 'vad_segmentation': False,
              'wall_s': time.perf_counter() - started}
    args.output_dir.mkdir(parents=True)
    frame.to_csv(args.output_dir / 'dev_runtime_audio_predictions.csv', index=False)
    (args.output_dir / 'dev_runtime_audio_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


# Purpose: Require higher all-modality dev weighted F1 without lowering macro F1.
# Input: comparisons: baseline/selected metrics by modality mode.
# Output: String 'selected' when both safeguards pass; otherwise 'baseline', including ties.
def choose_candidate(comparisons):
    baseline = comparisons['audio_text_vision']['baseline']
    selected = comparisons['audio_text_vision']['selected']
    return ('selected' if selected['weighted_f1'] > baseline['weighted_f1'] + 1e-12
            and selected['macro_f1'] + 1e-12 >= baseline['macro_f1'] else 'baseline')


# Purpose: Ensure both recipes were evaluated using identical development transcripts.
# Input: manifest: dev rows; baseline_path/candidate_path: saved text prediction CSV paths.
# Output: None; raises if IDs or recognized text differ.
def verify_same_text(manifest, baseline_path, candidate_path):
    frames = [pd.read_csv(path) for path in (baseline_path, candidate_path)]
    if any('text' not in frame for frame in frames):
        raise ValueError('Both text prediction files must retain their input text')
    aligned = [manifest[['id']].merge(frame[['id','text']], on='id', how='left',
                                     validate='one_to_one').text.fillna('').astype(str) for frame in frames]
    if not aligned[0].equals(aligned[1]):
        raise ValueError('Baseline and candidate must use the same development transcripts')


# Purpose: Compare two fixed complete recipes on development data.
# Input: args: saved audio/text/vision probabilities, fusion weights, output, and bootstrap options.
# Output: None; writes dev comparison/predictions and guarded recommendation without promoting a model.
def evaluate_compare(args):
    """Compare two complete recipes on dev; no retuning or automatic promotion."""
    manifest_path = args.repo / 'data/meld/dev.csv'
    manifest = pd.read_csv(manifest_path).query("status == 'ok'").reset_index(drop=True)
    if manifest.empty or not manifest.id.is_unique:
        raise ValueError('Development manifest requires unique usable clips')
    verify_same_text(manifest, args.baseline_text, args.candidate_text)
    weights = json.loads(args.weights.read_text())
    face = read_probabilities(manifest, args.vision)
    predicted = {}
    for name, audio_path, text_path in [('baseline', args.baseline_audio, args.baseline_text),
                                       ('selected', args.candidate_audio, args.candidate_text)]:
        channels = {'audio': read_probabilities(manifest, audio_path, required=True),
                    'text': read_probabilities(manifest, text_path), 'vision': face}
        predicted[name] = {mode: [] for mode in MODES}
        for index in range(len(manifest)):
            readings = {m: None if missing[index] else {'probs': dict(zip(EMOTIONS, np.exp(logs[index])))}
                        for m, (logs, missing) in channels.items()}
            for mode, modes in MODES.items():
                predicted[name][mode].append(fused_label(readings, modes, weights))
    comparison = summarize(manifest.label.to_numpy(dtype=int), manifest.Dialogue_ID.to_numpy(),
                           predicted, args.bootstrap, args.seed)
    report = {'split': 'dev', 'n': len(manifest), 'comparisons': comparison,
              'recommended_recipe': choose_candidate(comparison),
              'selection_rule': 'Higher full-recipe weighted F1 with macro F1 >= baseline; ties keep baseline',
              'test_read': False, 'weights_fitted': False, 'runtime_modified': False,
              'caveat': 'Development comparison and intervals are conditional on model selection; not held-out gains.',
              'sources': {name: source_record(path) for name,path in {
                  'manifest': manifest_path, 'baseline_audio': args.baseline_audio,
                  'candidate_audio': args.candidate_audio, 'baseline_text': args.baseline_text,
                  'candidate_text': args.candidate_text, 'vision': args.vision, 'weights': args.weights}.items()}}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'dev_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    frame = manifest[['id','emotion','label']].copy()
    for name in predicted:
        for mode, values in predicted[name].items(): frame[f'{name}_{mode}'] = [EMOTIONS[i] for i in values]
    frame.to_csv(args.output_dir / 'dev_predictions.csv', index=False)
    print(json.dumps({k:report[k] for k in ['n','recommended_recipe','selection_rule']}, indent=2))


# Purpose: Dispatch the audio, compare, or locked test evaluation command.
# Input: argv: optional CLI argument list; None uses process arguments.
# Output: None; runs the chosen evaluator and writes its requested outputs.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['audio','compare','test']:
        command = commands.add_parser(name)
        command.add_argument('--repo', type=Path, default=ROOT)
        command.add_argument('--output-dir', type=Path, required=True)
        command.add_argument('--threads', type=int, default=4)
        if name != 'audio':
            command.add_argument('--bootstrap', type=int, default=2000)
            command.add_argument('--seed', type=int, default=0)
        if name == 'audio':
            command.add_argument('--checkpoint', type=Path, default=CHECKPOINT_DIR/'emotion_cnn.pt')
            command.add_argument('--max-seconds', type=float, default=Settings().final_max_s)
        elif name == 'compare':
            for channel in ['baseline-audio','candidate-audio','baseline-text','candidate-text']:
                command.add_argument('--'+channel, type=Path, required=True)
            command.add_argument('--vision', type=Path, default=ROOT/'data/meld/vision_dev.csv')
            command.add_argument('--weights', type=Path, default=ROOT/'results/fusion_weights.json')
        else:
            command.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads < 1 or getattr(args,'bootstrap',1) < 1:
        parser.error('Threads and bootstrap count must be positive')
    if args.command == 'audio' and (not np.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error('Audio crop duration must be positive and finite')
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation evidence: {args.output_dir}')
    args.repo = args.repo.resolve()
    {'audio':evaluate_audio,'compare':evaluate_compare,'test':evaluate_test}[args.command](args)


if __name__ == '__main__':
    main()
