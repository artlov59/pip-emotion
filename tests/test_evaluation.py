"""Live fusion parity, development safeguards and paired comparison statistics."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.config import EMOTIONS
from app.pipeline import Engine
from training import evaluate as evaluation
from training import metrics_utils as scoring


class EvaluationTests(unittest.TestCase):
    # Purpose: Verify runtime weight lookup matches engine for missing subsets.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_runtime_weight_lookup_matches_engine_for_missing_subsets(self):
        engine = Engine.__new__(Engine)
        engine.weights = {'audio': 1., 'text': .9, 'vision': .3}
        engine.weights_all = {'audio+vision': {'audio': 1., 'vision': .2}}
        spec = {'live': engine.weights, 'all': engine.weights_all}
        for modes in evaluation.MODES.values():
            self.assertEqual(engine.weights_for(set(modes)), evaluation.fusion.runtime_weights(modes,spec))

    # Purpose: Verify confusion statistics match shared classification scores.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_confusion_statistics_match_shared_classification_scores(self):
        truth = np.tile(np.arange(7),3); predicted = truth.copy()
        predicted[[1,4,9,17]] = [0,5,2,0]
        groups = np.repeat(np.arange(3),7)
        cm = scoring.grouped_confusion(truth,predicted,groups,3).sum(axis=0)
        actual = scoring.confusion_scores(cm); expected = scoring.metrics(truth,predicted)
        for name in ['weighted_f1','macro_f1','accuracy']: self.assertAlmostEqual(actual[name],expected[name])
        draws = np.random.default_rng(0).multinomial(3,[1/3]*3,size=20)
        intervals = scoring.bootstrap_comparison(truth,predicted,predicted,groups,draws)
        for name in ['weighted_f1','macro_f1','accuracy']: self.assertEqual(intervals[name],[0,0])

    # Purpose: Verify development selection preserves macro floor and ties.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_development_selection_preserves_macro_floor_and_ties(self):
        # Purpose: Construct a minimal development-score fixture.
        # Input: w: weighted F1; m: macro F1.
        # Output: Dict containing the two scores.
        def result(w,m): return {'weighted_f1':w,'macro_f1':m}
        for candidate,expected in [(result(.6,.39),'baseline'),(result(.5,.4),'baseline'),(result(.6,.4),'selected')]:
            value = {'audio_text_vision':{'baseline':result(.5,.4),'selected':candidate}}
            self.assertEqual(evaluation.choose_candidate(value),expected)

    # Purpose: Verify development comparison reads cached predictions without touching test.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_development_comparison_reads_cached_predictions_without_touching_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=root/'data/meld'; data.mkdir(parents=True)
            manifest=pd.DataFrame({'id':[f'clip{i}' for i in range(7)],'emotion':EMOTIONS,
                                   'label':range(7),'status':['ok']*7,'Dialogue_ID':[0]*7})
            manifest.to_csv(data/'dev.csv',index=False)
            cached=manifest[['id','emotion']].copy(); cached['text']='spoken'
            for e in EMOTIONS: cached['p_'+e]=.02
            cached['p_neutral']=.88
            for name in ['audio','text','vision']: cached.to_csv(root/f'{name}_dev.csv',index=False)
            weights=root/'weights.json'; weights.write_text('{"live":{"audio":1,"text":1,"vision":0.3},"all":{}}')
            out=root/'out'
            args=['compare','--repo',str(root),'--output-dir',str(out),'--bootstrap','10',
                  '--weights',str(weights),'--vision',str(root/'vision_dev.csv')]
            for name in ['baseline-audio','candidate-audio','baseline-text','candidate-text']:
                args += ['--'+name,str(root/('audio_dev.csv' if name.endswith('audio') else 'text_dev.csv'))]
            with contextlib.redirect_stdout(io.StringIO()): evaluation.main(args)
            report=json.loads((out/'dev_comparison.json').read_text())
            self.assertEqual(report['recommended_recipe'],'baseline')
            self.assertFalse(report['test_read']); self.assertFalse(report['weights_fitted'])
            for row in report['comparisons'].values(): self.assertEqual(row['delta']['weighted_f1'],0)
            cached.loc[0,'text']='different'; cached.to_csv(root/'changed_dev.csv',index=False)
            with self.assertRaisesRegex(ValueError,'same development transcripts'):
                evaluation.verify_same_text(manifest,root/'text_dev.csv',root/'changed_dev.csv')

    # Purpose: Verify audio development mode rejects test arguments and invalid crop.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_audio_development_mode_rejects_test_arguments_and_invalid_crop(self):
        for args in [['audio','--split','test','--output-dir','unused'],
                     ['audio','--max-seconds','nan','--output-dir','unused']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): evaluation.main(args)


if __name__ == '__main__': unittest.main()
