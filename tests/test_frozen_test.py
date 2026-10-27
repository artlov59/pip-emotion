"""Lock/hash/live-fusion checks using synthetic files and fake inference only."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.config import EMOTIONS
from training import evaluate as evaluation


# Purpose: Construct a deterministic emotion result for locked-evaluation tests.
# Input: emotion: dominant MELD label.
# Output: Reading dict with fixed confidence, class probabilities, and zero timing.
def reading(emotion):
    probs={name:0.02 for name in EMOTIONS}; probs[emotion]=0.88
    return {'emotion':emotion,'confidence':0.88,'probs':probs,'ms':0.0}


class FrozenComparisonTests(unittest.TestCase):
    # Purpose: Create synthetic model artifacts, hashes, and a locked selection receipt.
    # Input: None.
    # Output: None; initializes isolated fixtures and registers cleanup where needed.
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name); self.output=self.root/'output'
        audio=self.root/'audio.pt'; audio.write_bytes(b'fake checkpoint')
        text=self.root/'text'; text.mkdir()
        for name in evaluation.TEXT_FILES: (text/name).write_text('{}')
        weights=self.root/'weights.json'; weights.write_text('{"live":{"audio":1,"text":1,"vision":0.3},"all":{}}')
        profile={'audio':{'path':str(audio),'sha256':evaluation.sha256(audio)},
                 'text':{'path':str(text),'files':{name:evaluation.sha256(text/name) for name in evaluation.TEXT_FILES}}}
        self.receipt={'schema':1,'locked_for_test':True,'locked_utc':'2026-10-23T20:00:00+00:00',
                      'selection_split':'dev','test_used_for_selection':False,'recipe':'fixed_recipe',
                      'protocol':{'device':'cpu','max_audio_seconds':10.0,'text_max_tokens':64},
                      'fusion_weights':{'path':str(weights),'sha256':evaluation.sha256(weights)},
                      'baseline':copy.deepcopy(profile),'selected':copy.deepcopy(profile)}
        self.receipt_path=self.root/'receipt.json'; self.save_receipt()

    # Purpose: Persist the current synthetic recipe lock.
    # Input: None explicitly; uses self.receipt and self.receipt_path.
    # Output: None; writes the fixture receipt JSON.
    def save_receipt(self):
        self.receipt_path.write_text(json.dumps(self.receipt))

    # Purpose: Verify missing lock stale checkpoint and incomplete text hashes fail before test access.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_missing_lock_stale_checkpoint_and_incomplete_text_hashes_fail_before_test_access(self):
        variants=[]
        lock=copy.deepcopy(self.receipt); lock['locked_for_test']=False; variants.append(lock)
        stale=copy.deepcopy(self.receipt); stale['selected']['audio']['sha256']='0'*64; variants.append(stale)
        missing=copy.deepcopy(self.receipt); del missing['selected']['text']['files']['tokenizer.json']; variants.append(missing)
        for receipt in variants:
            self.receipt_path.write_text(json.dumps(receipt))
            with self.subTest(receipt=receipt), patch.object(evaluation,'load_test_inputs') as load, self.assertRaises(ValueError):
                evaluation.main(['test','--receipt',str(self.receipt_path),'--repo',str(self.root),'--output-dir',str(self.output)])
            load.assert_not_called()

    # Purpose: Verify missing words choose the live audio vision subset weights.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_missing_words_choose_the_live_audio_vision_subset_weights(self):
        weights={'live':{'audio':1,'text':1,'vision':10},
                 'all':{'audio+vision':{'audio':1,'vision':0}}}
        readings={'audio':reading('anger'),'text':None,'vision':reading('joy')}
        label=evaluation.fused_label(readings,evaluation.MODES['audio_text_vision'],weights)
        self.assertEqual(label,EMOTIONS.index('anger'))
        self.assertEqual(evaluation.fused_label(readings,('text',),weights),EMOTIONS.index('neutral'))

    # Purpose: Verify end to end fake inference scores the locked recipe without selecting.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_end_to_end_fake_inference_scores_the_locked_recipe_without_selecting(self):
        directory=self.root/'data/meld'; directory.mkdir(parents=True)
        ids=[f'clip{i}' for i in range(7)]
        frame=pd.DataFrame({'id':ids,'status':['ok']*7,'emotion':EMOTIONS,'label':list(range(7)),
                            'Dialogue_ID':[0]*7,'wav_path':[f'wav/test/{id_}.wav' for id_ in ids]})
        frame.to_csv(directory/'test.csv',index=False)
        pd.DataFrame({'id':ids,'whisper':['']+['spoken']*6}).to_csv(directory/'whisper_test.csv',index=False)
        vision=frame[['id','emotion']].copy()
        for emotion in EMOTIONS: vision[f'p_{emotion}']=0.02
        vision['p_neutral']=0.88; vision.to_csv(directory/'vision_test.csv',index=False)

        class Audio:
            classes=EMOTIONS
            # Purpose: Accept the audio classifier constructor interface in a test fixture.
            # Input: *args/**kwargs: ignored classifier configuration.
            # Output: None; no model is created.
            def __init__(self,*args,**kwargs): pass
            # Purpose: Stand in for loading the synthetic audio classifier.
            # Input: None.
            # Output: None; no weights are loaded.
            def load(self): pass
            # Purpose: Return a fixed voice reading for locked-recipe evaluation.
            # Input: wav: ignored waveform fixture.
            # Output: Synthetic anger reading.
            def predict(self,wav): return reading('anger')
        class Text(Audio):
            model=SimpleNamespace(config=SimpleNamespace(id2label=dict(enumerate(EMOTIONS))))
            # Purpose: Return a fixed words reading only when a transcript exists.
            # Input: text: transcript fixture.
            # Output: Synthetic joy reading, or None for empty text.
            def predict(self,text): return reading('joy') if text else None

        with patch.object(evaluation,'EmotionClassifier',Audio), patch.object(evaluation,'TextEmotionClassifier',Text), \
                patch.object(evaluation,'load_audio',return_value=np.zeros(16000,dtype=np.float32)), \
                contextlib.redirect_stdout(io.StringIO()):
            evaluation.main(['test','--receipt',str(self.receipt_path),'--repo',str(self.root),
                             '--output-dir',str(self.output),'--bootstrap','10','--threads','1'])
        report=json.loads((self.output/'test_frozen_comparison.json').read_text())
        self.assertEqual(report['locked_recipe'],'fixed_recipe')
        self.assertTrue(report['method']['recipe_locked_before_test'])
        self.assertFalse(report['method']['fitting_or_tuning'])
        self.assertFalse(report['method']['recipe_recommendation'])
        self.assertEqual((self.output/'frozen_selection_receipt.json').read_bytes(),self.receipt_path.read_bytes())
        for comparison in report['comparisons'].values():
            self.assertEqual(comparison['delta']['weighted_f1'],0)
            self.assertEqual(comparison['paired_dialogue_bootstrap_delta_95ci']['weighted_f1'],[0,0])


if __name__=='__main__':
    unittest.main()
