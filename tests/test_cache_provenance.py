"""Small cache-finalization/provenance regressions without ASR or classifier loading."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from training import train_text, transcribe_cpu


class CacheReviewTests(unittest.TestCase):
    # Purpose: Create a temporary manifest, completed transcript CSV, and provenance sidecar.
    # Input: None.
    # Output: None; initializes isolated fixtures and registers cleanup where needed.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame = pd.DataFrame({'id':['a'], 'utterance':['same words'], 'status':['ok'],
                                   'gold':['same words'], 'label':[4], 'emotion':['neutral']})
        self.output = self.root/'cache.csv'
        pd.DataFrame([{'id':'a','whisper':'same words','hit_token_limit':False}]).to_csv(self.output,index=False)
        self.sidecar = Path(str(self.output)+'.meta.json')

    # Purpose: Verify resume finalizes after last csv write without loading asr.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_resume_finalizes_after_last_csv_write_without_loading_asr(self):
        manifest = self.root/'train.csv'
        self.frame.to_csv(manifest,index=False)
        self.sidecar.write_text(json.dumps({'backend':transcribe_cpu.BACKEND,'split':'train',
                                           'manifest_sha256':transcribe_cpu.sha256(manifest),
                                           'num_shards':1,'shard_index':0,'complete':False,
                                           'model':{'source_weights_sha256':'unchanged'}}))
        argv=['transcribe_cpu','--split','train','--output',str(self.output)]
        with patch.object(transcribe_cpu,'MELD_DIR',self.root), patch('sys.argv',argv), \
                patch.object(transcribe_cpu,'CPUWhisperASR') as asr, contextlib.redirect_stdout(io.StringIO()):
            transcribe_cpu.main()
        asr.assert_not_called()
        metadata=json.loads(self.sidecar.read_text())
        self.assertTrue(metadata['complete'])
        self.assertEqual(metadata['n_cached'],1)
        self.assertEqual(metadata['word_error_rate'],0)
        self.assertEqual(metadata['cache_sha256'],transcribe_cpu.sha256(self.output))
        self.assertEqual(metadata['model']['source_weights_sha256'],'unchanged')

    # Purpose: Verify completed cache cannot silently change on resume.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_completed_cache_cannot_silently_change_on_resume(self):
        rows=pd.read_csv(self.output).to_dict('records')
        with self.assertRaisesRegex(ValueError,'content changed'):
            transcribe_cpu.finalize_cache(self.frame,rows,{'complete':True,'cache_sha256':'wrong'},self.output)

    # Purpose: Verify training rejects sidecar checksum mismatch.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_training_rejects_sidecar_checksum_mismatch(self):
        self.sidecar.write_text(json.dumps({'split':'train','complete':True,'cache_sha256':'wrong'}))
        with self.assertRaisesRegex(ValueError,'checksum'):
            train_text.attach_asr(self.frame,self.output,'train')

    # Purpose: Verify incomplete sidecar requires explicit partial mode.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_incomplete_sidecar_requires_explicit_partial_mode(self):
        self.sidecar.write_text(json.dumps({'split':'train','complete':False}))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            train_text.attach_asr(self.frame,self.output,'train')
        joined, provenance=train_text.attach_asr(self.frame,self.output,'train',allow_partial=True)
        self.assertEqual(joined.whisper.tolist(),['same words'])
        self.assertFalse(provenance['generation_metadata']['complete'])


if __name__=='__main__':
    unittest.main()
