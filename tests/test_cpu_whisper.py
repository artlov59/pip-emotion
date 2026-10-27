"""Small deterministic conversion tests; source coverage also verified by CPUWhisperASR.load()."""
import unittest
import numpy as np
import torch
from app.cpu_asr import CPUWhisperASR, convert_tensor, hf_key, sinusoids, tokenizer_module


class ConversionTests(unittest.TestCase):
    # Purpose: Verify empty batch needs no loaded model.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_empty_batch_needs_no_loaded_model(self):
        adapter = CPUWhisperASR.__new__(CPUWhisperASR)
        self.assertEqual(adapter.transcribe_batch([]), [])

    # Purpose: Verify invalid batch inputs are rejected before feature extraction.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_invalid_batch_inputs_are_rejected_before_feature_extraction(self):
        adapter = CPUWhisperASR.__new__(CPUWhisperASR)
        for invalid in [np.zeros(0), np.zeros((2, 10)), np.array([np.nan]), np.zeros(480001)]:
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                adapter.transcribe_batch([invalid])

    # Purpose: Verify convolution axis conversion preserves every value.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_convolution_axis_conversion_preserves_every_value(self):
        values = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
        tensor = convert_tensor('encoder.conv1.weight', values)
        self.assertEqual(tuple(tensor.shape), (2, 4, 3))
        self.assertEqual(tensor.dtype, torch.float32)
        np.testing.assert_array_equal(tensor.numpy().transpose(0, 2, 1), values)

    # Purpose: Verify linear projection is not transposed.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_linear_projection_is_not_transposed(self):
        values = np.arange(12, dtype=np.float16).reshape(3, 4)
        np.testing.assert_array_equal(convert_tensor('decoder.blocks.0.attn.query.weight', values).numpy(), values)

    # Purpose: Verify attention and projection key mapping.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_attention_and_projection_key_mapping(self):
        expected = {
            'encoder.blocks.2.attn.query.weight': 'model.encoder.layers.2.self_attn.q_proj.weight',
            'decoder.blocks.3.cross_attn_ln.bias': 'model.decoder.layers.3.encoder_attn_layer_norm.bias',
            'decoder.blocks.4.cross_attn.out.weight': 'model.decoder.layers.4.encoder_attn.out_proj.weight',
            'decoder.blocks.5.mlp1.weight': 'model.decoder.layers.5.fc1.weight',
            'encoder.blocks.0.mlp_ln.weight': 'model.encoder.layers.0.final_layer_norm.weight',
            'decoder.token_embedding.weight': 'model.decoder.embed_tokens.weight',
            'decoder.positional_embedding': 'model.decoder.embed_positions.weight',
            'encoder.ln_post.bias': 'model.encoder.layer_norm.bias',
            'decoder.ln.weight': 'model.decoder.layer_norm.weight',
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(hf_key(source), target)

    # Purpose: Verify fixed positions and packaged tokenizer.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_fixed_positions_and_packaged_tokenizer(self):
        positions = sinusoids(3, 4)
        torch.testing.assert_close(positions[0], torch.tensor([0., 0., 1., 1.]))
        tokenizer = tokenizer_module().get_tokenizer(True, language='en', task='transcribe')
        self.assertEqual(tokenizer.sot, 50258)
        self.assertEqual(tokenizer.eot, 50257)
        text = 'A locally decoded sentence.'
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


if __name__ == '__main__':
    unittest.main()
