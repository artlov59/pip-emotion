"""CPU-only regressions for padded CNN pooling and mixup; no data/model downloads.

    python -m unittest tests.test_model_training
"""
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from model import AudioCNN, temporal_pool_lengths
from training.train import mixup


class TemporalPoolingTests(unittest.TestCase):
    # Purpose: Verify exact lengths match five stride two stages.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_exact_lengths_match_five_stride_two_stages(self):
        lengths = torch.tensor([1, 2, 31, 32, 33, 63, 64, 65, 399, 400, 401, 1000])
        pool = nn.MaxPool1d(3, stride=2, padding=1)
        expected = []
        for length in lengths.tolist():
            x = torch.ones(1, 1, length)
            for _ in range(5):
                x = pool(x)
            expected.append(x.shape[-1])
        actual = temporal_pool_lengths(lengths, 1000, 32, "exact")
        torch.testing.assert_close(actual, torch.tensor(expected))

    # Purpose: Verify exact lengths do not depend on batch padding.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_exact_lengths_do_not_depend_on_batch_padding(self):
        lengths = torch.tensor([31, 32, 33])
        small_batch = temporal_pool_lengths(lengths, 400, 13, "exact")
        large_batch = temporal_pool_lengths(lengths, 1000, 32, "exact")
        torch.testing.assert_close(small_batch, torch.tensor([1, 1, 2]))
        torch.testing.assert_close(small_batch, large_batch)
        # The historical formula includes an extra padded cell for length 31.
        self.assertEqual(temporal_pool_lengths(lengths, 400, 13, "legacy")[0].item(), 2)

    # Purpose: Verify legacy default preserves previous formula.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_legacy_default_preserves_previous_formula(self):
        lengths = torch.tensor([0, 1, 31, 32, 33, 200, 400, 1000])
        previous = torch.clamp(torch.ceil(lengths.float() * 13 / 400), min=1, max=13)
        torch.testing.assert_close(temporal_pool_lengths(lengths, 400, 13), previous)

    # Purpose: Verify invalid mode fails before allocating network.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_invalid_mode_fails_before_allocating_network(self):
        with self.assertRaises(ValueError):
            AudioCNN(pooling_mode="typo")
        with self.assertRaises(ValueError):
            temporal_pool_lengths(torch.tensor([32]), 32, 1, "typo")


class MixupTests(unittest.TestCase):
    # Purpose: Create unequal-length clips, labels, and a deterministic mixup donor permutation.
    # Input: None.
    # Output: None; initializes isolated fixtures and registers cleanup where needed.
    def setUp(self):
        self.x = torch.zeros(2, 1, 1, 8)
        self.x[0, ..., :2] = 1
        self.x[1] = 2
        self.lengths = torch.tensor([2, 8])
        self.y = torch.tensor([3, 6])
        self.permutation = torch.tensor([1, 0])

    # Purpose: Run mixup with controlled mixture weight and donor permutation.
    # Input: lam: fixed mixing coefficient; uses fixture features/lengths/labels.
    # Output: Mixup tuple (features, valid lengths, original labels, donor labels, lambda).
    def mixed(self, lam):
        with patch("training.train.np.random.beta", return_value=lam), patch(
            "training.train.torch.randperm", return_value=self.permutation
        ):
            return mixup(self.x, self.lengths, self.y)

    # Purpose: Verify donor audio and labels use the same permutation.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_donor_audio_and_labels_use_the_same_permutation(self):
        x, lengths, ya, yb, lam = self.mixed(0.25)
        torch.testing.assert_close(x, 0.25 * self.x + 0.75 * self.x[self.permutation])
        torch.testing.assert_close(lengths, torch.tensor([8, 8]))
        torch.testing.assert_close(ya, self.y)
        torch.testing.assert_close(yb, self.y[self.permutation])
        torch.testing.assert_close(self.lengths, torch.tensor([2, 8]))
        self.assertEqual(lam, 0.25)

    # Purpose: Verify endpoint mixture keeps only contributing clip length.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_endpoint_mixture_keeps_only_contributing_clip_length(self):
        for lam, expected in [(0.0, self.lengths[self.permutation]), (1.0, self.lengths)]:
            with self.subTest(lam=lam):
                x, lengths, *_ = self.mixed(lam)
                torch.testing.assert_close(lengths, expected)
                torch.testing.assert_close(x, self.x[self.permutation] if lam == 0 else self.x)


if __name__ == "__main__":
    unittest.main()
