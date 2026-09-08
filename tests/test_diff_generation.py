"""CPU checks for residual normalization and experiment initialization."""

import unittest

import numpy as np

from run_diff_video import generation_experiments, mix_initial_latent, normalize_dark_residual


class DiffGenerationTests(unittest.TestCase):
    def test_weight_sweep_changes_only_residual_component(self):
        noise = np.full((2, 3, 4, 4), 2.0)
        source = np.full_like(noise, 10.0)
        residual = np.full_like(noise, 3.0)
        baseline = mix_initial_latent(noise, residual, source, 0.25, weight=0)
        for weight in (0, 0.1, 0.3, 1):
            actual = mix_initial_latent(noise, residual, source, 0.25, weight=weight)
            np.testing.assert_allclose(actual - baseline, weight * residual)
        np.testing.assert_array_equal(noise, 2.0)
        np.testing.assert_array_equal(source, 10.0)
        np.testing.assert_array_equal(residual, 3.0)

    def test_zero_weight_omits_residual_and_does_not_alias_noise(self):
        noise = np.ones((1, 1, 2, 2))
        residual = np.full_like(noise, np.nan)
        result = mix_initial_latent(noise, residual, weight=0)
        np.testing.assert_array_equal(result, noise)
        result[:] = 100
        np.testing.assert_array_equal(noise, 1)

    def test_comparison_outputs_are_complete_and_unique(self):
        experiments = generation_experiments()
        self.assertEqual([weight for branch, weight, _ in experiments if branch == 'condition'], [0, 0.1, 0.3, 1])
        self.assertEqual(len({filename for _, _, filename in experiments}), len(experiments))
        self.assertIn(('condition', 1.0, 'condition_noise_with_diff.mp4'), experiments)

    def test_dark_side_and_shared_temporal_scale(self):
        # Equal channel values make this a grayscale example over two frames.
        frames = np.array([0, 128, 255, 64, 128, 255], dtype=np.uint8).reshape(2, 1, 1, 3)
        result, peak = normalize_dark_residual(frames, baseline=128)
        np.testing.assert_array_equal(result.reshape(-1), [0, 255, 255, 128, 255, 255])
        self.assertEqual(peak, 128)

    def test_no_dark_residual_is_finite_white(self):
        result, peak = normalize_dark_residual(np.full((2, 2, 2, 3), 255, dtype=np.uint8))
        self.assertTrue((result == 255).all())
        self.assertEqual(peak, 0)

    def test_random_and_conditioned_addition(self):
        noise = np.full((2, 3, 4, 4), 2.0)
        residual = np.full_like(noise, 3.0)
        source = np.full_like(noise, 10.0)
        np.testing.assert_array_equal(mix_initial_latent(noise, residual), 5.0)
        np.testing.assert_array_equal(mix_initial_latent(noise, residual, source, 0.25), 11.0)
        # Source must disappear at sigma=1, and be retained at sigma=0.
        np.testing.assert_array_equal(mix_initial_latent(noise, residual, source, 1), 5.0)
        np.testing.assert_array_equal(mix_initial_latent(noise, residual, source, 0), 13.0)
        np.testing.assert_array_equal(noise, 2.0)

    def test_no_broadcasting_or_invalid_sigma(self):
        value = np.zeros((2, 3, 4, 4))
        with self.assertRaises(ValueError):
            mix_initial_latent(value, value[:1])
        with self.assertRaises(ValueError):
            mix_initial_latent(value, value, value, 1.1)


if __name__ == "__main__":
    unittest.main()
