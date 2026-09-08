"""CPU checks for raw residual and experiment initialization."""

import unittest

import numpy as np

from run_diff_video import generation_experiments, mix_initial_latent, raw_latent_residual


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
        self.assertEqual([w for branch, w, _ in experiments if branch == 'first_frame_plus_diff'], [0, 0.1, 0.3, 1])
        self.assertEqual([w for branch, w, _ in experiments if branch == 'origin_minus_diff'], [0, -0.1, -0.3, -1])
        self.assertEqual(len({name for _, _, name in experiments}), 8)

    def test_original_subtraction_and_first_frame_addition(self):
        noise = np.full((1, 2, 2, 2), 2.0)
        original = np.full_like(noise, 10.0)
        first = np.full_like(noise, 4.0)
        residual = np.full_like(noise, 3.0)
        np.testing.assert_allclose(mix_initial_latent(noise, residual, first, 0.25, weight=1), 6.5)
        np.testing.assert_allclose(mix_initial_latent(noise, residual, original, 0.25, weight=-1), 5.0)

    def test_raw_residual_retains_both_signs_and_clean_latent_identities(self):
        original = np.array([1., -3., 7.])
        first = np.array([2., 4., -1.])
        residual = raw_latent_residual(original, first)
        np.testing.assert_array_equal(residual, [-1., -7., 8.])
        np.testing.assert_array_equal(first + residual, original)
        np.testing.assert_array_equal(original - residual, first)
        np.testing.assert_array_equal(raw_latent_residual(first, first), np.zeros_like(first))

    def test_raw_residual_rejects_broadcasting(self):
        with self.assertRaises(ValueError):
            raw_latent_residual(np.zeros((2, 3)), np.zeros((1, 3)))

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
