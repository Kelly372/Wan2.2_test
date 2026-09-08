"""CPU checks of the actual scheduler setup and native I2V call contracts."""

import ast
from pathlib import Path
import types
import unittest

import numpy as np

from run_sigma_video import make_scheduler, schedule_metadata, sigma_grid
from run_i2v_clean import native_branches, padded_frame_count


class Array(np.ndarray):
    def to(self, device=None, dtype=None):
        return self.astype(dtype or self.dtype).view(Array)


def scheduler_setup_method():
    # Execute the repository's real setup method, without importing its GPU stack.
    path = Path(__file__).resolve().parents[1] / 'wan/utils/fm_solvers_unipc.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'FlowUniPCMultistepScheduler')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'set_timesteps')
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {'np': np, 'torch': types.SimpleNamespace(
        from_numpy=lambda value: value.view(Array), int64=np.int64)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['set_timesteps']


class Scheduler:
    set_timesteps = scheduler_setup_method()

    def __init__(self, num_train_timesteps, shift, use_dynamic_shifting):
        self.config = types.SimpleNamespace(num_train_timesteps=num_train_timesteps,
                                            shift=shift, use_dynamic_shifting=use_dynamic_shifting,
                                            final_sigmas_type='zero', solver_order=2)
        self.sigma_max = float(np.float32(1 - 1 / num_train_timesteps))
        self.sigma_min = 0.0
        self.solver_p = None

    def set_begin_index(self, index):
        self._begin_index = index

    @property
    def begin_index(self):
        return self._begin_index


class SigmaTests(unittest.TestCase):
    def test_actual_sigma_not_shifted_twice_and_25_updates(self):
        for target in (0.2, 0.4, 0.6):
            scheduler = make_scheduler(Scheduler, target, 25, 'cpu', 5, 1000)
            info = schedule_metadata(scheduler)
            self.assertAlmostEqual(info['start_sigma'], target, places=6)
            self.assertEqual(info['steps'], 25)
            self.assertEqual(len(info['sigmas']), 26)
            self.assertEqual(info['sigmas'][-1], 0)
            self.assertTrue(np.all(np.diff(info['sigmas']) < 0))
            self.assertEqual(len(set(info['timesteps'])), 25)
            self.assertEqual(scheduler.begin_index, 0)
            self.assertTrue(all(x is None for x in scheduler.model_outputs))

    def test_reference_exactly_matches_previous_q1_tail(self):
        previous = Scheduler(1000, 1, False)
        previous.set_timesteps(50, device='cpu', shift=5)
        reference = make_scheduler(Scheduler, None, 25, 'cpu', 5, 1000)
        info = schedule_metadata(reference)
        self.assertEqual(info['timesteps'], previous.timesteps[25:].tolist())
        self.assertEqual(info['sigmas'], previous.sigmas[25:].tolist())
        self.assertAlmostEqual(info['start_sigma'], 0.833055, places=6)

    def test_custom_update_count_preserves_reference_sigma(self):
        a = make_scheduler(Scheduler, None, 25, 'cpu', 5, 1000)
        b = make_scheduler(Scheduler, None, 10, 'cpu', 5, 1000)
        self.assertEqual(schedule_metadata(b)['steps'], 10)
        self.assertAlmostEqual(float(a.sigmas[25]), float(b.sigmas[0]), places=6)

    def test_invalid_sigma_grid(self):
        for sigma in (0, 1, -0.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                sigma_grid(sigma, 25, 5)
        with self.assertRaises(ValueError):
            sigma_grid(0.2, 0, 5)


class NativeI2VTests(unittest.TestCase):
    def test_padding_preserves_supported_lengths(self):
        self.assertEqual([padded_frame_count(n) for n in (1, 2, 99, 101, 221)],
                         [1, 5, 101, 101, 221])
        with self.assertRaises(ValueError):
            padded_frame_count(0)

    def test_native_pair_shares_sampling_settings_and_runs_lazily(self):
        calls = []

        def call(name, kwargs):
            calls.append((name, kwargs))
            return name

        pipe = types.SimpleNamespace(vae_stride=(4, 16, 16), sample_neg_prompt='negative',
                                     i2v=lambda **kw: call('i2v', kw),
                                     t2v=lambda **kw: call('t2v', kw))
        image = types.SimpleNamespace(width=672, height=384)
        args = types.SimpleNamespace(prompt='scene', inference_step=25, seed=42)
        cfg = types.SimpleNamespace(sample_shift=5, sample_guide_scale=5)
        branches = native_branches(pipe, image, 99, args, cfg)
        self.assertEqual(calls, [])
        self.assertEqual(next(branches), ('i2v_clean.mp4', 'i2v'))
        self.assertEqual(len(calls), 1)
        self.assertEqual(next(branches), ('i2v_no_anchor.mp4', 't2v'))
        first, second = calls[0][1].copy(), calls[1][1].copy()
        self.assertIs(first.pop('img'), image)
        self.assertEqual(first.pop('max_area'), 672 * 384)
        self.assertEqual(second.pop('size'), (672, 384))
        self.assertEqual(first, second)
        self.assertEqual(first['frame_num'], 101)
        self.assertEqual(first['sampling_steps'], 25)
        self.assertEqual(first['seed'], 42)
        self.assertTrue(first['offload_model'])


if __name__ == '__main__':
    unittest.main()
