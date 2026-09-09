"""CPU checks for partial-layer hooks and late-window inference."""

from contextlib import nullcontext
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import run_layer_video as layer
import run_replace_video as shared
from test_attention_swap import Model, tensor, save_tensor, load_tensor


class LayerVideoTests(unittest.TestCase):
    def setUp(self):
        self.torch = types.SimpleNamespace(
            save=save_tensor, load=load_tensor, isfinite=np.isfinite,
            amp=types.SimpleNamespace(autocast=lambda *a, **kw: nullcontext()))
        self.modules = patch.dict('sys.modules', {
            'torch': self.torch,
            'tqdm': types.SimpleNamespace(tqdm=lambda values, **kw: values)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def model(self):
        model = Model(30)
        for i, block in enumerate(model.blocks):
            block.self_attn.factor = (i + 1) * 0.001
        return model

    def test_groups_cover_30_layers_and_six_distinct_outputs(self):
        self.assertEqual(layer.SWAP_UPDATES, (20, 21, 22, 23, 24))
        self.assertEqual([i for _, group in layer.layer_groups() for i in group], list(range(30)))
        experiments = layer.layer_experiments()
        self.assertEqual(len({item['filename'] for item in experiments}), 6)
        self.assertTrue(all(item['filename'].endswith('_updates21-25.mp4') for item in experiments))
        self.assertEqual(len(layer.layer_experiments('middle')), 2)
        with self.assertRaises(ValueError):
            layer.layer_groups('invalid')

    def test_pairs_have_twenty_layers_distinct_names_and_separate_outputs(self):
        expected = {'front_middle': tuple(range(20)),
                    'middle_back': tuple(range(10, 30)),
                    'front_back': tuple(range(10)) + tuple(range(20, 30))}
        self.assertEqual(dict(layer.layer_groups('pairs')), expected)
        for mode, indices in expected.items():
            self.assertEqual(len(set(indices)), 20)
            self.assertEqual(layer.layer_groups(mode), [(mode, indices)])
            self.assertEqual(len(layer.layer_experiments(mode)), 2)
            self.assertEqual(layer.layer_output_directory('clip', mode).name, 'attention_layer_pairs')
        names = {item['filename'] for item in layer.layer_experiments('pairs')}
        old_names = {item['filename'] for item in layer.layer_experiments()}
        self.assertEqual(len(names), 6)
        self.assertTrue(names.isdisjoint(old_names))
        self.assertIn('layer_front_back_00-09+20-29_swap_A_from_B_updates21-25.mp4', names)
        self.assertEqual(layer.layer_output_directory('clip', 'pairs').name, 'attention_layer_pairs')
        self.assertEqual(layer.layer_output_directory('clip', 'all').name, 'attention_layers')

    def test_only_selected_hooks_replace_outputs_other_layers_use_recipient_state(self):
        model = self.model()
        factors = [block.self_attn.factor for block in model.blocks]
        for _, indices in layer.layer_groups() + layer.layer_groups('pairs'):
            with tempfile.TemporaryDirectory() as directory:
                with shared.attention_feature_hooks(model, directory, 'capture', indices):
                    self.assertEqual([i for i, b in enumerate(model.blocks) if b.self_attn.hooks], list(indices))
                    donor_output = model(tensor([1]))
                self.assertEqual({p.name for p in Path(directory).glob('*.pt')},
                                 {f'layer_{i:02d}.pt' for i in indices})
                # Independent residual-block calculation: only selected attention
                # outputs come from the donor; all other outputs use recipient state.
                donor, expected = 1.0, 3.0
                for i, factor in enumerate(factors):
                    expected += factor * (donor if i in indices else expected)
                    donor += factor * donor
                with shared.attention_feature_hooks(model, directory, 'replace', indices):
                    actual = model(tensor([3]))
                np.testing.assert_allclose(actual, [expected], rtol=1e-6)
                self.assertFalse(np.allclose(actual, donor_output))
                with shared.attention_feature_hooks(model, directory, 'replace', indices):
                    self_result = model(tensor([1]))
                np.testing.assert_array_equal(self_result, donor_output)
                self.assertTrue(all(not b.self_attn.hooks for b in model.blocks))

    def test_invalid_groups_and_exception_cleanup(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            for indices in ((), (0, 0), (-1,), (30,), (1.5,)):
                with self.assertRaises(ValueError):
                    with shared.attention_feature_hooks(model, directory, 'capture', indices):
                        pass
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                with shared.attention_feature_hooks(model, directory, 'capture', range(10, 20)):
                    raise RuntimeError('failed')
            self.assertTrue(all(not b.self_attn.hooks for b in model.blocks))

    def test_all_groups_both_cfg_branches_only_at_updates_21_to_25(self):
        class Scheduler:
            def __init__(self):
                self.timesteps = [tensor(i) for i in range(50, 0, -1)]
                self.calls = []
                self.history = tensor(0)

            def set_begin_index(self, index):
                self.begin = index

            def step(self, velocity, timestep, sample, return_dict=False):
                self.calls.append(int(timestep))
                result = sample - 0.001 * (velocity + 0.1 * self.history)
                self.history = velocity.clone()
                return (result,)

        blocks_model = self.model()

        class DiT:
            blocks = blocks_model.blocks

            def __call__(self, values, t, context, seq_len):
                return [blocks_model(values[0]) + context]

        pipe = types.SimpleNamespace(model=DiT(), device='cpu', param_dtype=np.float32,
                                     patch_size=(1, 2, 2))
        initial = {'A': tensor(np.ones((1, 1, 2, 2))),
                   'B': tensor(np.full((1, 1, 2, 2), 3))}
        baselines = {}
        with tempfile.TemporaryDirectory() as directory:
            for name in ('A', 'B'):
                _, baselines[name], _ = shared.attention_trajectory(
                    pipe, Scheduler(), initial[name], 1, 0, 2, directory,
                    capture=True, capture_indices=range(26))
            for experiment in layer.layer_experiments() + layer.layer_experiments('pairs'):
                recipient, donor = experiment['recipient'], experiment['donor']
                solver, metrics = Scheduler(), []
                with patch.object(shared, 'swapped_model_prediction', wraps=shared.swapped_model_prediction) as spy:
                    final, _, _ = shared.attention_trajectory(
                        pipe, solver, initial[recipient], 1, 0, 2, directory,
                        swap_indices=layer.SWAP_UPDATES, layer_indices=experiment['layers'],
                        donor_snapshots=baselines[donor], recipient_snapshots=baselines[recipient],
                        diagnostics=metrics)
                self.assertEqual(len(solver.calls), 25)
                self.assertEqual(solver.begin, 25)
                self.assertEqual(len(spy.call_args_list), 10)
                for n, call in enumerate(spy.call_args_list):
                    index = 20 + n // 2
                    self.assertEqual(call.kwargs['layer_indices'], experiment['layers'])
                    self.assertEqual(call.args[4], 1 if n % 2 == 0 else 0)
                    np.testing.assert_array_equal(call.args[2], baselines[donor][index])
                    self.assertEqual(float(call.args[3].flat[0]), 25 - index)
                self.assertEqual([m['update_1based'] for m in metrics if m['intervened']], [21, 22, 23, 24, 25])
                self.assertTrue(all(m['trajectory_vs_recipient_baseline']['rmse'] == 0 for m in metrics[:20]))
                self.assertGreater(metrics[-1]['local_next_latent']['rmse'], 0)
                self.assertFalse(np.allclose(final, baselines[recipient][25]))
                self.assertTrue(all(not b.self_attn.hooks for b in blocks_model.blocks))

    def test_pipeline_prepares_and_forwards_mode(self):
        self.torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        for mode in ('back', 'pairs', *layer.PAIR_MODES):
            with patch.object(layer, 'resolve_checkpoint', return_value=Path('models')), \
                    patch.object(layer, 'prepare_video') as prepare, \
                    patch.object(layer, 'run_layer_swap') as run:
                layer.run_pipeline('models', 'clip', mode)
            self.assertEqual(prepare.call_args.args[1].name, 'clip_lowResolution.mp4')
            self.assertEqual(run.call_args.args[0].layer_mode, mode)


if __name__ == '__main__':
    unittest.main()
