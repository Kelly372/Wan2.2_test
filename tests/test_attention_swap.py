"""CPU contract tests for attention hooks; lightweight modules emulate hook calls."""

from pathlib import Path
from contextlib import nullcontext
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from run_replace_video import attention_feature_hooks, attention_swap_points
import run_replace_video as pipeline


class Tensor(np.ndarray):
    @property
    def device(self):
        return 'cpu'

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return self.copy()

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis).view(Tensor)

    def expand(self, *shape):
        return np.broadcast_to(self, shape).view(Tensor)

    def to(self, device=None, dtype=None):
        return self.astype(dtype or self.dtype).view(Tensor)


def tensor(value):
    return np.asarray(value, dtype=np.float32).view(Tensor)


def save_tensor(value, path):
    with open(path, 'wb') as output:
        np.save(output, np.asarray(value), allow_pickle=False)


def load_tensor(path, **kwargs):
    with open(path, 'rb') as source:
        return np.load(source, allow_pickle=False).view(Tensor)


class Attention:
    def __init__(self, factor):
        self.factor = factor
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return types.SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def __call__(self, value):
        output = value * self.factor
        for hook in self.hooks:
            replacement = hook(self, (value,), output)
            if replacement is not None:
                output = replacement
        return output


class Model:
    def __init__(self, count=2):
        self.blocks = [types.SimpleNamespace(self_attn=Attention(i + 1)) for i in range(count)]

    def __call__(self, value):
        for block in self.blocks:
            value = value + block.self_attn(value)
        return value


class AttentionSwapTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.dict('sys.modules', {'torch': types.SimpleNamespace(save=save_tensor, load=load_tensor)})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_pipeline_prepares_inputs_and_runs_exchange_only(self):
        events = []
        cuda = types.SimpleNamespace(is_available=lambda: True)
        with patch.dict('sys.modules', {'torch': types.SimpleNamespace(cuda=cuda)}), \
                patch.object(pipeline, 'resolve_checkpoint', return_value=Path('models')), \
                patch.object(pipeline, 'prepare_video', side_effect=lambda *args: events.append('prepare')) as prepare, \
                patch.object(pipeline, 'run_attention_swap', side_effect=lambda args: events.append('swap')) as swap:
            pipeline.run_pipeline('models', 'clip')
            self.assertEqual(events, ['prepare', 'swap'])
            self.assertEqual(prepare.call_args.args[1].name, 'clip_lowResolution.mp4')
            self.assertEqual(swap.call_args.args[0].video_tag, 'clip')

    def test_all_layers_replaced_while_recipient_residual_is_retained(self):
        model = Model()
        with tempfile.TemporaryDirectory() as directory:
            with attention_feature_hooks(model, directory, 'capture'):
                donor = model(tensor([1]))
            with attention_feature_hooks(model, directory, 'replace'):
                recipient = model(tensor([10]))
            np.testing.assert_array_equal(donor, [6])
            # Donor layer outputs are 1 and 4; recipient retains 10 -> 11 -> 15.
            np.testing.assert_array_equal(recipient, [15])
            self.assertTrue(all(not b.self_attn.hooks for b in model.blocks))

    def test_self_exchange_and_all_thirty_layer_files(self):
        model = Model(30)
        with tempfile.TemporaryDirectory() as directory:
            with attention_feature_hooks(model, directory, 'capture'):
                baseline = model(tensor([0.001]))
            self.assertEqual(len(list(Path(directory).glob('layer_*.pt'))), 30)
            with attention_feature_hooks(model, directory, 'replace'):
                control = model(tensor([0.001]))
            np.testing.assert_array_equal(control, baseline)

    def test_exception_removes_hooks_and_shape_mismatch_is_rejected(self):
        model = Model()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'forward failed'):
                with attention_feature_hooks(model, directory, 'capture'):
                    raise RuntimeError('forward failed')
            self.assertTrue(all(not b.self_attn.hooks for b in model.blocks))
            with attention_feature_hooks(model, directory, 'capture'):
                model(tensor([1]))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                with attention_feature_hooks(model, directory, 'replace'):
                    model(tensor([1, 2]))
            self.assertTrue(all(not b.self_attn.hooks for b in model.blocks))

    def test_missing_layer_execution_is_not_silent(self):
        model = Model()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Only 1/2'):
                with attention_feature_hooks(model, directory, 'capture'):
                    model.blocks[0].self_attn(tensor([1]))

    def test_remaining_steps_match_update_indices(self):
        self.assertEqual(attention_swap_points(), ((0, 25), (5, 20), (10, 15), (15, 10), (20, 5)))

    def test_trajectory_swaps_once_for_each_cfg_branch_and_continues_25_updates(self):
        class Scheduler:
            def __init__(self):
                self.timesteps = [tensor(value) for value in range(50, 0, -1)]
                self.calls = []

            def set_begin_index(self, index):
                self.begin = index

            def step(self, velocity, timestep, sample, return_dict=False):
                self.calls.append((float(timestep), sample.copy(), velocity.copy()))
                return (sample - velocity * 0.01,)

        def model(values, t, context, seq_len):
            return [values[0] * 0.1 + context]

        pipe = types.SimpleNamespace(device='cpu', patch_size=(1, 2, 2),
                                     param_dtype=np.float32, model=model)
        fake_torch = types.SimpleNamespace(
            amp=types.SimpleNamespace(autocast=lambda *a, **kw: nullcontext()),
            isfinite=np.isfinite,
        )
        initial = tensor(np.ones((1, 1, 2, 2)))
        donors = {index: tensor(np.full(initial.shape, 100 + index))
                  for index, _ in attention_swap_points()}
        for exchange_index, _ in attention_swap_points():
            scheduler = Scheduler()
            exchanges = []

            def exchange(pipe, recipient, donor, timestep, context, seq_len, cache_dir):
                exchanges.append((recipient.copy(), donor.copy(), context))
                return donor * 0.1 + context

            with tempfile.TemporaryDirectory() as directory, \
                    patch.dict('sys.modules', {
                        'torch': fake_torch,
                        'tqdm': types.SimpleNamespace(tqdm=lambda values, **kwargs: values),
                    }), \
                    patch.object(pipeline, 'swapped_model_prediction', side_effect=exchange):
                final, snapshots, _ = pipeline.attention_trajectory(
                    pipe, scheduler, initial, 1, 0, 2, directory,
                    swap_index=exchange_index, donor_snapshots=donors, capture=True)
            self.assertEqual(scheduler.begin, 25)
            self.assertEqual(len(scheduler.calls), 25)
            self.assertEqual([context for _, _, context in exchanges], [1, 0])
            for recipient, donor, _ in exchanges:
                np.testing.assert_array_equal(donor, donors[exchange_index])
                np.testing.assert_array_equal(recipient, snapshots[exchange_index])
            # The solver updates the recipient latent, not the donor latent.
            np.testing.assert_array_equal(scheduler.calls[exchange_index][1][0], snapshots[exchange_index])
            np.testing.assert_allclose(scheduler.calls[exchange_index][2][0], donors[exchange_index] * 0.1 + 2)
            self.assertEqual(set(snapshots), {0, 5, 10, 15, 20})
            self.assertEqual(final.shape, initial.shape)
            np.testing.assert_array_equal(initial, 1)


if __name__ == '__main__':
    unittest.main()
