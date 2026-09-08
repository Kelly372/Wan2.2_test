"""Single-update, five-update and full-run all-layer attention exchanges.

Usage: python run_replace_video.py --model_path /path/to/Wan2.2-TI2V-5B --video_tag clip
"""

import argparse
import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import tempfile

from run_diff_video import (
    DATA_DIR, file_name, prepare_video, prepare_tensor,
    read_rgb_video, resolve_checkpoint, save_decoded,
)


def attention_swap_points():
    """Zero-based update indices and remaining updates in the 25-step run."""
    return ((0, 25), (5, 20), (10, 15), (15, 10), (20, 5))


def exchange_experiments(experiment_set='all'):
    if experiment_set not in ('all', 'single', 'window', 'full'):
        raise ValueError('Unknown experiment set.')
    groups = [('single', (index,)) for index, _ in attention_swap_points()]
    groups += [('window', tuple(range(start, start + 5))) for start in range(0, 25, 5)]
    groups += [('full', tuple(range(25)))]
    return [(kind, indices) for kind, indices in groups
            if experiment_set in ('all', kind)]


def difference_metrics(changed, reference):
    """CPU float64 reductions avoid adding large temporary GPU allocations."""
    import numpy as np

    if changed.shape != reference.shape:
        raise ValueError('Diagnostic tensors must have identical shapes.')
    a = changed.detach().cpu().float().numpy().astype(np.float64)
    b = reference.detach().cpu().float().numpy().astype(np.float64)
    delta = a - b
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise RuntimeError('Non-finite values in exchange diagnostics.')
    rms = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(b * b)))
    return {'rmse': rms, 'max_abs': float(np.max(np.abs(delta))),
            'reference_rms': reference_rms,
            'relative_l2': rms / max(reference_rms, 1e-12)}


def preview_scheduler_step(scheduler, prediction, timestep, latent):
    """Counterfactual update with identical history; never advance the live solver."""
    preview = copy.deepcopy(scheduler)
    return preview.step(prediction.unsqueeze(0), timestep, latent.clone().unsqueeze(0),
                        return_dict=False)[0].squeeze(0)


@contextmanager
def attention_feature_hooks(model, directory, mode, layer_indices=None):
    """Capture/replace selected self-attention outputs (all layers by default).

    Disk-backed, one layer per file: only the current layer needs a transfer.
    Hooks never target cross-attention, FFN, latent input or scheduler state.
    """
    import torch

    if mode not in ('capture', 'replace'):
        raise ValueError('Unknown attention hook mode.')
    blocks = list(model.blocks)
    if not blocks:
        raise ValueError('No DiT blocks found.')
    indices = tuple(range(len(blocks))) if layer_indices is None else tuple(layer_indices)
    if (not indices or len(set(indices)) != len(indices) or
            any(not isinstance(index, int) or index not in range(len(blocks)) for index in indices)):
        raise ValueError('Layer indices must be nonempty, unique integers within the model.')
    seen = set()
    handles = []

    def hook_for(index):
        def hook(module, inputs, output):
            if index in seen:
                raise RuntimeError(f'Self-attention block {index} ran more than once.')
            seen.add(index)
            path = Path(directory) / f'layer_{index:02d}.pt'
            if mode == 'capture':
                torch.save(output.detach().cpu(), path)
                return None
            donor = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if donor.shape != output.shape or donor.dtype != output.dtype:
                raise ValueError(f'Donor/recipient shape or dtype mismatch at block {index}.')
            return donor.to(device=output.device, dtype=output.dtype)
        return hook

    try:
        for index in indices:
            handles.append(blocks[index].self_attn.register_forward_hook(hook_for(index)))
        yield
        if len(seen) != len(indices):
            raise RuntimeError(f'Only {len(seen)}/{len(indices)} self-attention layers ran.')
    finally:
        for handle in handles:
            handle.remove()


def swapped_model_prediction(pipe, recipient, donor, timestep, context, seq_len, cache_dir,
                             layer_indices=None):
    # The donor forward is never modified, and always starts from its own
    # independently recorded baseline latent at this exact update.
    with attention_feature_hooks(pipe.model, cache_dir, 'capture', layer_indices):
        donor_prediction = pipe.model([donor], t=timestep, context=context, seq_len=seq_len)
    del donor_prediction
    with attention_feature_hooks(pipe.model, cache_dir, 'replace', layer_indices):
        return pipe.model([recipient], t=timestep, context=context, seq_len=seq_len)[0]


def attention_trajectory(pipe, scheduler, initial, context, context_null,
                         guide_scale, output_dir, swap_index=None,
                         donor_snapshots=None, capture=False, self_check=False,
                         swap_indices=None, diagnostics=None,
                         recipient_snapshots=None, capture_indices=None,
                         layer_indices=None):
    import torch
    from tqdm import tqdm

    # Keep the existing 50-step shifted schedule and its final 25 updates.
    start = len(scheduler.timesteps) - 25
    if start < 0:
        raise ValueError('The schedule must have at least 25 timesteps.')
    scheduler.set_begin_index(start)
    timesteps = scheduler.timesteps[start:]
    latent = initial.to(pipe.device).clone()
    _, time, height, width = latent.shape
    seq_len = time * (height // pipe.patch_size[1]) * (width // pipe.patch_size[2])
    selected = {index for index, _ in attention_swap_points()}
    if swap_index is not None and swap_indices is not None:
        raise ValueError('Specify only one exchange index argument.')
    exchange_indices = set(swap_indices or ()) if swap_index is None else {swap_index}
    if any(index not in range(25) for index in exchange_indices) or (
            exchange_indices and (donor_snapshots is None or
                                  not exchange_indices.issubset(donor_snapshots))):
        raise ValueError('Invalid exchange point or missing donor baseline snapshots.')
    capture_set = selected if capture_indices is None else set(capture_indices)
    if recipient_snapshots is not None and not set(range(1, 26)).issubset(recipient_snapshots):
        raise ValueError('Diagnostics require all recipient baseline post-update snapshots.')
    snapshots = {}
    checks = []
    swap_count = 0
    layer_kwargs = {} if layer_indices is None else {'layer_indices': tuple(layer_indices)}
    # Only one CFG pass worth of donor features is kept on disk at a time.
    with tempfile.TemporaryDirectory(prefix='attention_cache_', dir=output_dir) as cache_dir:
        with torch.amp.autocast('cuda', dtype=pipe.param_dtype):
            for index, t in enumerate(tqdm(timesteps, desc='Attention experiment')):
                if capture and index in capture_set:
                    snapshots[index] = latent.detach().cpu().clone()
                timestep = t.reshape(1).expand(1, seq_len)
                predictions = []
                donor = None
                if index in exchange_indices:
                    donor = donor_snapshots[index].to(pipe.device)
                    swap_count += 1
                normal_predictions = []
                record = {'update_1based': index + 1, 'timestep': int(t),
                          'intervened': donor is not None}
                for label, ctx in (('conditional', context), ('unconditional', context_null)):
                    if donor is None:
                        prediction = pipe.model([latent], t=timestep, context=ctx, seq_len=seq_len)[0]
                    else:
                        if diagnostics is not None:
                            normal = pipe.model([latent], t=timestep, context=ctx, seq_len=seq_len)[0]
                            normal_predictions.append(normal)
                        prediction = swapped_model_prediction(
                            pipe, latent, donor, timestep, ctx, seq_len, cache_dir, **layer_kwargs)
                        if diagnostics is not None:
                            record[f'{label}_prediction'] = difference_metrics(prediction, normal)
                            del normal
                    if self_check and index in selected:
                        # This extra forward does not call scheduler.step or change
                        # the trajectory. Re-injecting one's own features must agree.
                        control = swapped_model_prediction(
                            pipe, latent, latent, timestep, ctx, seq_len, cache_dir, **layer_kwargs)
                        max_error = float((prediction - control).abs().max())
                        checks.append({'update': index + 1, 'cfg_branch': label,
                                       'max_abs_prediction_error': max_error})
                        if not torch.allclose(prediction, control, rtol=1e-4, atol=1e-4):
                            raise RuntimeError(f'Self-swap check failed at update {index + 1}, {label}: {max_error}')
                        del control
                    predictions.append(prediction)
                del donor
                velocity = predictions[1] + guide_scale * (predictions[0] - predictions[1])
                if normal_predictions:
                    normal_velocity = normal_predictions[1] + guide_scale * (
                        normal_predictions[0] - normal_predictions[1])
                    record['cfg_prediction'] = difference_metrics(velocity, normal_velocity)
                    no_swap_next = preview_scheduler_step(scheduler, normal_velocity, t, latent)
                    del normal_velocity
                normal_predictions.clear()
                before_update = latent
                latent = scheduler.step(velocity.unsqueeze(0), t, latent.unsqueeze(0),
                                        return_dict=False)[0].squeeze(0)
                if diagnostics is not None:
                    if record['intervened']:
                        record['local_next_latent'] = difference_metrics(latent, no_swap_next)
                        record['local_update'] = difference_metrics(
                            latent - before_update, no_swap_next - before_update)
                        del no_swap_next
                    if recipient_snapshots is not None:
                        record['trajectory_vs_recipient_baseline'] = difference_metrics(
                            latent, recipient_snapshots[index + 1])
                    diagnostics.append(record)
                del before_update
                del predictions, prediction, velocity
    if capture and 25 in capture_set:
        snapshots[25] = latent.detach().cpu().clone()
    if swap_count != len(exchange_indices):
        raise RuntimeError('The trajectory did not execute all requested exchange updates.')
    if not torch.isfinite(latent).all():
        raise RuntimeError('Non-finite final latent in attention experiment.')
    return latent.detach().cpu(), snapshots, checks


def run_attention_swap(args):
    import torch
    from wan import WanTI2V
    from wan.configs import WAN_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    checkpoint = resolve_checkpoint(args.model_path)
    cfg = WAN_CONFIGS['ti2v-5B']
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    output_dir = DATA_DIR / 'video' / args.video_tag / 'attention_extended'
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {'A': DATA_DIR / 'video' / f'{args.video_tag}_lowResolution.mp4',
             'B': DATA_DIR / 'video' / f'{args.video_tag}_first_frame.mp4'}
    pipe = WanTI2V(cfg, str(checkpoint), device_id=0, t5_cpu=True,
                   init_on_cpu=True, convert_model_dtype=True)
    if len(pipe.model.blocks) != 30:
        raise ValueError('Expected the 30-block TI2V-5B model.')

    def new_scheduler():
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps,
                                                shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(50, device=device, shift=cfg.sample_shift)
        return scheduler

    with torch.inference_mode():
        encoded = {}
        shape = None
        fps = None
        for label, path in paths.items():
            frames, current_fps = read_rgb_video(path)
            if shape is None:
                shape, fps = frames.shape, current_fps
            elif frames.shape != shape or not math.isclose(fps, current_fps, rel_tol=1e-4, abs_tol=1e-3):
                raise ValueError('A/B must have matching shape and frame rate.')
            video = prepare_tensor(frames, 32).to(device)
            del frames
            encoded[label] = pipe.vae.encode([video])[0].cpu()
            del video
            if not torch.isfinite(encoded[label]).all():
                raise RuntimeError(f'Non-finite latent for {label}.')
        pipe.vae.model.cpu()
        torch.cuda.empty_cache()
        context = [v.to(device) for v in pipe.text_encoder([''], torch.device('cpu'))]
        context_null = [v.to(device) for v in pipe.text_encoder([pipe.sample_neg_prompt], torch.device('cpu'))]
        schedule = new_scheduler()
        sigma = float(schedule.sigmas[25])
        generator = torch.Generator(device=device).manual_seed(42)
        noise = torch.randn(encoded['A'].shape, device=device, dtype=torch.float32, generator=generator).cpu()
        initial = {label: (1 - sigma) * z + sigma * noise for label, z in encoded.items()}
        del encoded, noise
        pipe.model.to(device)
        snapshots = {}
        outputs = {}
        controls = {}
        # Baselines are completed before any cross-trajectory intervention.
        for label in ('A', 'B'):
            print(f'Baseline Process({label}); 25 updates', flush=True)
            final, snapshots[label], controls[label] = attention_trajectory(
                pipe, new_scheduler(), initial[label], context, context_null,
                cfg.sample_guide_scale, output_dir, capture=True, self_check=True,
                capture_indices=range(26))
            outputs[f'extended_process_{label}_baseline.mp4'] = final
        experiments = []
        for recipient, donor in (('A', 'B'), ('B', 'A')):
            for kind, indices in exchange_experiments(getattr(args, 'experiment_set', 'all')):
                span = f'{indices[0] + 1:02d}-{indices[-1] + 1:02d}'
                filename = f'{kind}_swap_{recipient}_from_{donor}_updates{span}.mp4'
                print(f'{filename}: exchange all layers at updates {span}/25', flush=True)
                diagnostics = []
                final, _, _ = attention_trajectory(
                    pipe, new_scheduler(), initial[recipient], context, context_null,
                    cfg.sample_guide_scale, output_dir, swap_indices=indices,
                    donor_snapshots=snapshots[donor], diagnostics=diagnostics,
                    recipient_snapshots=snapshots[recipient])
                outputs[filename] = final
                info = {'file': filename, 'kind': kind, 'recipient': recipient, 'donor': donor,
                        'updates_1based': [index + 1 for index in indices],
                        'schedule_indices': [25 + index for index in indices],
                        'timesteps': [int(schedule.timesteps[25 + index]) for index in indices],
                        'sigmas': [float(schedule.sigmas[25 + index]) for index in indices]}
                diagnostic_name = filename.replace('.mp4', '_diagnostics.json')
                (output_dir / diagnostic_name).write_text(json.dumps(
                    {'experiment': info, 'steps': diagnostics}, indent=2), encoding='utf-8')
                experiments.append(dict(info, diagnostics=diagnostic_name))
        pipe.model.cpu()
        del snapshots, initial
        torch.cuda.empty_cache()
        pipe.vae.model.to(device)
        count, height, width, _ = shape
        for filename, latent in outputs.items():
            decoded = pipe.vae.decode([latent.to(device)])[0][:, :count, :height, :width]
            if tuple(decoded.shape) != (3, count, height, width) or not torch.isfinite(decoded).all():
                raise RuntimeError(f'Invalid decoded output: {filename}')
            save_decoded(output_dir / filename, decoded, fps)
            del decoded
            torch.cuda.empty_cache()
    metadata = {
        'experiment': 'extended all-layer self-attention output exchange',
        'experiment_set': getattr(args, 'experiment_set', 'all'),
        'inputs': {label: str(path) for label, path in paths.items()},
        'checkpoint': str(checkpoint), 'seed': 42, 'shared_noise': True,
        'initialization': '(1-sigma)*E(A or B) + sigma*same_noise; no residual injection',
        'solver': 'UniPC', 'full_schedule_steps': 50, 'start_schedule_index': 25,
        'actual_updates': 25, 'initial_sigma': sigma, 'shift': cfg.sample_shift,
        'guide_scale': cfg.sample_guide_scale, 'prompt': '',
        'negative_prompt': pipe.sample_neg_prompt,
        'layers_0based': list(range(30)),
        'hook': 'block.self_attn output after self_attn.o, before recipient gate/residual addition',
        'cfg': 'conditional donor -> conditional recipient; unconditional -> unconditional',
        'donor': 'unmodified baseline snapshot at the same pre-update timestep',
        'recipient_solver_history': 'retained; only internal attention output is exchanged',
        'diagnostics': {
            'conditional_prediction/unconditional_prediction/cfg_prediction':
                'swapped versus normal prediction on the same current recipient latent',
            'local_next_latent': 'swapped versus no-swap next latent from identical input and cloned pre-update solver history',
            'local_update': 'same comparison after subtracting the current latent; relative_l2 uses the normal update as reference',
            'trajectory_vs_recipient_baseline': 'post-update latent versus unmodified recipient baseline at the same update, including after intervention ends',
            'relative_l2': 'RMSE(delta) / max(RMS(reference), 1e-12); near-zero references can produce large ratios',
        },
        'self_swap_checks': controls, 'experiments': experiments,
        'fps': fps, 'video_shape_THWC': list(shape),
        'interpretation': 'Intervention sensitivity at this module; not proof of a unique layout-determining timestep.'
    }
    (output_dir / 'attention_extended_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Attention swap experiments complete: {output_dir}', flush=True)


def run_pipeline(model_path, video_tag, experiment_set='all'):
    import torch

    checkpoint = resolve_checkpoint(model_path)
    if not torch.cuda.is_available():
        raise ValueError("The attention exchange experiment requires a CUDA GPU.")
    video_dir = DATA_DIR / "video"
    print("[1/2] Resize video and create the first-frame reference", flush=True)
    prepare_video(
        video_dir / f"{video_tag}.mp4",
        video_dir / f"{video_tag}_lowResolution.mp4",
        video_dir / f"{video_tag}_first_frame.mp4",
    )
    print(f"[2/2] Run A/B baselines and {experiment_set} attention exchanges with diagnostics", flush=True)
    run_attention_swap(argparse.Namespace(model_path=str(checkpoint), video_tag=video_tag,
                                          experiment_set=experiment_set))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True,
                        help="Complete Wan2.2-TI2V-5B directory including Wan2.2_VAE.pth")
    parser.add_argument("--video_tag", required=True, type=file_name)
    parser.add_argument('--experiment_set', choices=('all', 'single', 'window', 'full'),
                        default='all', help='Default: all 22 exchanges plus two baselines.')
    args = parser.parse_args()
    run_pipeline(args.model_path, args.video_tag, args.experiment_set)


if __name__ == "__main__":
    main()
