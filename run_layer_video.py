"""Compare self-attention layer groups, exchanging at updates 21-25 of 25.

python run_layer_video.py --model_path /path/to/Wan2.2-TI2V-5B --video_tag clip
"""

import argparse
import json
import math

from run_diff_video import (
    DATA_DIR, file_name, prepare_video, prepare_tensor, read_rgb_video,
    resolve_checkpoint, save_decoded,
)
from run_replace_video import attention_trajectory

SWAP_UPDATES = tuple(range(20, 25))  # Zero-based; actual update numbers are 21-25.


def layer_groups(mode='all'):
    groups = [('front', tuple(range(0, 10))),
              ('middle', tuple(range(10, 20))),
              ('back', tuple(range(20, 30)))]
    if mode not in ('all', 'front', 'middle', 'back'):
        raise ValueError('Unknown layer mode.')
    return [(name, indices) for name, indices in groups if mode in ('all', name)]


def layer_experiments(mode='all'):
    return [dict(mode=name, layers=layers, recipient=recipient, donor=donor,
                 filename=f'layer_{name}_{layers[0]:02d}-{layers[-1]:02d}_'
                          f'swap_{recipient}_from_{donor}_updates21-25.mp4')
            for recipient, donor in (('A', 'B'), ('B', 'A'))
            for name, layers in layer_groups(mode)]


def run_layer_swap(args):
    import torch
    from wan import WanTI2V
    from wan.configs import WAN_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    checkpoint = resolve_checkpoint(args.model_path)
    cfg = WAN_CONFIGS['ti2v-5B']
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    output_dir = DATA_DIR / 'video' / args.video_tag / 'attention_layers'
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {'A': DATA_DIR / 'video' / f'{args.video_tag}_lowResolution.mp4',
             'B': DATA_DIR / 'video' / f'{args.video_tag}_first_frame.mp4'}
    pipe = WanTI2V(cfg, str(checkpoint), device_id=0, t5_cpu=True,
                   init_on_cpu=True, convert_model_dtype=True)
    if len(pipe.model.blocks) != 30:
        raise ValueError('Layer groups require the 30-block Wan TI2V-5B model.')

    def new_scheduler():
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps,
                                                shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(50, device=device, shift=cfg.sample_shift)
        return scheduler

    with torch.inference_mode():
        encoded = {}
        shape, fps = None, None
        for label, path in paths.items():
            frames, current_fps = read_rgb_video(path)
            if shape is None:
                shape, fps = frames.shape, current_fps
            elif frames.shape != shape or not math.isclose(fps, current_fps, rel_tol=1e-4, abs_tol=1e-3):
                raise ValueError('A and B must have matching shape and FPS.')
            video = prepare_tensor(frames, 32).to(device)
            del frames
            encoded[label] = pipe.vae.encode([video])[0].cpu()
            del video
            if not torch.isfinite(encoded[label]).all():
                raise RuntimeError(f'Non-finite VAE latent: {label}.')
        pipe.vae.model.cpu()
        torch.cuda.empty_cache()
        context = [v.to(device) for v in pipe.text_encoder([''], torch.device('cpu'))]
        context_null = [v.to(device) for v in pipe.text_encoder([pipe.sample_neg_prompt], torch.device('cpu'))]
        schedule = new_scheduler()
        sigma = float(schedule.sigmas[25])
        generator = torch.Generator(device=device).manual_seed(42)
        noise = torch.randn(encoded['A'].shape, device=device, dtype=torch.float32,
                            generator=generator).cpu()
        initial = {label: (1 - sigma) * value + sigma * noise for label, value in encoded.items()}
        del encoded, noise
        snapshots, outputs, controls = {}, {}, {}
        pipe.model.to(device)
        for label in ('A', 'B'):
            print(f'Layer baseline Process({label}): 25 updates', flush=True)
            final, snapshots[label], controls[label] = attention_trajectory(
                pipe, new_scheduler(), initial[label], context, context_null,
                cfg.sample_guide_scale, output_dir, capture=True,
                capture_indices=range(26), self_check=True)
            outputs[f'layer_process_{label}_baseline.mp4'] = final

        experiments = []
        for experiment in layer_experiments(args.layer_mode):
            filename = experiment['filename']
            recipient, donor = experiment['recipient'], experiment['donor']
            print(f'{filename}: {experiment["layers"]}, updates 21-25', flush=True)
            diagnostics = []
            final, _, _ = attention_trajectory(
                pipe, new_scheduler(), initial[recipient], context, context_null,
                cfg.sample_guide_scale, output_dir, swap_indices=SWAP_UPDATES,
                layer_indices=experiment['layers'], donor_snapshots=snapshots[donor],
                recipient_snapshots=snapshots[recipient], diagnostics=diagnostics)
            outputs[filename] = final
            info = dict(experiment, layers_0based=experiment['layers'],
                        updates_1based=[i + 1 for i in SWAP_UPDATES],
                        schedule_indices=[25 + i for i in SWAP_UPDATES],
                        timesteps=[int(schedule.timesteps[25 + i]) for i in SWAP_UPDATES],
                        sigmas=[float(schedule.sigmas[25 + i]) for i in SWAP_UPDATES])
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
                raise RuntimeError(f'Invalid decoded video: {filename}')
            save_decoded(output_dir / filename, decoded, fps)
            del decoded
            torch.cuda.empty_cache()

    metadata = {
        'experiment': 'self-attention layer ablation at updates 21-25',
        'layer_mode': args.layer_mode, 'checkpoint': str(checkpoint),
        'inputs': {name: str(path) for name, path in paths.items()},
        'seed': 42, 'shared_noise': True, 'prompt': '',
        'negative_prompt': pipe.sample_neg_prompt,
        'initialization': '(1-sigma)*E(A/B) + sigma*same_noise; no residual injection',
        'solver': 'UniPC', 'full_schedule_steps': 50, 'start_schedule_index': 25,
        'actual_updates': 25, 'initial_sigma': sigma, 'shift': cfg.sample_shift,
        'guide_scale': cfg.sample_guide_scale, 'fps': fps, 'video_shape_THWC': list(shape),
        'hook': 'selected block.self_attn output after projection, before gate/residual',
        'cfg': 'conditional -> conditional; unconditional -> unconditional',
        'donor': 'unmodified baseline snapshot at the same pre-update timestep',
        'recipient': 'other layers, latent input and solver history retained',
        'self_swap_checks': controls, 'experiments': experiments,
        'diagnostics': 'same definitions as run_replace_video: predictions and cloned-history single-step update comparisons; all 25 post-update latents compared with recipient baseline',
        'interpretation': 'layer-group intervention sensitivity at a fixed late window; not proof of a unique camera-control layer',
    }
    (output_dir / 'layer_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Done: {output_dir}')


def run_pipeline(model_path, video_tag, layer_mode='all'):
    import torch

    layer_groups(layer_mode)
    checkpoint = resolve_checkpoint(model_path)
    if not torch.cuda.is_available():
        raise ValueError('Layer exchange requires a CUDA GPU and CUDA-enabled PyTorch.')
    video_dir = DATA_DIR / 'video'
    print('[1/2] Resize input and prepare A/B', flush=True)
    prepare_video(video_dir / f'{video_tag}.mp4',
                  video_dir / f'{video_tag}_lowResolution.mp4',
                  video_dir / f'{video_tag}_first_frame.mp4')
    print('[2/2] Run layer-group exchanges at updates 21-25', flush=True)
    run_layer_swap(argparse.Namespace(model_path=str(checkpoint), video_tag=video_tag,
                                      layer_mode=layer_mode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--video_tag', required=True, type=file_name)
    parser.add_argument('--layer_mode', choices=('all', 'front', 'middle', 'back'), default='all',
                        help='Default: all three groups in both directions, plus two baselines.')
    args = parser.parse_args()
    run_pipeline(args.model_path, args.video_tag, args.layer_mode)


if __name__ == '__main__':
    main()
