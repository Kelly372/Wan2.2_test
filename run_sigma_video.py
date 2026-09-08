"""Q2: compare source-video denoising at different actual starting sigmas."""

import argparse
import json
import math

import numpy as np

from run_diff_video import (
    DATA_DIR, denoise, file_name, prepare_tensor, prepare_video,
    read_rgb_video, resolve_checkpoint, save_decoded,
)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be a positive integer.')
    return value


def parse_args(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--video_tag', required=True, type=file_name)
    parser.add_argument('--inference_step', type=positive_int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt', default='')
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0 or args.device_id < 0:
        parser.error('seed and device_id must be nonnegative for reproducible comparisons.')
    return args


def prepare_input(video_tag):
    """Rebuild from the source so stale resized videos cannot enter comparisons."""
    video_dir = DATA_DIR / 'video'
    source = video_dir / f'{video_tag}_lowResolution.mp4'
    prepare_video(video_dir / f'{video_tag}.mp4', source,
                  video_dir / f'{video_tag}_first_frame.mp4')
    output_dir = video_dir / video_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    return source, output_dir


def sigma_grid(sigma, steps, shift):
    """Invert the scheduler's shift before spacing its unshifted sigma grid."""
    if not math.isfinite(sigma) or not 0 < sigma < 1:
        raise ValueError('Starting sigma must be finite and in (0, 1).')
    if steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError('Steps and shift must be positive.')
    unshifted_start = sigma / (shift - (shift - 1) * sigma)
    return np.linspace(unshifted_start, 0, steps + 1)[:-1]


def make_scheduler(factory, sigma, steps, device, shift, num_train_timesteps):
    scheduler = factory(num_train_timesteps=num_train_timesteps, shift=1,
                        use_dynamic_shifting=False)
    if sigma is None:
        # Preserve the exact Q1 schedule, including its integer timestep rounding.
        scheduler.set_timesteps(50, device=device, shift=shift)
        if steps == 25:
            scheduler.set_begin_index(25)
            return scheduler
        sigma = float(scheduler.sigmas[25])
    scheduler.set_timesteps(steps, device=device,
                            sigmas=sigma_grid(sigma, steps, shift), shift=shift)
    scheduler.set_begin_index(0)
    return scheduler


def schedule_metadata(scheduler):
    start = scheduler.begin_index
    return {'start_sigma': float(scheduler.sigmas[start]),
            'steps': len(scheduler.timesteps[start:]),
            'timesteps': scheduler.timesteps[start:].tolist(),
            'sigmas': scheduler.sigmas[start:].tolist()}


def run(args):
    import torch
    from wan import WanTI2V
    from wan.configs import WAN_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    checkpoint = resolve_checkpoint(args.model_path)
    if not torch.cuda.is_available():
        raise ValueError('Generation requires CUDA-enabled PyTorch and a CUDA GPU.')
    torch.cuda.set_device(args.device_id)
    source_path, output_dir = prepare_input(args.video_tag)
    frames, fps = read_rgb_video(source_path)
    count, height, width, _ = frames.shape
    cfg = WAN_CONFIGS['ti2v-5B']
    pipe = WanTI2V(cfg, str(checkpoint), device_id=args.device_id, t5_cpu=True,
                   init_on_cpu=True, convert_model_dtype=True)
    metadata = {'experiment': 'Q2', 'checkpoint': str(checkpoint),
                'source_video': str(source_path), 'shape': [count, height, width],
                'fps': fps, 'seed': args.seed, 'prompt': args.prompt,
                'negative_prompt': pipe.sample_neg_prompt,
                'solver': 'unipc', 'shift': cfg.sample_shift,
                'guide_scale': cfg.sample_guide_scale,
                'initialization': '(1-sigma)*E(A) + sigma*noise; no residual or clean anchor',
                'branches': {}}
    with torch.inference_mode():
        source_tensor = prepare_tensor(frames, pipe.vae_stride[1] * pipe.patch_size[1]).to(pipe.device)
        source_latent = pipe.vae.encode([source_tensor])[0]
        del frames, source_tensor
        if not torch.isfinite(source_latent).all():
            raise RuntimeError('VAE produced non-finite latent values.')
        pipe.vae.model.cpu()
        torch.cuda.empty_cache()
        context = [v.to(pipe.device) for v in pipe.text_encoder([args.prompt], torch.device('cpu'))]
        context_null = [v.to(pipe.device) for v in pipe.text_encoder([pipe.sample_neg_prompt], torch.device('cpu'))]
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        noise = torch.randn(source_latent.shape, device=pipe.device,
                            dtype=torch.float32, generator=generator)
        for label, requested_sigma in (('0.2', 0.2), ('0.4', 0.4), ('0.6', 0.6), ('reference', None)):
            scheduler = make_scheduler(FlowUniPCMultistepScheduler, requested_sigma,
                                       args.inference_step, pipe.device,
                                       cfg.sample_shift, cfg.num_train_timesteps)
            info = schedule_metadata(scheduler)
            sigma = info['start_sigma']
            initial = (1 - sigma) * source_latent + sigma * noise
            print(f'Q2 sigma={sigma:.8f}, {info["steps"]} updates', flush=True)
            pipe.model.to(pipe.device)
            result = denoise(pipe, scheduler, initial,
                             scheduler.timesteps[scheduler.begin_index:],
                             context, context_null, cfg.sample_guide_scale)
            del initial, scheduler
            pipe.model.cpu()
            torch.cuda.empty_cache()
            pipe.vae.model.to(pipe.device)
            decoded = pipe.vae.decode([result])[0][:, :count, :height, :width]
            if tuple(decoded.shape) != (3, count, height, width) or not torch.isfinite(decoded).all():
                raise RuntimeError('Invalid decoded shape or values.')
            filename = f'sigma_{label}.mp4'
            save_decoded(output_dir / filename, decoded, fps)
            del result, decoded
            pipe.vae.model.cpu()
            torch.cuda.empty_cache()
            metadata['branches'][filename] = dict(info, requested_sigma=requested_sigma)
            (output_dir / 'sigma_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Done: {output_dir}')


if __name__ == '__main__':
    run(parse_args())
