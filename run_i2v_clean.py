"""Q3: native Wan I2V with a persistent clean first-frame latent condition."""

import json

import cv2

from run_diff_video import TARGET_SIZE, read_video, resolve_checkpoint, save_decoded
from run_sigma_video import parse_args, prepare_input, schedule_metadata


def padded_frame_count(count, temporal_stride=4):
    if count < 1 or temporal_stride < 1:
        raise ValueError('Frame count and temporal stride must be positive.')
    return count + (-(count - 1) % temporal_stride)


def native_branches(pipe, image, count, args, cfg):
    """Yield one native result at a time to avoid holding two decoded videos."""
    common = dict(input_prompt=args.prompt,
                  frame_num=padded_frame_count(count, pipe.vae_stride[0]),
                  shift=cfg.sample_shift, sample_solver='unipc',
                  sampling_steps=args.inference_step, guide_scale=cfg.sample_guide_scale,
                  n_prompt=pipe.sample_neg_prompt, seed=args.seed, offload_model=True)
    yield 'i2v_clean.mp4', pipe.i2v(img=image, max_area=image.width * image.height, **common)
    yield 'i2v_no_anchor.mp4', pipe.t2v(size=(image.width, image.height), **common)


def run(args):
    import torch
    from PIL import Image
    from wan import WanTI2V
    from wan.configs import WAN_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import best_output_size

    checkpoint = resolve_checkpoint(args.model_path)
    if not torch.cuda.is_available():
        raise ValueError('Generation requires CUDA-enabled PyTorch and a CUDA GPU.')
    torch.cuda.set_device(args.device_id)
    source_path, output_dir = prepare_input(args.video_tag)
    first, count, fps = read_video(source_path)
    image = Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
    width, height = image.size
    cfg = WAN_CONFIGS['ti2v-5B']
    pipe = WanTI2V(cfg, str(checkpoint), device_id=args.device_id, t5_cpu=True,
                   init_on_cpu=True, convert_model_dtype=True)
    dw, dh = pipe.patch_size[2] * pipe.vae_stride[2], pipe.patch_size[1] * pipe.vae_stride[1]
    if image.size != TARGET_SIZE or best_output_size(width, height, dw, dh, width * height) != image.size:
        raise ValueError('I2V and its no-anchor control must use exactly the same spatial dimensions.')
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps,
                                            shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(args.inference_step, device=pipe.device, shift=cfg.sample_shift)
    scheduler.set_begin_index(0)
    metadata = {'experiment': 'Q3', 'checkpoint': str(checkpoint),
                'source_video': str(source_path), 'shape': [count, height, width],
                'generated_frames': padded_frame_count(count, pipe.vae_stride[0]),
                'fps': fps, 'seed': args.seed, 'prompt': args.prompt,
                'negative_prompt': pipe.sample_neg_prompt,
                'solver': 'unipc', 'shift': cfg.sample_shift,
                'guide_scale': cfg.sample_guide_scale, 'schedule': schedule_metadata(scheduler),
                'i2v_clean': 'native i2v: first latent slice is clean, token timestep=0, restored after every update; other slices start from noise',
                'i2v_no_anchor': 'native t2v: same Gaussian seed, shape, text and schedule; no image condition',
                'comparison_limit': 'Unlike Q2, no full-video source latent initialization. This does not prescribe the original camera trajectory.',
                'completed_outputs': []}
    del scheduler
    with torch.inference_mode():
        for filename, decoded in native_branches(pipe, image, count, args, cfg):
            decoded = decoded[:, :count, :height, :width]
            if tuple(decoded.shape) != (3, count, height, width) or not torch.isfinite(decoded).all():
                raise RuntimeError('Invalid decoded shape or values.')
            save_decoded(output_dir / filename, decoded, fps)
            del decoded
            torch.cuda.empty_cache()
            metadata['completed_outputs'].append(filename)
            (output_dir / 'i2v_clean_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Done: {output_dir}')


if __name__ == '__main__':
    run(parse_args(__doc__))
