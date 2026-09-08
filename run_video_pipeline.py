"""Complete low-resolution residual video generation pipeline.

Usage: python run_video_pipeline.py --model_path /path/to/Wan2.2-TI2V-5B --video_tag clip
Requires the repository's full inference dependencies and a CUDA GPU.
This file contains all three processing stages; it does not import the earlier scripts.
"""

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET_SIZE = (672, 384)



def file_name(value):
    """Accept a filename, never a relative or absolute directory path."""
    if not value.strip() or value in {'.', '..'} or any((character in value for character in '/\\:*?"<>|')):
        raise argparse.ArgumentTypeError('Please provide a filename without a path.')
    return value


def read_video(path):
    if not path.is_file():
        raise FileNotFoundError(f'Video does not exist: {path}')
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid video frame rate: {fps}')
        ok, first_frame = capture.read()
        if not ok:
            raise ValueError(f'Cannot read the first frame: {path}')
        frame_count = 1
        while capture.grab():
            frame_count += 1
        return (first_frame, frame_count, fps)
    finally:
        capture.release()


def write_rgb_video(path, frames, frame_count, fps, size):
    """Stream uint8 RGB frames to a verified H.264/yuv420p MP4."""
    import imageio_ffmpeg
    import numpy as np
    width, height = size
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f'H.264/yuv420p requires positive even dimensions: {size}')
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Frame count and FPS must be positive.')
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.mp4', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    writer = None
    try:
        writer = imageio_ffmpeg.write_frames(str(temporary_path), (width, height), fps=fps, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1, quality=None, input_params=['-r', format(fps, '.15g')], output_params=['-crf', '18', '-preset', 'medium', '-movflags', '+faststart'])
        writer.send(None)
        written = 0
        for frame in frames:
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError('Each frame must be a uint8 RGB array matching the video size.')
            writer.send(np.ascontiguousarray(frame))
            written += 1
        writer.close()
        writer = None
        if written != frame_count:
            raise RuntimeError(f'Incorrect input frame count: {written}/{frame_count}')
        first, actual_count, actual_fps = read_video(temporary_path)
        if actual_count != frame_count or first.shape[:2] != (height, width) or (not math.isclose(actual_fps, fps, rel_tol=0.0001, abs_tol=0.001)):
            raise RuntimeError('Encoded video frame count, dimensions or FPS do not match.')
        temporary_path.replace(path)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            temporary_path.unlink(missing_ok=True)
    print(f'Saved: {path} ({frame_count} frames, {fps:g} FPS)')


def write_still_video(path, frame, frame_count, fps):
    from itertools import repeat
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    write_rgb_video(path, repeat(rgb, frame_count), frame_count, fps, (width, height))


def prepare_video(source_path, low_resolution_path, first_frame_path):
    first, frame_count, fps = read_video(source_path)
    height, width = first.shape[:2]
    print(f'Input: F={frame_count}, H={height}, W={width}, FPS={fps:g}')

    def resized_frames():
        capture = cv2.VideoCapture(str(source_path))
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {source_path}')
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError('Input video dimensions change between frames.')
                resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                yield cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
    write_rgb_video(low_resolution_path, resized_frames(), frame_count, fps, TARGET_SIZE)
    first, actual_count, actual_fps = read_video(low_resolution_path)
    write_still_video(first_frame_path, first, actual_count, actual_fps)


def read_rgb_video(path):
    import cv2
    import numpy as np
    if not path.is_file():
        raise FileNotFoundError(f'Missing input video: {path}')
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid FPS in {path}: {fps}')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f'No readable frames: {path}')
    return (np.stack(frames), fps)


def prepare_tensor(frames, stride):
    import torch
    import torch.nn.functional as functional
    video = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    video = video.div_(127.5).sub_(1.0)
    _, count, height, width = video.shape
    padding = (0, -width % stride, 0, -height % stride, 0, -(count - 1) % 4)
    return functional.pad(video.unsqueeze(0), padding, mode='replicate')[0]


def load_vae(checkpoint, version, device, dtype=None):
    import torch
    suffix = version.replace('.', '_')
    module_path = Path(__file__).resolve().parent / 'wan' / 'modules' / f'vae{suffix}.py'
    spec = importlib.util.spec_from_file_location(f'standalone_vae{suffix}', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vae_class = getattr(module, f'Wan{suffix}_VAE')
    return vae_class(vae_pth=str(checkpoint), device=device, dtype=torch.float32 if dtype is None else dtype)


def save_latent(path, tensor):
    import torch
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        torch.save(tensor, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_decoded(path, video, fps):
    _, count, height, width = video.shape

    def rgb_frames():
        for index in range(count):
            frame = video[:, index].clamp(-1, 1).add(1).mul(127.5)
            yield frame.round().byte().permute(1, 2, 0).cpu().numpy()
    write_rgb_video(path, rgb_frames(), count, fps, (width, height))


def run_vae_diff(args):
    import torch
    checkpoint = Path(args.vae_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing VAE checkpoint: {checkpoint}')
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA is unavailable. Use --device cpu or install CUDA-enabled PyTorch.')
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    dtype = getattr(torch, args.vae_dtype)
    if device.type != 'cuda' and dtype != torch.float32:
        raise ValueError('Use --vae_dtype float32 for CPU execution.')
    if device.type == 'cuda' and dtype == torch.bfloat16 and (not torch.cuda.is_bf16_supported()):
        raise ValueError('This GPU does not support bfloat16; use float32 or float16.')
    print(f'PyTorch={torch.__version__}, CUDA={torch.version.cuda}, cuDNN={torch.backends.cudnn.version()}, cuDNN enabled={torch.backends.cudnn.enabled}, VAE dtype={args.vae_dtype}, device={device}', flush=True)
    video_dir = DATA_DIR / 'video'
    paths = {'A': video_dir / f'{args.video_tag}_lowResolution.mp4', 'B': video_dir / f'{args.video_tag}_first_frame.mp4'}
    reference_shape = None
    fps = None
    for label, path in paths.items():
        frames, current_fps = read_rgb_video(path)
        if reference_shape is None:
            reference_shape, fps = (frames.shape, current_fps)
        if frames.shape != reference_shape or not math.isclose(current_fps, fps, rel_tol=0.0001, abs_tol=0.001):
            raise ValueError(f'{label}: shape/FPS {frames.shape}/{current_fps} differs from A: {reference_shape}/{fps}. Regenerate B with prepare_videos.py.')
        del frames
    count, height, width, _ = reference_shape
    if height % 2 or width % 2:
        raise ValueError('MP4 output requires even input width and height.')
    output_dir = video_dir / args.video_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    vae = load_vae(checkpoint, args.vae_type, device, dtype=dtype)
    stride = 16 if args.vae_type == '2.2' else 8
    latents = {}
    with torch.inference_mode():
        for label, path in paths.items():
            print(f'Encoding {label}: {path}', flush=True)
            frames, _ = read_rgb_video(path)
            video = prepare_tensor(frames, stride).to(device)
            del frames
            latent = vae.encode([video])[0].cpu().contiguous()
            del video
            if not torch.isfinite(latent).all():
                raise RuntimeError(f'Non-finite latent for {label}')
            latents[label] = latent
            save_latent(output_dir / f'{path.stem}.pt', latent)
        name = 'A-B'
        print(f'Decoding {name}', flush=True)
        difference = latents['A'] - latents['B']
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(device)
            print(f'GPU={torch.cuda.get_device_name(device)}, latent={tuple(difference.shape)}, free VRAM={free / 2 ** 30:.2f}/{total / 2 ** 30:.2f} GiB', flush=True)
        decoded = vae.decode([difference.to(device)])[0]
        if decoded.shape[1] < count or decoded.shape[2] < height or decoded.shape[3] < width:
            raise RuntimeError(f'Decoded {name} is smaller than the original video.')
        decoded = decoded[:, :count, :height, :width]
        if not torch.isfinite(decoded).all():
            raise RuntimeError(f'Non-finite decoded video for {name}')
        save_decoded(output_dir / f'{name}.mp4', decoded, fps)
        del difference, decoded
    metadata = {'inputs': {key: str(path) for key, path in paths.items()}, 'vae_type': args.vae_type, 'vae_checkpoint': str(checkpoint), 'vae_dtype': args.vae_dtype, 'cudnn_enabled': torch.backends.cudnn.enabled, 'fps': fps, 'original_shape_THWC': reference_shape, 'latent_shape_CTHW': list(latents['A'].shape), 'padding': 'Repeat last frame to 4n+1; replicate right/bottom edges to VAE stride.', 'operation': 'decode(encode(A) - encode(B)) using Wan normalized latents', 'output': 'Crop to original F/H/W; map decoder [-1,1] to [0,255]; no audio.', 'video_encoding': 'H.264 (libx264), yuv420p, CRF 18, faststart'}
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Done: {output_dir}')


def normalize_dark_residual(frames, baseline=127.5):
    """Keep dark-side magnitude; white background, strongest dark residual black.

    Use one scalar for the entire clip, preserving relative temporal amplitudes.
    This is intensity selection, not a spatial segmentation mask.
    """
    if not 0 < baseline <= 255:
        raise ValueError('Residual baseline must be in (0, 255].')
    magnitude = np.maximum(baseline - frames.astype(np.float32), 0)
    peak = float(magnitude.max())
    if peak == 0:
        return (np.full(frames.shape, 255, dtype=np.uint8), peak)
    normalized = np.rint(255 * (1 - magnitude / peak)).astype(np.uint8)
    return (normalized, peak)


def mix_initial_latent(noise, residual, source=None, sigma=None):
    """Add the residual exactly once, without concatenation or amplitude scaling."""
    if noise.shape != residual.shape or (source is not None and source.shape != noise.shape):
        raise ValueError('All latent shapes must match for elementwise addition.')
    if source is None:
        return noise + residual
    if sigma is None or not 0 <= sigma <= 1:
        raise ValueError('Conditioned initialization needs sigma in [0, 1].')
    return (1 - sigma) * source + sigma * noise + residual


def resolve_checkpoint(value):
    directory = Path(value).expanduser().resolve()
    required = ['Wan2.2_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth', 'google/umt5-xxl', 'config.json']
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f'Incomplete Wan2.2-TI2V-5B directory: {directory}. Missing: {missing}. Set --model_path to the complete model directory.')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config.get('in_dim') != 48 or config.get('out_dim') != 48:
        raise ValueError('This experiment requires the TI2V-5B model with 48-channel latents.')
    return directory


def denoise(pipe, scheduler, initial, timesteps, context, context_null, guide_scale):
    import torch
    from tqdm import tqdm
    latent = initial
    _, time, height, width = latent.shape
    seq_len = time * (height // pipe.patch_size[1]) * (width // pipe.patch_size[2])
    with torch.amp.autocast('cuda', dtype=pipe.param_dtype):
        for t in tqdm(timesteps, desc='Denoising'):
            timestep = t.reshape(1).expand(1, seq_len)
            conditional = pipe.model([latent], t=timestep, context=context, seq_len=seq_len)[0]
            unconditional = pipe.model([latent], t=timestep, context=context_null, seq_len=seq_len)[0]
            prediction = unconditional + guide_scale * (conditional - unconditional)
            latent = scheduler.step(prediction.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)
    if not torch.isfinite(latent).all():
        raise RuntimeError('Denoising produced non-finite latent values.')
    return latent


def run_generation(args):
    import torch
    from wan import WanTI2V
    from wan.configs import WAN_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    if not torch.cuda.is_available():
        raise ValueError('Generation requires a CUDA GPU and CUDA-enabled PyTorch.')
    if args.inference_step < 1 or not 0 < args.strength <= 1:
        raise ValueError('inference_step must be positive and strength must be in (0, 1].')
    checkpoint = resolve_checkpoint(args.model_path)
    video_dir = DATA_DIR / 'video'
    output_dir = video_dir / args.video_tag
    source_path = video_dir / f'{args.video_tag}_lowResolution.mp4'
    original, fps = read_rgb_video(source_path)
    residual, residual_fps = read_rgb_video(output_dir / 'A-B.mp4')
    if original.shape != residual.shape or not math.isclose(fps, residual_fps, rel_tol=0.0001, abs_tol=0.001):
        raise ValueError('Original and A-B.mp4 must have matching frame count, dimensions and FPS.')
    count, height, width, _ = original.shape
    if height % 2 or width % 2:
        raise ValueError('H.264/yuv420p requires even video dimensions.')
    normalized, peak = normalize_dark_residual(residual, args.baseline)
    del residual
    if peak == 0:
        raise ValueError('No dark-side residual was found; check A-B.mp4 and --baseline.')
    write_rgb_video(output_dir / 'diff_normalized.mp4', normalized, count, fps, (width, height))
    cfg = WAN_CONFIGS['ti2v-5B']
    torch.cuda.set_device(args.device_id)
    pipe = WanTI2V(cfg, str(checkpoint), device_id=args.device_id, t5_cpu=True, init_on_cpu=True, convert_model_dtype=True)
    alignment = pipe.vae_stride[1] * pipe.patch_size[1]
    with torch.inference_mode():
        original_tensor = prepare_tensor(original, alignment).to(pipe.device)
        source_latent = pipe.vae.encode([original_tensor])[0].cpu()
        del original_tensor, original
        residual_tensor = prepare_tensor(normalized, alignment).to(pipe.device)
        residual_latent = pipe.vae.encode([residual_tensor])[0].cpu()
        del residual_tensor, normalized
        if not torch.isfinite(source_latent).all() or not torch.isfinite(residual_latent).all():
            raise RuntimeError('VAE produced non-finite latent values.')
        save_latent(output_dir / 'diff_normalized.pt', residual_latent)
        pipe.vae.model.cpu()
        torch.cuda.empty_cache()
        context = [v.to(pipe.device) for v in pipe.text_encoder([args.prompt], torch.device('cpu'))]
        context_null = [v.to(pipe.device) for v in pipe.text_encoder([pipe.sample_neg_prompt], torch.device('cpu'))]
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        noise = torch.randn(source_latent.shape, device=pipe.device, dtype=torch.float32, generator=generator)
        source_latent = source_latent.to(pipe.device)
        residual_latent = residual_latent.to(pipe.device)
        branch_info = {}
        for branch in ('condition', 'random'):
            scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            scheduler.set_timesteps(args.inference_step, device=pipe.device, shift=cfg.sample_shift)
            start = 0 if branch == 'random' else args.inference_step - max(1, int(args.inference_step * args.strength))
            scheduler.set_begin_index(start)
            sigma = float(scheduler.sigmas[start])
            initial = mix_initial_latent(noise, residual_latent, source=source_latent if branch == 'condition' else None, sigma=sigma if branch == 'condition' else None)
            pipe.model.to(pipe.device)
            print(f'{branch}: {len(scheduler.timesteps[start:])} steps, start sigma={sigma:g}', flush=True)
            result = denoise(pipe, scheduler, initial, scheduler.timesteps[start:], context, context_null, cfg.sample_guide_scale)
            del initial
            pipe.model.cpu()
            torch.cuda.empty_cache()
            pipe.vae.model.to(pipe.device)
            decoded = pipe.vae.decode([result])[0][:, :count, :height, :width]
            if tuple(decoded.shape) != (3, count, height, width) or not torch.isfinite(decoded).all():
                raise RuntimeError('Invalid decoded output shape or values.')
            save_decoded(output_dir / f'{branch}_noise_with_diff.mp4', decoded, fps)
            del result, decoded
            pipe.vae.model.cpu()
            torch.cuda.empty_cache()
            branch_info[branch] = {'start_index': start, 'sigma': sigma, 'steps': len(scheduler.timesteps[start:])}
    metadata = {'model': 'ti2v-5B', 'checkpoint': str(checkpoint), 'seed': args.seed, 'source_video': str(source_path), 'prompt': args.prompt, 'negative_prompt': pipe.sample_neg_prompt, 'solver': 'unipc', 'inference_step': args.inference_step, 'shift': cfg.sample_shift, 'guide_scale': cfg.sample_guide_scale, 'strength': args.strength, 'branches': branch_info, 'normalization': 'u=max(baseline-RGB,0); image=255*(1-u/global_max(u))', 'baseline': args.baseline, 'dark_peak': peak, 'injection': 'once before denoising; elementwise addition with weight 1', 'random': 'noise + E(normalized_diff)', 'condition': '(1-sigma)*E(original) + sigma*noise + E(normalized_diff)', 'latent_shape': list(source_latent.shape), 'fps': fps, 'original_shape': [count, height, width, 3]}
    (output_dir / 'generation_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Done: {output_dir}')


def run_pipeline(model_path, video_tag):
    import torch

    checkpoint = resolve_checkpoint(model_path)
    if not torch.cuda.is_available():
        raise ValueError("The complete pipeline requires a CUDA GPU.")
    video_dir = DATA_DIR / "video"
    print("[1/3] Resize video and create the first-frame reference", flush=True)
    prepare_video(
        video_dir / f"{video_tag}.mp4",
        video_dir / f"{video_tag}_lowResolution.mp4",
        video_dir / f"{video_tag}_first_frame.mp4",
    )
    print("[2/3] Encode A/B and decode the latent difference", flush=True)
    run_vae_diff(argparse.Namespace(
        video_tag=video_tag, vae_checkpoint=str(checkpoint / "Wan2.2_VAE.pth"),
        vae_type="2.2", device="cuda:0", vae_dtype="float32", disable_cudnn=False,
    ))
    # Stage two's VAE is out of scope; release its allocations before loading
    # the full generation pipeline (T5, DiT and VAE).
    gc.collect()
    torch.cuda.empty_cache()
    print("[3/3] Normalize the dark residual and generate both videos", flush=True)
    run_generation(argparse.Namespace(
        video_tag=video_tag, model_path=str(checkpoint), inference_step=50,
        strength=0.5, seed=42, prompt="", baseline=127.5, device_id=0,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True,
                        help="Complete Wan2.2-TI2V-5B directory, including Wan2.2_VAE.pth")
    parser.add_argument("--video_tag", required=True, type=file_name)
    args = parser.parse_args()
    run_pipeline(args.model_path, args.video_tag)


if __name__ == "__main__":
    main()
