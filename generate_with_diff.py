"""Generate videos by adding a one-sided residual latent before denoising."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from prepare_videos import DATA_DIR, file_name, write_rgb_video
from vae_video_diff import prepare_tensor, read_rgb_video, save_decoded, save_latent


ROOT = Path(__file__).resolve().parent


def normalize_dark_residual(frames, baseline=127.5):
    """Keep dark-side magnitude; white background, strongest dark residual black.

    Use one scalar for the entire clip, preserving relative temporal amplitudes.
    This is intensity selection, not a spatial segmentation mask.
    """
    if not 0 < baseline <= 255:
        raise ValueError("Residual baseline must be in (0, 255].")
    magnitude = np.maximum(baseline - frames.astype(np.float32), 0)
    peak = float(magnitude.max())
    if peak == 0:
        return np.full(frames.shape, 255, dtype=np.uint8), peak
    normalized = np.rint(255 * (1 - magnitude / peak)).astype(np.uint8)
    return normalized, peak


def mix_initial_latent(noise, residual, source=None, sigma=None):
    """Add the residual exactly once, without concatenation or amplitude scaling."""
    if noise.shape != residual.shape or (source is not None and source.shape != noise.shape):
        raise ValueError("All latent shapes must match for elementwise addition.")
    if source is None:
        return noise + residual
    if sigma is None or not 0 <= sigma <= 1:
        raise ValueError("Conditioned initialization needs sigma in [0, 1].")
    return (1 - sigma) * source + sigma * noise + residual


def resolve_checkpoint(value):
    directory = Path(value).expanduser().resolve()
    required = ["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl", "config.json"]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete Wan2.2-TI2V-5B directory: {directory}. Missing: {missing}. "
            "Set WAN_CKPT_DIR or --ckpt_dir to the complete model directory."
        )
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("in_dim") != 48 or config.get("out_dim") != 48:
        raise ValueError("This experiment requires the TI2V-5B model with 48-channel latents.")
    return directory


def denoise(pipe, scheduler, initial, timesteps, context, context_null, guide_scale):
    import torch
    from tqdm import tqdm

    latent = initial
    _, time, height, width = latent.shape
    seq_len = time * (height // pipe.patch_size[1]) * (width // pipe.patch_size[2])
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype):
        for t in tqdm(timesteps, desc="Denoising"):
            # All tokens are noisy: the official TI2V t2v path uses this same
            # uniform token timestep (there is no image-conditioning mask).
            timestep = t.reshape(1).expand(1, seq_len)
            conditional = pipe.model([latent], t=timestep, context=context, seq_len=seq_len)[0]
            unconditional = pipe.model([latent], t=timestep, context=context_null, seq_len=seq_len)[0]
            prediction = unconditional + guide_scale * (conditional - unconditional)
            latent = scheduler.step(
                prediction.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
    if not torch.isfinite(latent).all():
        raise RuntimeError("Denoising produced non-finite latent values.")
    return latent


def run(args):
    import torch
    from wan import WanTI2V
    from wan.configs import WAN_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    if not torch.cuda.is_available():
        raise ValueError("Generation requires a CUDA GPU and CUDA-enabled PyTorch.")
    if args.inference_step < 1 or not 0 < args.strength <= 1:
        raise ValueError("inference_step must be positive and strength must be in (0, 1].")
    checkpoint = resolve_checkpoint(args.ckpt_dir)
    video_dir = DATA_DIR / "video"
    output_dir = video_dir / args.video_tag
    original, fps = read_rgb_video(video_dir / f"{args.video_tag}.mp4")
    residual, residual_fps = read_rgb_video(output_dir / "A-B.mp4")
    if original.shape != residual.shape or not math.isclose(fps, residual_fps, rel_tol=1e-4, abs_tol=1e-3):
        raise ValueError("Original and A-B.mp4 must have matching frame count, dimensions and FPS.")
    count, height, width, _ = original.shape
    if height % 2 or width % 2:
        raise ValueError("H.264/yuv420p requires even video dimensions.")
    normalized, peak = normalize_dark_residual(residual, args.baseline)
    del residual
    if peak == 0:
        raise ValueError("No dark-side residual was found; check A-B.mp4 and --baseline.")
    write_rgb_video(output_dir / "diff_normalized.mp4", normalized, count, fps, (width, height))

    cfg = WAN_CONFIGS["ti2v-5B"]
    torch.cuda.set_device(args.device_id)
    pipe = WanTI2V(cfg, str(checkpoint), device_id=args.device_id,
                   t5_cpu=True, init_on_cpu=True, convert_model_dtype=True)
    # The VAE stride is 16; the DiT also needs 2x2 latent patches, hence 32.
    alignment = pipe.vae_stride[1] * pipe.patch_size[1]
    with torch.inference_mode():
        original_tensor = prepare_tensor(original, alignment).to(pipe.device)
        source_latent = pipe.vae.encode([original_tensor])[0].cpu()
        del original_tensor, original
        residual_tensor = prepare_tensor(normalized, alignment).to(pipe.device)
        residual_latent = pipe.vae.encode([residual_tensor])[0].cpu()
        del residual_tensor, normalized
        if not torch.isfinite(source_latent).all() or not torch.isfinite(residual_latent).all():
            raise RuntimeError("VAE produced non-finite latent values.")
        # Encode pre-compression normalized pixels; MP4 is a visualization only.
        save_latent(output_dir / "diff_normalized.pt", residual_latent)
        pipe.vae.model.cpu()
        torch.cuda.empty_cache()
        context = [v.to(pipe.device) for v in pipe.text_encoder([args.prompt], torch.device("cpu"))]
        context_null = [v.to(pipe.device) for v in pipe.text_encoder([pipe.sample_neg_prompt], torch.device("cpu"))]
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        noise = torch.randn(source_latent.shape, device=pipe.device, dtype=torch.float32, generator=generator)
        source_latent = source_latent.to(pipe.device)
        residual_latent = residual_latent.to(pipe.device)
        branch_info = {}
        for branch in ("condition", "random"):
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False
            )
            scheduler.set_timesteps(args.inference_step, device=pipe.device, shift=cfg.sample_shift)
            start = 0 if branch == "random" else args.inference_step - max(1, int(args.inference_step * args.strength))
            scheduler.set_begin_index(start)
            sigma = float(scheduler.sigmas[start])
            initial = mix_initial_latent(
                noise, residual_latent,
                source=source_latent if branch == "condition" else None,
                sigma=sigma if branch == "condition" else None,
            )
            pipe.model.to(pipe.device)
            print(f"{branch}: {len(scheduler.timesteps[start:])} steps, start sigma={sigma:g}", flush=True)
            result = denoise(pipe, scheduler, initial, scheduler.timesteps[start:], context, context_null, cfg.sample_guide_scale)
            del initial
            pipe.model.cpu()
            torch.cuda.empty_cache()
            pipe.vae.model.to(pipe.device)
            decoded = pipe.vae.decode([result])[0][:, :count, :height, :width]
            if tuple(decoded.shape) != (3, count, height, width) or not torch.isfinite(decoded).all():
                raise RuntimeError("Invalid decoded output shape or values.")
            save_decoded(output_dir / f"{branch}_noise_with_diff.mp4", decoded, fps)
            del result, decoded
            pipe.vae.model.cpu()
            torch.cuda.empty_cache()
            branch_info[branch] = {"start_index": start, "sigma": sigma, "steps": len(scheduler.timesteps[start:])}

    metadata = {
        "model": "ti2v-5B", "checkpoint": str(checkpoint), "seed": args.seed,
        "prompt": args.prompt, "negative_prompt": pipe.sample_neg_prompt,
        "solver": "unipc", "inference_step": args.inference_step,
        "shift": cfg.sample_shift, "guide_scale": cfg.sample_guide_scale,
        "strength": args.strength, "branches": branch_info,
        "normalization": "u=max(baseline-RGB,0); image=255*(1-u/global_max(u))",
        "baseline": args.baseline, "dark_peak": peak,
        "injection": "once before denoising; elementwise addition with weight 1",
        "random": "noise + E(normalized_diff)",
        "condition": "(1-sigma)*E(original) + sigma*noise + E(normalized_diff)",
        "latent_shape": list(source_latent.shape), "fps": fps,
        "original_shape": [count, height, width, 3],
    }
    (output_dir / "generation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_tag", required=True, type=file_name)
    parser.add_argument("--ckpt_dir", default=os.environ.get("WAN_CKPT_DIR", str(ROOT / "Wan2.2-TI2V-5B")))
    parser.add_argument("--inference_step", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.5, help="Fraction of schedule used by the original-video branch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="", help="Optional shared text prompt; default empty")
    parser.add_argument("--baseline", type=float, default=127.5)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
