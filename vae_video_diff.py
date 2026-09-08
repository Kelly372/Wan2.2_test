"""Encode original/first-frame videos and decode A-B in Wan latent space."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import tempfile

from prepare_videos import DATA_DIR, file_name


def read_rgb_video(path):
    import cv2
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"Missing input video: {path}")
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS in {path}: {fps}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No readable frames: {path}")
    return np.stack(frames), fps


def prepare_tensor(frames, stride):
    import torch
    import torch.nn.functional as functional

    video = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    video = video.div_(127.5).sub_(1.0)
    _, count, height, width = video.shape
    # Edge padding preserves every source pixel and frame; crop after decoding.
    padding = (0, -width % stride, 0, -height % stride, 0, -(count - 1) % 4)
    return functional.pad(video.unsqueeze(0), padding, mode="replicate")[0]


def load_vae(checkpoint, version, device, dtype=None):
    import torch

    # Load only the standalone VAE, avoiding wan.__init__ and diffusion models.
    suffix = version.replace(".", "_")
    module_path = Path(__file__).resolve().parent / "wan" / "modules" / f"vae{suffix}.py"
    spec = importlib.util.spec_from_file_location(f"standalone_vae{suffix}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vae_class = getattr(module, f"Wan{suffix}_VAE")
    return vae_class(
        vae_pth=str(checkpoint), device=device,
        dtype=torch.float32 if dtype is None else dtype,
    )


def save_latent(path, tensor):
    import torch

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        torch.save(tensor, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_decoded(path, video, fps):
    from prepare_videos import write_rgb_video

    _, count, height, width = video.shape

    def rgb_frames():
        for index in range(count):
            frame = video[:, index].clamp(-1, 1).add(1).mul(127.5)
            yield frame.round().byte().permute(1, 2, 0).cpu().numpy()

    write_rgb_video(path, rgb_frames(), count, fps, (width, height))


def run(args):
    import torch

    checkpoint = Path(args.vae_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing VAE checkpoint: {checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or install CUDA-enabled PyTorch.")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    dtype = getattr(torch, args.vae_dtype)
    if device.type != "cuda" and dtype != torch.float32:
        raise ValueError("Use --vae_dtype float32 for CPU execution.")
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; use float32 or float16.")
    print(
        f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, "
        f"cuDNN={torch.backends.cudnn.version()}, cuDNN enabled={torch.backends.cudnn.enabled}, "
        f"VAE dtype={args.vae_dtype}, device={device}", flush=True,
    )
    video_dir = DATA_DIR / "video"
    paths = {
        "A": video_dir / f"{args.video_tag}.mp4",
        "B": video_dir / f"{args.video_tag}_first_frame.mp4",
    }
    # Validate all inputs before loading the model or writing experiment results.
    reference_shape = None
    fps = None
    for label, path in paths.items():
        frames, current_fps = read_rgb_video(path)
        if reference_shape is None:
            reference_shape, fps = frames.shape, current_fps
        if frames.shape != reference_shape or not math.isclose(
            current_fps, fps, rel_tol=1e-4, abs_tol=1e-3
        ):
            raise ValueError(
                f"{label}: shape/FPS {frames.shape}/{current_fps} differs from "
                f"A: {reference_shape}/{fps}. Regenerate B with prepare_videos.py."
            )
        del frames
    count, height, width, _ = reference_shape
    if height % 2 or width % 2:
        raise ValueError("MP4 output requires even input width and height.")

    output_dir = video_dir / args.video_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    vae = load_vae(checkpoint, args.vae_type, device, dtype=dtype)
    stride = 16 if args.vae_type == "2.2" else 8
    latents = {}
    with torch.inference_mode():
        for label, path in paths.items():
            print(f"Encoding {label}: {path}", flush=True)
            frames, _ = read_rgb_video(path)
            video = prepare_tensor(frames, stride).to(device)
            del frames
            latent = vae.encode([video])[0].cpu().contiguous()
            del video
            if not torch.isfinite(latent).all():
                raise RuntimeError(f"Non-finite latent for {label}")
            latents[label] = latent
            save_latent(output_dir / f"{path.stem}.pt", latent)

        name = "A-B"
        print(f"Decoding {name}", flush=True)
        difference = latents["A"] - latents["B"]
        if device.type == "cuda":
            # Release unused allocator blocks left by encoding. This does not
            # free live tensors or guarantee enough workspace for decoding.
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(device)
            print(
                f"GPU={torch.cuda.get_device_name(device)}, latent={tuple(difference.shape)}, "
                f"free VRAM={free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True,
            )
        decoded = vae.decode([difference.to(device)])[0]
        if decoded.shape[1] < count or decoded.shape[2] < height or decoded.shape[3] < width:
            raise RuntimeError(f"Decoded {name} is smaller than the original video.")
        decoded = decoded[:, :count, :height, :width]
        if not torch.isfinite(decoded).all():
            raise RuntimeError(f"Non-finite decoded video for {name}")
        save_decoded(output_dir / f"{name}.mp4", decoded, fps)
        del difference, decoded

    metadata = {
        "inputs": {key: str(path) for key, path in paths.items()},
        "vae_type": args.vae_type,
        "vae_checkpoint": str(checkpoint),
        "vae_dtype": args.vae_dtype,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "fps": fps,
        "original_shape_THWC": reference_shape,
        "latent_shape_CTHW": list(latents["A"].shape),
        "padding": "Repeat last frame to 4n+1; replicate right/bottom edges to VAE stride.",
        "operation": "decode(encode(A) - encode(B)) using Wan normalized latents",
        "output": "Crop to original F/H/W; map decoder [-1,1] to [0,255]; no audio.",
        "video_encoding": "H.264 (libx264), yuv420p, CRF 18, faststart",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_tag", required=True, type=file_name)
    parser.add_argument("--vae_checkpoint", required=True, help="Path to the VAE .pth weights")
    parser.add_argument("--vae_type", choices=("2.1", "2.2"), default="2.2")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--vae_dtype", choices=("float32", "float16", "bfloat16"), default="float32",
                        help="VAE autocast dtype; lower precision can reduce activation memory on CUDA")
    parser.add_argument("--disable_cudnn", action="store_true",
                        help="Use native PyTorch kernels instead of cuDNN (may be slower/use more memory)")
    parser.add_argument("--debug", action="store_true", help="Show the complete traceback on failure")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        if args.debug:
            raise
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
