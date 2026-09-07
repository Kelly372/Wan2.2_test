"""Create a first-frame video from an existing video in data/video.

Usage: python prepare_videos.py --video_tag clip
Requires opencv-python, numpy and imageio-ffmpeg (in requirements.txt).
"""

import argparse
import math
from pathlib import Path
import tempfile

import cv2


DATA_DIR = Path(__file__).resolve().parent / "data"


def file_name(value):
    """Accept a filename, never a relative or absolute directory path."""
    if not value.strip() or value in {".", ".."} or any(
        character in value for character in '/\\:*?"<>|'
    ):
        raise argparse.ArgumentTypeError("Please provide a filename without a path.")
    return value


def read_video(path):
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video frame rate: {fps}")
        ok, first_frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot read the first frame: {path}")
        # Count decoded frames instead of relying on approximate metadata.
        frame_count = 1
        while capture.grab():
            frame_count += 1
        return first_frame, frame_count, fps
    finally:
        capture.release()


def write_rgb_video(path, frames, frame_count, fps, size):
    """Stream uint8 RGB frames to a verified H.264/yuv420p MP4."""
    import imageio_ffmpeg
    import numpy as np

    width, height = size
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f"H.264/yuv420p requires positive even dimensions: {size}")
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame count and FPS must be positive.")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".mp4", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    writer = None
    try:
        writer = imageio_ffmpeg.write_frames(
            str(temporary_path), (width, height), fps=fps,
            codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            macro_block_size=1, quality=None,
            # Override imageio's two-decimal input FPS to retain fractional rates.
            input_params=["-r", format(fps, ".15g")],
            output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"],
        )
        writer.send(None)
        written = 0
        for frame in frames:
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError("Each frame must be a uint8 RGB array matching the video size.")
            writer.send(np.ascontiguousarray(frame))
            written += 1
        writer.close()
        writer = None
        if written != frame_count:
            raise RuntimeError(f"Incorrect input frame count: {written}/{frame_count}")
        # Verify the actual output before replacing an existing result.
        first, actual_count, actual_fps = read_video(temporary_path)
        if (actual_count != frame_count or first.shape[:2] != (height, width)
                or not math.isclose(actual_fps, fps, rel_tol=1e-4, abs_tol=1e-3)):
            raise RuntimeError("Encoded video frame count, dimensions or FPS do not match.")
        temporary_path.replace(path)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            temporary_path.unlink(missing_ok=True)
    print(f"Saved: {path} ({frame_count} frames, {fps:g} FPS)")


def write_still_video(path, frame, frame_count, fps):
    from itertools import repeat

    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    write_rgb_video(path, repeat(rgb, frame_count), frame_count, fps, (width, height))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_tag", required=True, type=file_name)
    args = parser.parse_args()

    video_dir = DATA_DIR / "video"
    source_path = video_dir / f"{args.video_tag}.mp4"
    first_frame_path = video_dir / f"{args.video_tag}_first_frame.mp4"
    try:
        first_frame, frame_count, fps = read_video(source_path)
        write_still_video(first_frame_path, first_frame, frame_count, fps)
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
