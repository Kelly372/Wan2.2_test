"""Create a first-frame video from an existing video in data/video.

Usage: python prepare_videos.py --video_tag clip
Requires opencv-python and numpy (already included in requirements.txt).
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


def write_still_video(path, frame, frame_count, fps):
    height, width = frame.shape[:2]
    # OpenCV's MP4 writer truncates odd dimensions; fail instead of silently
    # changing the requested resolution.
    if width % 2 or height % 2:
        raise ValueError(f"MP4 output requires even dimensions, got {width}x{height}.")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".mp4", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    writer = None
    try:
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot initialize the MP4 encoder for: {path}")
        for _ in range(frame_count):
            writer.write(frame)
        writer.release()
        writer = None
        # Verify the actual output before replacing an existing result.
        _, actual_count, _ = read_video(temporary_path)
        if actual_count != frame_count:
            raise RuntimeError(f"Incomplete output: {actual_count}/{frame_count} frames")
        temporary_path.replace(path)
    finally:
        if writer is not None:
            writer.release()
        temporary_path.unlink(missing_ok=True)
    print(f"Saved: {path} ({frame_count} frames, {fps:g} FPS)")


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
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
