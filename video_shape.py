"""Print the decoded frame count, height and width of a video.

Usage: python video_shape.py --video_path data/video/example.mp4
"""

import argparse
from pathlib import Path

import cv2


def video_shape(path):
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        count = 0
        shape = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if shape is None:
                shape = frame.shape[:2]
            elif frame.shape[:2] != shape:
                raise ValueError(f"Video dimensions change at frame {count + 1}.")
            count += 1
        if shape is None:
            raise ValueError(f"No readable frames: {path}")
        height, width = shape
        return count, height, width
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True, type=Path)
    args = parser.parse_args()
    try:
        count, height, width = video_shape(args.video_path.expanduser())
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"F={count}, H={height}, W={width}")


if __name__ == "__main__":
    main()
