"""Run with: python -m unittest discover -s tests -p test_video_encoding.py"""

from pathlib import Path
import subprocess
import tempfile
import unittest

import imageio_ffmpeg
import numpy as np

from run_diff_video import prepare_video, read_video, write_rgb_video, write_still_video


class VideoEncodingTests(unittest.TestCase):
    def test_preparation_resizes_motion_and_repeats_first_frame(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            low = Path(directory) / "source_lowResolution.mp4"
            still = Path(directory) / "source_first_frame.mp4"
            frames = [np.full((400, 704, 3), value, dtype=np.uint8) for value in (20, 100, 220)]
            write_rgb_video(source, frames, 3, 24, (704, 400))
            original_bytes = source.read_bytes()
            prepare_video(source, low, still)
            self.assertEqual(source.read_bytes(), original_bytes)
            for path in (low, still):
                first, count, fps = read_video(path)
                self.assertEqual(first.shape, (384, 672, 3))
                self.assertEqual(count, 3)
                self.assertAlmostEqual(fps, 24)
            for path, expected in ((low, (20, 100, 220)), (still, (20, 20, 20))):
                capture = cv2.VideoCapture(str(path))
                try:
                    for value in expected:
                        ok, frame = capture.read()
                        self.assertTrue(ok)
                        self.assertLess(abs(frame.mean() - value), 6)
                finally:
                    capture.release()

    def test_h264_color_dimensions_fps_and_faststart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "first_frame.mp4"
            # Non-macroblock-aligned size and fractional FPS must be preserved.
            frame = np.zeros((34, 50, 3), dtype=np.uint8)
            frame[:, :, 2] = 255  # OpenCV BGR red.
            write_still_video(path, frame, 7, 24000 / 1001)
            decoded, count, fps = read_video(path)
            self.assertEqual(decoded.shape, frame.shape)
            self.assertEqual(count, 7)
            self.assertAlmostEqual(fps, 24000 / 1001, places=3)
            self.assertGreater(decoded[:, :, 2].mean(), 240)
            self.assertLess(decoded[:, :, 0].mean(), 10)
            probe = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                capture_output=True, text=True,
            ).stderr
            self.assertIn("Video: h264", probe)
            self.assertIn("yuv420p", probe)
            data = path.read_bytes()
            self.assertGreater(data.find(b"moov"), 0)
            self.assertLess(data.find(b"moov"), data.find(b"mdat"))

    def test_bad_frame_count_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.mp4"
            path.write_bytes(b"existing output")
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            with self.assertRaises(RuntimeError):
                write_rgb_video(path, [frame], 2, 24, (32, 32))
            self.assertEqual(path.read_bytes(), b"existing output")
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
