"""Check stage wiring without loading model weights or requiring CUDA."""

from pathlib import Path
import types
import unittest
from unittest.mock import patch

import run_video_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_stage_order_and_checkpoint_arguments(self):
        events = []
        checkpoint = Path("models/Wan2.2-TI2V-5B")
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True, empty_cache=lambda: events.append("release")))
        with patch.dict("sys.modules", {"torch": fake_torch}), \
                patch.object(pipeline, "resolve_checkpoint", return_value=checkpoint), \
                patch.object(pipeline, "prepare_video", side_effect=lambda *args: events.append("prepare")) as prepare, \
                patch.object(pipeline, "run_vae_diff", side_effect=lambda args: events.append("vae")) as vae, \
                patch.object(pipeline, "run_generation", side_effect=lambda args: events.append("generate")) as generate:
            pipeline.run_pipeline(str(checkpoint), "clip")
        self.assertEqual(events, ["prepare", "vae", "release", "generate"])
        self.assertEqual(prepare.call_args.args[1].name, "clip_lowResolution.mp4")
        self.assertEqual(vae.call_args.args[0].vae_checkpoint, str(checkpoint / "Wan2.2_VAE.pth"))
        self.assertEqual(generate.call_args.args[0].model_path, str(checkpoint))
        self.assertEqual(generate.call_args.args[0].video_tag, "clip")

    def test_failed_preparation_stops_later_stages(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with patch.dict("sys.modules", {"torch": fake_torch}), \
                patch.object(pipeline, "resolve_checkpoint", return_value=Path("models")), \
                patch.object(pipeline, "prepare_video", side_effect=ValueError("bad video")), \
                patch.object(pipeline, "run_vae_diff") as vae, \
                patch.object(pipeline, "run_generation") as generate:
            with self.assertRaisesRegex(ValueError, "bad video"):
                pipeline.run_pipeline("models", "clip")
            vae.assert_not_called()
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
