import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from youtube_to_docs.storage import Storage
from youtube_to_docs.video import create_video, process_videos


class TestVideo(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, "test_image.png")
        self.audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        self.output_path = os.path.join(self.temp_dir, "test_output.mp4")

        # Create dummy image and audio files
        with open(self.image_path, "wb") as f:
            f.write(b"fake image data")
        with open(self.audio_path, "wb") as f:
            f.write(b"fake audio data")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("youtube_to_docs.video.subprocess.run")
    @patch("static_ffmpeg.run.get_or_fetch_platform_executables_else_raise")
    def test_create_video_success(self, mock_get_ffmpeg, mock_run):
        mock_get_ffmpeg.return_value = ("/usr/bin/ffmpeg", None)
        mock_run.return_value = MagicMock(returncode=0)

        result = create_video(self.image_path, self.audio_path, self.output_path)
        self.assertTrue(result)
        mock_run.assert_called_once()

        # Verify ffmpeg command args
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "/usr/bin/ffmpeg")
        self.assertIn("-loop", args)
        self.assertIn("1", args)
        self.assertIn(self.image_path, args)
        self.assertIn(self.audio_path, args)
        self.assertIn(self.output_path, args)

    @patch("youtube_to_docs.video.subprocess.run")
    @patch("static_ffmpeg.run.get_or_fetch_platform_executables_else_raise")
    def test_create_video_failure(self, mock_get_ffmpeg, mock_run):
        mock_get_ffmpeg.return_value = ("/usr/bin/ffmpeg", None)
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

        result = create_video(self.image_path, self.audio_path, self.output_path)
        self.assertFalse(result)

    def test_process_videos(self):
        # Mock Storage
        mock_storage = MagicMock(spec=Storage)
        mock_storage.exists.return_value = True
        mock_storage.read_bytes.return_value = b"data"
        mock_storage.upload_file.return_value = "http://mock-link/video.mp4"

        # Mock create_video to avoid actual ffmpeg call
        def create_video_side_effect(img, audio, output):
            with open(output, "wb") as f:
                f.write(b"dummy video")
            return True

        with patch(
            "youtube_to_docs.video.create_video", side_effect=create_video_side_effect
        ) as mock_create:
            # Create a mock DataFrame
            data = {
                "Title": ["Test Video"],
                "Summary Infographic File Model1": ["/path/to/info.png"],
                "Summary Audio File Model1 File": ["/path/to/audio.wav"],
            }
            df = pl.DataFrame(data)

            # Define exists side effect
            def exists_side_effect(path):
                if "video-files" in path or ".mp4" in path:
                    return False
                return True

            mock_storage.exists.side_effect = exists_side_effect

            # Run process_videos
            updated_df = process_videos(df, mock_storage, base_dir=self.temp_dir)

            # Assertions
            self.assertIn("Video File", updated_df.columns)
            self.assertEqual(updated_df["Video File"][0], "http://mock-link/video.mp4")

            mock_create.assert_called_once()
            mock_storage.ensure_directory.assert_called()
            mock_storage.read_bytes.assert_called()
            mock_storage.upload_file.assert_called()


if __name__ == "__main__":
    unittest.main()
