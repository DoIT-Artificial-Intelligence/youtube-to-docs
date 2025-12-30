import subprocess
import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs import video


class TestVideo(unittest.TestCase):
    @patch("youtube_to_docs.video.run.get_or_fetch_platform_executables_else_raise")
    @patch("youtube_to_docs.video.subprocess.run")
    def test_create_video_success(self, mock_run, mock_get_executables):
        # Mock ffmpeg path retrieval from static_ffmpeg
        mock_get_executables.return_value = ("/usr/bin/ffmpeg", None)

        # Mock subprocess.run success
        mock_run.return_value = MagicMock(returncode=0)

        result = video.create_video("image.png", "audio.mp3", "output.mp4")

        self.assertTrue(result)
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0][0], "/usr/bin/ffmpeg")
        self.assertIn("-i", args[0])
        self.assertIn("image.png", args[0])
        self.assertIn("audio.mp3", args[0])
        self.assertIn("output.mp4", args[0])
        # Verify static_ffmpeg was called
        mock_get_executables.assert_called_once()

    @patch("youtube_to_docs.video.run.get_or_fetch_platform_executables_else_raise")
    @patch("youtube_to_docs.video.subprocess.run")
    def test_create_video_failure(self, mock_run, mock_get_executables):
        # Mock ffmpeg path retrieval from static_ffmpeg
        mock_get_executables.return_value = ("/usr/bin/ffmpeg", None)

        # Mock subprocess.run failure
        mock_run.side_effect = subprocess.CalledProcessError(1, ["ffmpeg"])

        result = video.create_video("image.png", "audio.mp3", "output.mp4")

        self.assertFalse(result)
        mock_run.assert_called_once()
        mock_get_executables.assert_called_once()


if __name__ == "__main__":
    unittest.main()
