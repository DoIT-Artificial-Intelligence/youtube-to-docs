import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from youtube_to_docs.storage import Storage
from youtube_to_docs.video import (
    burn_captions,
    create_video,
    process_captioned_video,
    process_videos,
)


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


class TestCaptionedVideo(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.srt = os.path.join(self.temp_dir, "c.srt")
        with open(self.srt, "w") as f:
            f.write("1\n00:00:00,000 --> 00:00:01,000\n[Jane Doe] Hi.\n")
        self.video = os.path.join(self.temp_dir, "v.mp4")
        with open(self.video, "wb") as f:
            f.write(b"video")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("youtube_to_docs.video.subprocess.Popen")
    @patch(
        "static_ffmpeg.run.get_or_fetch_platform_executables_else_raise",
        return_value=("/usr/bin/ffmpeg", None),
    )
    def test_burn_captions_command_and_progress(self, mock_get, mock_popen):
        seen = {}

        def fake_popen(command, **kwargs):
            seen["command"] = command
            with open(os.path.join(kwargs["cwd"], "captions.srt")) as f:
                seen["srt"] = f.read()
            process = MagicMock(returncode=0)
            process.stdout = iter(
                ["out_time_us=5000000\n", "progress=continue\n", "out_time_us=x\n"]
            )
            return process

        mock_popen.side_effect = fake_popen
        output = os.path.join(self.temp_dir, "out.mp4")
        with patch("youtube_to_docs.video.rprint") as mock_print:
            self.assertTrue(burn_captions(self.video, self.srt, output, 10.0))

        command = seen["command"]
        video_filter = command[command.index("-vf") + 1]
        self.assertTrue(video_filter.startswith("subtitles=captions.srt:"))
        self.assertIn("BorderStyle=3", video_filter)
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertEqual(command[-1], os.path.abspath(output))
        self.assertIn("[Jane Doe] Hi.", seen["srt"])
        printed = [call.args[0] for call in mock_print.call_args_list]
        self.assertIn("  Burning captions: 50%", printed)

    @patch("youtube_to_docs.video.subprocess.Popen")
    @patch(
        "static_ffmpeg.run.get_or_fetch_platform_executables_else_raise",
        return_value=("/usr/bin/ffmpeg", None),
    )
    def test_burn_captions_failure(self, mock_get, mock_popen):
        mock_popen.return_value = MagicMock(returncode=1, stdout=iter([]))
        self.assertFalse(burn_captions(self.video, self.srt, "out.mp4"))

    def _storage(self, existing=()):
        storage = MagicMock()
        storage.exists.side_effect = lambda path: path in existing
        storage.get_full_path.side_effect = lambda path: path
        storage.get_local_file.side_effect = lambda path, download_dir=None: (
            self.srt if path == "cc.srt" else None
        )
        storage.upload_file.side_effect = lambda local, target, content_type=None: (
            target
        )
        return storage

    def _process(self, row, storage, source="gemini-3.5-transcribe"):
        process_captioned_video(
            row=row,
            storage=storage,
            captions_path="cc.srt",
            model_name="gemini-4-argon",
            source=source,
            video_id="vid",
            safe_title="Title",
            video_dir="video-files",
            duration_seconds=60.0,
            vprint=lambda *a, **k: None,
        )

    @patch("youtube_to_docs.video.burn_captions", return_value=True)
    @patch("youtube_to_docs.video.download_video")
    def test_process_captioned_video_saves(self, mock_download, mock_burn):
        mock_download.return_value = self.video
        row = {}
        storage = self._storage()
        self._process(row, storage)
        target = os.path.join(
            "video-files",
            "gemini-4-argon - vid - Title - captioned video "
            "(from gemini-3.5-transcribe).mp4",
        )
        self.assertEqual(
            row["Captioned Video File gemini-4-argon from gemini-3.5-transcribe"],
            target,
        )
        self.assertEqual(storage.upload_file.call_args.args[1], target)
        self.assertEqual(
            storage.upload_file.call_args.kwargs["content_type"], "video/mp4"
        )
        self.assertEqual(mock_burn.call_args.args[:2], (self.video, self.srt))
        self.assertEqual(mock_burn.call_args.args[3], 60.0)

    @patch("youtube_to_docs.video.download_video")
    def test_process_captioned_video_skips_existing(self, mock_download):
        target = os.path.join(
            "video-files",
            "gemini-4-argon - vid - Title - captioned video (from youtube).mp4",
        )
        row = {}
        self._process(row, self._storage({target}), source="youtube")
        mock_download.assert_not_called()
        self.assertEqual(
            row["Captioned Video File gemini-4-argon from youtube"], target
        )

    @patch("youtube_to_docs.video.burn_captions")
    @patch("youtube_to_docs.video.download_video", return_value=None)
    def test_process_captioned_video_download_failure(self, mock_download, mock_burn):
        row = {}
        self._process(row, self._storage())
        mock_burn.assert_not_called()
        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
