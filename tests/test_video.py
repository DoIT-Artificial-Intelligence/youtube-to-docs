import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from youtube_to_docs.storage import Storage
from youtube_to_docs.video import (
    create_video,
    process_accessible_video,
    process_videos,
    render_accessible_video,
    save_original_video,
    write_player_page,
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


class TestAccessibleVideo(unittest.TestCase):
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

    def _render(self, narration=None, returncode=0):
        seen = {}

        def fake_popen(command, **kwargs):
            seen["command"] = command
            with open(os.path.join(kwargs["cwd"], "captions.srt")) as f:
                seen["srt"] = f.read()
            process = MagicMock(returncode=returncode)
            process.stdout = iter(
                ["out_time_us=5000000\n", "progress=continue\n", "out_time_us=x\n"]
            )
            return process

        output = os.path.join(self.temp_dir, "out.mp4")
        with (
            patch(
                "static_ffmpeg.run.get_or_fetch_platform_executables_else_raise",
                return_value=("/usr/bin/ffmpeg", None),
            ),
            patch("youtube_to_docs.video.subprocess.Popen", side_effect=fake_popen),
            patch("youtube_to_docs.video.rprint") as mock_print,
        ):
            ok = render_accessible_video(self.video, self.srt, output, 10.0, narration)
        printed = [call.args[0] for call in mock_print.call_args_list]
        return ok, seen, printed, output

    def test_render_captions_only(self):
        ok, seen, printed, output = self._render()
        self.assertTrue(ok)
        command = seen["command"]
        video_filter = command[command.index("-vf") + 1]
        self.assertTrue(video_filter.startswith("subtitles=captions.srt:"))
        self.assertIn("BorderStyle=3", video_filter)
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertEqual(command[-1], os.path.abspath(output))
        self.assertIn("[Jane Doe] Hi.", seen["srt"])
        self.assertIn("  Rendering video: 50%", printed)

    def test_render_with_narration_ducks_programme_audio(self):
        ok, seen, _, _ = self._render(narration="narration.wav")
        self.assertTrue(ok)
        command = seen["command"]
        self.assertIn(os.path.abspath("narration.wav"), command)
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("subtitles=captions.srt", graph)
        self.assertIn("sidechaincompress", graph)
        self.assertIn("amix=inputs=2", graph)
        self.assertEqual(command[command.index("-c:a") + 1], "aac")

    def test_render_failure(self):
        ok, _, _, _ = self._render(returncode=1)
        self.assertFalse(ok)

    def _storage(self, existing=(), texts=None):
        storage = MagicMock()
        storage.exists.side_effect = lambda path: path in existing
        storage.get_full_path.side_effect = lambda path: path
        # The corrected captions as a Google Doc exports them: one line a cue.
        texts = {
            "cc.srt": "\ufeff1 00:00:01,000 --> 00:00:02,000 Hi.\r\n",
            **(texts or {}),
        }
        storage.read_text.side_effect = lambda path: texts[path]
        storage.upload_file.side_effect = lambda local, target, content_type=None: (
            target
        )
        return storage

    def _process(self, row, storage, descriptions=None, vtt=None, video="v"):
        get_video = MagicMock(return_value=self.video if video == "v" else video)
        process_accessible_video(
            row=row,
            storage=storage,
            captions_path="cc.srt",
            model_name="gemini-4-argon",
            source="gemini-3.5-transcribe",
            descriptions=descriptions,
            descriptions_vtt=vtt,
            tts_arg=None,
            get_video=get_video,
            video_id="vid",
            safe_title="Title",
            video_dir="video-files",
            duration_seconds=60.0,
            vprint=lambda *a, **k: None,
        )
        return get_video

    TARGET = os.path.join(
        "video-files",
        "gemini-4-argon - vid - Title - captioned and described video "
        "(from gemini-3.5-transcribe).mp4",
    )
    COLUMN = (
        "Captioned And Described Video File gemini-4-argon from gemini-3.5-transcribe"
    )

    @patch("youtube_to_docs.video.render_accessible_video", return_value=True)
    @patch("youtube_to_docs.audio_description.write_narration_track", return_value=True)
    def test_uses_this_runs_narration(self, mock_track, mock_render):
        rendered = []
        mock_render.side_effect = lambda video, srt, *args: (
            rendered.append(open(srt, encoding="utf-8").read()) or True
        )
        row = {}
        storage = self._storage()
        described = [MagicMock()]
        self._process(row, storage, descriptions=described)
        self.assertEqual(row[self.COLUMN], self.TARGET)
        self.assertIs(mock_track.call_args.args[0], described)
        video, srt, _, duration, narration = mock_render.call_args.args
        self.assertEqual((video, duration), (self.video, 60.0))
        self.assertEqual(os.path.basename(srt), "captions.srt")
        self.assertEqual(rendered, ["1\n00:00:01,000 --> 00:00:02,000\nHi.\n"])
        self.assertTrue(narration.endswith("narration.wav"))
        self.assertEqual(storage.upload_file.call_args.args[1], self.TARGET)

    @patch("youtube_to_docs.video.render_accessible_video", return_value=True)
    @patch("youtube_to_docs.audio_description.write_narration_track", return_value=True)
    @patch("youtube_to_docs.audio_description.synthesize")
    def test_narrates_the_saved_descriptions_track(
        self, mock_tts, mock_track, mock_render
    ):
        vtt = "WEBVTT\n\n00:00:05.000 --> 00:00:07.000\nSlide: Q&amp;A.\n"
        storage = self._storage(existing={"d.vtt"}, texts={"d.vtt": vtt})
        self._process({}, storage, descriptions=None, vtt="d.vtt")
        narrated = mock_tts.call_args.args[0]
        self.assertEqual(
            [(d.start, d.spoken) for d in narrated], [(5.0, "Slide: Q&A.")]
        )
        self.assertEqual(mock_tts.call_args.args[1], "gemini-3.8-flash-tts-Kore")

    def test_skips_existing(self):
        row = {}
        get_video = self._process(row, self._storage({self.TARGET}))
        get_video.assert_not_called()
        self.assertEqual(row[self.COLUMN], self.TARGET)

    @patch("youtube_to_docs.video.render_accessible_video")
    def test_download_failure(self, mock_render):
        row = {}
        self._process(row, self._storage(), video=None)
        mock_render.assert_not_called()
        self.assertEqual(row, {})

    def test_save_original_video(self):
        row = {}
        storage = self._storage()
        save_original_video(
            row=row,
            storage=storage,
            get_video=lambda: self.video,
            path="video-files/vid - Title - original video.mp4",
        )
        self.assertEqual(
            row["Original Video File"], "video-files/vid - Title - original video.mp4"
        )
        storage.upload_file.assert_called_once_with(
            self.video,
            "video-files/vid - Title - original video.mp4",
            content_type="video/mp4",
        )

    def test_player_page(self):
        row = {}
        storage = MagicMock()
        storage.write_text.side_effect = lambda path, text: path
        tracks = {
            os.path.join("out", "vtt-files", "captions.vtt"): "WEBVTT\n\nA </script>",
            os.path.join("out", "vtt-files", "extended.vtt"): "WEBVTT\n\nSlide.",
        }
        storage.exists.side_effect = lambda path: path in tracks
        storage.read_text.side_effect = tracks.__getitem__
        write_player_page(
            row=row,
            storage=storage,
            path=os.path.join("out", "vtt-files", "page.html"),
            title="AIB Meeting & More",
            video_path=os.path.join("out", "video-files", "vid - original video.mp4"),
            captions_path=os.path.join("out", "vtt-files", "captions.vtt"),
            descriptions_path=os.path.join("out", "vtt-files", "extended.vtt"),
            transcript_path=os.path.join("out", "dt", "transcript.md"),
        )
        page = storage.write_text.call_args.args[1]
        self.assertIn("<title>AIB Meeting &amp; More: accessible player</title>", page)
        self.assertIn('src="../video-files/vid%20-%20original%20video.mp4"', page)
        self.assertIn('<track kind="captions" src="captions.vtt"', page)
        self.assertIn('<track kind="descriptions" src="extended.vtt"', page)
        self.assertIn('data-desc-pause-default="on"', page)
        self.assertIn('data-state-descriptions="on"', page)
        self.assertIn('href="../dt/transcript.md"', page)
        self.assertIn("ableplayer@5.0.0/build/ableplayer.min.js", page)
        self.assertEqual(
            row["Player Page"], os.path.join("out", "vtt-files", "page.html")
        )
        # Copies of the tracks for pages opened from disk (file://), with "<"
        # escaped so a track can't close the script element.
        copies = page.split('id="track-copies">')[1].split("</script>")[0]
        self.assertEqual(
            json.loads(copies),
            {"captions": "WEBVTT\n\nA </script>", "descriptions": "WEBVTT\n\nSlide."},
        )
        self.assertIn("\\u003c/script>", copies)
        self.assertIn('if (location.protocol === "file:")', page)

    def test_embed_page(self):
        # Streams from YouTube and always uses the embedded tracks, so it can
        # be pasted into a site with no files beside it.
        row = {}
        storage = MagicMock()
        storage.write_text.side_effect = lambda path, text: path
        storage.exists.return_value = True
        storage.read_text.return_value = "WEBVTT\n\nHi."
        write_player_page(
            row=row,
            storage=storage,
            path="embed.html",
            title="Title",
            video_path="video-files/vid.mp4",
            captions_path="https://drive.google.com/file/d/c/view",
            descriptions_path="https://drive.google.com/file/d/d/view",
            transcript_path="https://docs.google.com/document/d/t/edit",
            youtube_id="Fpg75xTOMe4",
            column="Embed Page",
        )
        page = storage.write_text.call_args.args[1]
        self.assertIn('data-state-descriptions="on" data-youtube-id="Fpg75xTOMe4">', page)
        self.assertNotIn("<source", page)
        self.assertIn("if (true) {", page)
        self.assertIn('href="https://docs.google.com/document/d/t/edit"', page)
        self.assertEqual(row, {"Embed Page": "embed.html"})

    def test_embed_page_skips_relative_transcript(self):
        storage = MagicMock()
        storage.exists.return_value = False
        write_player_page(
            row={},
            storage=storage,
            path="embed.html",
            title="Title",
            video_path="video.mp4",
            captions_path="captions.vtt",
            descriptions_path=None,
            transcript_path="dt/transcript.md",
            youtube_id="vid",
        )
        self.assertNotIn("Descriptive transcript", storage.write_text.call_args.args[1])

    def test_player_page_without_track_copies(self):
        storage = MagicMock()
        storage.exists.return_value = False
        write_player_page(
            row={},
            storage=storage,
            path="page.html",
            title="Title",
            video_path="video.mp4",
            captions_path="captions.vtt",
            descriptions_path=None,
            transcript_path=None,
        )
        page = storage.write_text.call_args.args[1]
        self.assertIn('<track kind="captions" src="captions.vtt"', page)
        self.assertNotIn("track-copies", page)
        self.assertNotIn('kind="descriptions"', page)


if __name__ == "__main__":
    unittest.main()
