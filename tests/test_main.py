import os
import tempfile
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, mock_open, patch

import polars as pl

from youtube_to_docs import main
from youtube_to_docs.captions import CaptionFiles
from youtube_to_docs.llms import build_summary_prompt


class TestMain(unittest.TestCase):
    def setUp(self):
        self.test_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_obj.name
        self.outfile = os.path.join(self.test_dir, "test_output_main.csv")

        self.sleep_patcher = patch("youtube_to_docs.main.time.sleep")
        self.mock_sleep = self.sleep_patcher.start()

        # Create dummy audio file for tests that need it inside the temp dir
        self.dummy_audio = os.path.join(self.test_dir, "dummy_audio.m4a")
        with open(self.dummy_audio, "wb") as f:
            f.write(b"dummy audio content")

    def tearDown(self):
        self.sleep_patcher.stop()
        self.test_dir_obj.cleanup()

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_create_new_file(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)

        with patch(
            "sys.argv",
            ["main.py", "vid1", "-o", self.outfile, "-m", "gemini-test", "--verbose"],
        ):
            with patch("builtins.open", mock_open()):
                main.main()

        self.assertTrue(os.path.exists(self.outfile))
        df = pl.read_csv(self.outfile)
        self.assertEqual(len(df), 1)
        self.assertEqual(df[0, "URL"], "https://www.youtube.com/watch?v=vid1")
        self.assertEqual(df[0, "Summary Text gemini-test from youtube"], "Summary 1")
        self.assertIn("Summary File gemini-test from youtube", df.columns)
        self.assertIn("gemini-test summary cost from youtube ($)", df.columns)
        self.assertAlmostEqual(
            df[0, "gemini-test summary cost from youtube ($)"], 0.0
        )  # Since pricing is mocked to 0.0
        self.assertIn("Transcript File human generated", df.columns)
        self.assertIn("Transcript characters from youtube", df.columns)

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_no_verbose_no_cost_columns(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)

        # Run without --verbose
        with patch(
            "sys.argv", ["main.py", "vid1", "-o", self.outfile, "-m", "gemini-test"]
        ):
            with patch("builtins.open", mock_open()):
                main.main()

        self.assertTrue(os.path.exists(self.outfile))
        df = pl.read_csv(self.outfile)
        self.assertEqual(len(df), 1)

        # Verify cost columns are NOT present
        self.assertNotIn("gemini-test summary cost from youtube ($)", df.columns)
        # Note: One sentence summary might not be generated
        # if not explicitly requested or if it depends on summary
        # But here we just check if the cost column is missing, which it should be.

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_append_new_video(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        # Create initial CSV
        initial_data = pl.DataFrame(
            {
                "URL": ["https://www.youtube.com/watch?v=vid1"],
                "Title": ["Title 1"],
                "Description": ["Desc"],
                "Data Published": ["2023-01-01"],
                "Channel": ["Chan"],
                "Tags": ["Tags"],
                "Duration": ["0:01:00"],
                "Transcript File human generated": ["path1"],
                "Summary File gemini-test from youtube": ["spath1"],
                "Summary Text gemini-test from youtube": ["Summary 1"],
            }
        )
        initial_data.write_csv(self.outfile)

        mock_resolve.return_value = ["vid2"]
        mock_details.return_value = (
            "Title 2",
            "Desc 2",
            "2023-01-02",
            "Chan 2",
            "Tags 2",
            "0:02:00",
            "url2",
            120.0,
        )
        mock_fetch_trans.return_value = ("Transcript 2", False, "")
        mock_gen_summary.return_value = ("Summary 2", 200, 100)
        mock_get_pricing.return_value = (0.0, 0.0)

        with patch(
            "sys.argv", ["main.py", "vid2", "-o", self.outfile, "-m", "gemini-test"]
        ):
            with patch("builtins.open", mock_open()):
                main.main()

        df = pl.read_csv(self.outfile)
        self.assertEqual(len(df), 2)
        # Verify both videos exist
        self.assertIn("https://www.youtube.com/watch?v=vid1", df["URL"].to_list())
        self.assertIn("https://www.youtube.com/watch?v=vid2", df["URL"].to_list())

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("os.makedirs")
    def test_skip_existing(
        self, mock_makedirs, mock_fetch_trans, mock_details, mock_resolve, mock_svc
    ):
        # Create a dummy transcript file
        dummy_transcript = "transcript_dummy.txt"
        with open(dummy_transcript, "w", encoding="utf-8") as f:
            f.write("Dummy transcript content")

        try:
            # Create initial CSV with summary
            initial_data = pl.DataFrame(
                {
                    "URL": ["https://www.youtube.com/watch?v=vid1"],
                    "Title": ["Title 1"],
                    "Description": ["Desc"],
                    "Data Published": ["2023-01-01"],
                    "Channel": ["Chan"],
                    "Tags": ["Tags"],
                    "Duration": ["0:01:00"],
                    "Transcript characters from youtube": [12],
                    "Transcript File human generated": [dummy_transcript],
                    "Summary File gemini-test from youtube": ["spath1"],
                    "Summary Text gemini-test from youtube": ["Summary 1"],
                    "gemini-test summary cost from youtube ($)": [0.0],
                }
            )
            initial_data.write_csv(self.outfile)

            mock_resolve.return_value = ["vid1"]

            with patch(
                "sys.argv", ["main.py", "vid1", "-o", self.outfile, "-m", "gemini-test"]
            ):
                main.main()

            # If skipped, these should NOT be called
            mock_details.assert_not_called()
            mock_fetch_trans.assert_not_called()
        finally:
            if os.path.exists(dummy_transcript):
                os.remove(dummy_transcript)

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_add_new_summary_column(
        self,
        mock_makedirs,
        mock_exists,
        mock_gen_tags,
        mock_gen_summary,
        mock_get_pricing,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        # Existing CSV without 'Summary Text haiku'
        initial_data = pl.DataFrame(
            {
                "URL": ["https://www.youtube.com/watch?v=vid1"],
                "Title": ["Title 1"],
                "Description": ["Desc"],
                "Data Published": ["2023-01-01"],
                "Channel": ["Chan"],
                "Tags": ["Tags"],
                "Duration": ["0:01:00"],
                "Transcript characters from youtube": [12],
                "Transcript File human generated": ["transcript_vid1.txt"],
                "Summary File gemini-test from youtube": ["spath1"],
                "Summary Text gemini-test from youtube": ["Summary Gemini"],
            }
        )
        initial_data.write_csv(self.outfile)

        mock_resolve.return_value = ["vid1"]
        mock_gen_summary.return_value = ("Summary Haiku", 300, 150)
        mock_get_pricing.return_value = (0.0, 0.0)

        # Mock transcript file existence and reading
        def side_effect(path):
            if path == self.outfile:
                return True
            if "transcript_vid1.txt" in path:
                return True
            return False

        mock_exists.side_effect = side_effect

        with patch(
            "sys.argv",
            ["main.py", "vid1", "-o", self.outfile, "-m", "haiku", "--verbose"],
        ):
            with patch("builtins.open", mock_open(read_data="Transcript Content")):
                main.main()

        df = pl.read_csv(self.outfile)
        self.assertIn("Summary Text haiku from youtube", df.columns)
        self.assertIn("Summary File haiku from youtube", df.columns)
        self.assertIn("haiku summary cost from youtube ($)", df.columns)
        self.assertIn("Summary Text gemini-test from youtube", df.columns)
        self.assertIn("Summary File gemini-test from youtube", df.columns)
        self.assertIn("Transcript File human generated", df.columns)

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_column_ordering(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)

        with patch(
            "sys.argv",
            ["main.py", "vid1", "-o", self.outfile, "-m", "gemini-test", "--verbose"],
        ):
            with patch("builtins.open", mock_open()):
                main.main()

        df = pl.read_csv(self.outfile)
        cols = df.columns

        # Expected order based on logic
        expected_start = [
            "Title",
            "URL",
            "Description",
            "Data Published",
            "Channel",
            "Tags",
            "Duration",
            "Transcript File human generated",
            "Transcript characters from youtube",
        ]
        for i, col in enumerate(expected_start):
            self.assertEqual(cols[i], col)

    @patch("youtube_to_docs.main.process_from_video")
    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_from_video_runs_when_summary_already_exists(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_summary,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
        mock_from_video,
    ):
        """--from-video must run even when transcript outputs already exist (the
        per-model loop skips ahead in that case)."""
        mock_gen_tags.return_value = ("tag1", 1, 1)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 1, 1)
        argv = [
            "main.py",
            "vid1",
            "-o",
            self.outfile,
            "-m",
            "gemini-test",
            "--from-video",
            "agentic",
        ]
        with patch("builtins.open", mock_open()):
            for _ in range(2):
                with patch("sys.argv", argv):
                    main.main()

        self.assertEqual(mock_from_video.call_count, 2)
        self.assertEqual(mock_from_video.call_args.kwargs["modes"], ["agentic"])
        self.assertEqual(mock_from_video.call_args.kwargs["model_name"], "gemini-test")

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_infographic")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_multiple_infographic_models(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_info,
        mock_get_pricing,
        mock_gen_summary,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        """A comma-separated -i makes one infographic per model, saved with the
        extension of the returned image format."""
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)
        mock_gen_info.return_value = (b"\xff\xd8\xff\xe0fake_jpeg", 100, 1290)

        with patch(
            "sys.argv",
            [
                "main.py",
                "vid1",
                "-o",
                self.outfile,
                "-m",
                "gemini-test",
                "--infographic",
                "image-a, image-b",
            ],
        ):
            with patch("builtins.open", mock_open()):
                main.main()

                called_models = [c.args[0] for c in mock_gen_info.call_args_list]
                self.assertEqual(called_models, ["image-a", "image-b"])

                df = pl.read_csv(self.outfile)
                for image_model in ["image-a", "image-b"]:
                    col = (
                        "Summary Infographic File gemini-test from youtube "
                        f"{image_model}"
                    )
                    self.assertIn(col, df.columns)
                    self.assertIn(f" - {image_model} - ", df[0, col])
                    self.assertTrue(df[0, col].endswith("infographic.jpg"))

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_infographic")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_infographic_storage(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_info,
        mock_get_pricing,
        mock_gen_summary,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)
        mock_gen_info.return_value = (b"fake_image_bytes", 100, 1290)

        with patch(
            "sys.argv",
            [
                "main.py",
                "vid1",
                "-o",
                self.outfile,
                "-m",
                "gemini-test",
                "--infographic",
                "gemini-image",
                "--verbose",
            ],
        ):
            with patch("builtins.open", mock_open()):
                main.main()

                # Verify infographic-files directory was created
                any_infographic_dir_call = any(
                    "infographic-files" in str(call)
                    for call in mock_makedirs.call_args_list
                )
                self.assertTrue(any_infographic_dir_call)

                # Verify infographic file path in CSV
                df = pl.read_csv(self.outfile)
                col_name = (
                    "Summary Infographic File gemini-test from youtube gemini-image"
                )
                self.assertIn(col_name, df.columns)
                path = df[0, col_name]
                self.assertIn("infographic-files", path)
                self.assertTrue(path.endswith(".png"))

                # Verify Infographic Cost Column
                cost_col = (
                    "Summary Infographic Cost gemini-test from youtube gemini-image ($)"
                )
                self.assertIn(cost_col, df.columns)
                # Cost should be 0.0 because pricing is mocked to (0.0, 0.0)
                self.assertEqual(df[0, cost_col], 0.0)

                # Verify infographic-prompts directory was created
                any_prompt_dir_call = any(
                    "infographic-prompts" in str(call)
                    for call in mock_makedirs.call_args_list
                )
                self.assertTrue(any_prompt_dir_call)

                # Verify the infographic prompt file path is recorded in CSV
                prompt_col = (
                    "Infographic Prompt Path gemini-test from youtube gemini-image"
                )
                self.assertIn(prompt_col, df.columns)
                prompt_path = df[0, prompt_col]
                self.assertIn("infographic-prompts", prompt_path)
                self.assertTrue(prompt_path.endswith(".txt"))

                # Verify Speaker Columns (updated format)
                self.assertIn("Speakers gemini-test from youtube", df.columns)
                self.assertIn("Speakers File gemini-test from youtube", df.columns)
                self.assertIn(
                    "gemini-test Speaker extraction cost from youtube ($)", df.columns
                )

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_tags")
    def test_summary_prompt_storage(
        self,
        mock_gen_tags,
        mock_get_pricing,
        mock_gen_summary,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)

        with patch(
            "sys.argv",
            ["main.py", "vid1", "-o", self.outfile, "-m", "gemini-test"],
        ):
            main.main()

        df = pl.read_csv(self.outfile)
        prompt_col = "Summary Prompt Path gemini-test from youtube"
        self.assertIn(prompt_col, df.columns)
        prompt_path = df[0, prompt_col]
        self.assertIn("summary-prompts", prompt_path)
        self.assertTrue(prompt_path.endswith(".txt"))

        # The persisted prompt must match what was sent to the model
        with open(prompt_path) as f:
            saved_prompt = f.read()
        self.assertEqual(
            saved_prompt,
            build_summary_prompt(
                "Transcript 1", "Title 1", "https://www.youtube.com/watch?v=vid1"
            ),
        )

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_translation_columns(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcripción 1", False, "")
        mock_gen_summary.return_value = ("Resumen 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)

        with patch(
            "sys.argv",
            [
                "main.py",
                "vid1",
                "-o",
                self.outfile,
                "-m",
                "gemini-test",
                "--translate",
                "gemini-test-es",
            ],
        ):
            with patch("builtins.open", mock_open()):
                main.main()

        self.assertTrue(os.path.exists(self.outfile))
        df = pl.read_csv(self.outfile)
        self.assertEqual(len(df), 1)
        self.assertEqual(df[0, "URL"], "https://www.youtube.com/watch?v=vid1")
        # Column names should have (es)
        self.assertEqual(
            df[0, "Summary Text gemini-test from youtube (es)"], "Resumen 1"
        )
        self.assertIn("Summary File gemini-test from youtube (es)", df.columns)
        self.assertIn("Transcript File human generated (es)", df.columns)
        self.assertIn("Transcript characters from youtube (es)", df.columns)

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_multiple_languages(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript", False, "")
        mock_gen_summary.return_value = ("Summary", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)

        with patch(
            "sys.argv",
            [
                "main.py",
                "vid1",
                "-o",
                self.outfile,
                "-m",
                "gemini-test",
                "--translate",
                "gemini-test-es",
            ],
        ):
            with patch("builtins.open", mock_open()):
                main.main()

        self.assertTrue(os.path.exists(self.outfile))
        df = pl.read_csv(self.outfile)

        # Check EN columns
        self.assertIn("Summary Text gemini-test from youtube", df.columns)
        self.assertIn("Transcript File human generated", df.columns)

        # Check ES columns
        self.assertIn("Summary Text gemini-test from youtube (es)", df.columns)
        self.assertIn("Transcript File human generated (es)", df.columns)

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.process_tts")
    @patch("youtube_to_docs.main.generate_infographic")
    @patch("youtube_to_docs.main.extract_speakers")
    @patch("youtube_to_docs.main.generate_qa")
    @patch("youtube_to_docs.main.extract_audio")
    @patch("youtube_to_docs.storage.LocalStorage.upload_file")
    @patch("youtube_to_docs.providers.get_provider")
    @patch("youtube_to_docs.main.generate_one_sentence_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_all_gemini_flash_flag(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_one_sentence,
        mock_get_provider,
        mock_upload_file,
        mock_extract_audio,
        mock_gen_qa,
        mock_extract_speakers,
        mock_gen_info,
        mock_tts,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_gen_one_sentence.return_value = ("One sentence", 5, 2)

        from youtube_to_docs.providers import BaseProvider, STTProvider

        class MockSTTProvider(BaseProvider, STTProvider):
            def transcribe(self, *args, **kwargs):
                return "AI Transcript", "SRT content", 100, 50

        mock_p = MockSTTProvider("test")
        cast(Any, mock_p).transcribe = MagicMock(
            return_value=("AI Transcript", "SRT content", 100, 50)
        )
        mock_get_provider.return_value = mock_p

        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)
        mock_gen_info.return_value = (b"fake_image_bytes", 100, 1290)
        mock_extract_speakers.return_value = ("Speaker 1", 10, 10)
        mock_gen_qa.return_value = ("Q&A", 20, 20)
        mock_gen_one_sentence.return_value = ("One sentence summary", 5, 5)
        mock_extract_audio.return_value = self.dummy_audio
        mock_upload_file.return_value = "audio-files/vid1.m4a"

        # Ensure process_tts returns the dataframe it receives
        mock_tts.side_effect = lambda df, *args, **kwargs: df

        with patch(
            "sys.argv",
            [
                "main.py",
                "vid1",
                "-o",
                self.outfile,
                "--all",
                "gemini-flash",
            ],
        ):
            # We don't mock open because we want storage to actually work if it's local
            # or at least not fail on write_csv
            main.main()

        # Check if generate_summary was called with gemini-3.5-flash-lite
        mock_gen_summary.assert_called()
        any_flash_summary = any(
            call.args[0] == "gemini-3.5-flash-lite"
            for call in mock_gen_summary.call_args_list
        )
        self.assertTrue(any_flash_summary)

        # Check if process_tts was called with gemini-3.8-flash-lite-tts-Kore
        mock_tts.assert_called()
        self.assertEqual(mock_tts.call_args.args[1], "gemini-3.8-flash-lite-tts-Kore")

        # Check if generate_infographic was called with gemini-nano-banana-2.1
        mock_gen_info.assert_called()
        self.assertEqual(mock_gen_info.call_args.args[0], "gemini-nano-banana-2.1")

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.process_tts")
    @patch("youtube_to_docs.main.generate_infographic")
    @patch("youtube_to_docs.main.extract_speakers")
    @patch("youtube_to_docs.main.generate_qa")
    @patch("youtube_to_docs.main.extract_audio")
    @patch("youtube_to_docs.storage.LocalStorage.upload_file")
    @patch("youtube_to_docs.providers.get_provider")
    @patch("youtube_to_docs.main.generate_one_sentence_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_all_gemini_pro_flag(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_one_sentence,
        mock_get_provider,
        mock_upload_file,
        mock_extract_audio,
        mock_gen_qa,
        mock_extract_speakers,
        mock_gen_info,
        mock_tts,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_gen_one_sentence.return_value = ("One sentence", 5, 2)

        from youtube_to_docs.providers import BaseProvider, STTProvider

        class MockSTTProvider(BaseProvider, STTProvider):
            def transcribe(self, *args, **kwargs):
                return "AI Transcript", "SRT content", 100, 50

        mock_p = MockSTTProvider("test")
        cast(Any, mock_p).transcribe = MagicMock(
            return_value=("AI Transcript", "SRT content", 100, 50)
        )
        mock_get_provider.return_value = mock_p

        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)
        mock_gen_info.return_value = (b"fake_image_bytes", 100, 1290)
        mock_extract_speakers.return_value = ("Speaker 1", 10, 10)
        mock_gen_qa.return_value = ("Q&A", 20, 20)
        mock_gen_one_sentence.return_value = ("One sentence summary", 5, 5)
        mock_extract_audio.return_value = self.dummy_audio
        mock_upload_file.return_value = "audio-files/vid1.m4a"

        # Ensure process_tts returns the dataframe it receives
        mock_tts.side_effect = lambda df, *args, **kwargs: df

        with patch(
            "sys.argv",
            [
                "main.py",
                "vid1",
                "-o",
                self.outfile,
                "--all",
                "gemini-pro",
            ],
        ):
            main.main()

        # Check if generate_summary was called with gemini-4-argon
        mock_gen_summary.assert_called()
        any_pro_summary = any(
            call.args[0] == "gemini-4-argon" for call in mock_gen_summary.call_args_list
        )
        self.assertTrue(any_pro_summary)

        # Check if process_tts was called with gemini-3.8-flash-tts-Kore
        mock_tts.assert_called()
        self.assertEqual(mock_tts.call_args.args[1], "gemini-3.8-flash-tts-Kore")

        # Check if generate_infographic was called with gemini-3-pro-image
        mock_gen_info.assert_called()
        self.assertEqual(mock_gen_info.call_args.args[0], "gemini-3-pro-image")

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.process_tts")
    @patch("youtube_to_docs.main.generate_infographic")
    @patch("youtube_to_docs.main.extract_speakers")
    @patch("youtube_to_docs.main.generate_qa")
    @patch("youtube_to_docs.main.extract_audio")
    @patch("youtube_to_docs.storage.LocalStorage.upload_file")
    @patch("youtube_to_docs.providers.get_provider")
    @patch("youtube_to_docs.main.generate_one_sentence_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_all_gemini_flash_pro_image_flag(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_one_sentence,
        mock_get_provider,
        mock_upload_file,
        mock_extract_audio,
        mock_gen_qa,
        mock_extract_speakers,
        mock_gen_info,
        mock_tts,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_gen_one_sentence.return_value = ("One sentence", 5, 2)

        from youtube_to_docs.providers import BaseProvider, STTProvider

        class MockSTTProvider(BaseProvider, STTProvider):
            def transcribe(self, *args, **kwargs):
                return "AI Transcript", "SRT content", 100, 50

        mock_p = MockSTTProvider("test")
        cast(Any, mock_p).transcribe = MagicMock(
            return_value=("AI Transcript", "SRT content", 100, 50)
        )
        mock_get_provider.return_value = mock_p

        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)
        mock_gen_info.return_value = (b"fake_image_bytes", 100, 1290)
        mock_extract_speakers.return_value = ("Speaker 1", 10, 10)
        mock_gen_qa.return_value = ("Q&A", 20, 20)
        mock_gen_one_sentence.return_value = ("One sentence summary", 5, 5)
        mock_extract_audio.return_value = self.dummy_audio
        mock_upload_file.return_value = "audio-files/vid1.m4a"

        # Ensure process_tts returns the dataframe it receives
        mock_tts.side_effect = lambda df, *args, **kwargs: df

        with patch(
            "sys.argv",
            [
                "main.py",
                "vid1",
                "-o",
                self.outfile,
                "--all",
                "gemini-flash-pro-image",
            ],
        ):
            main.main()

        # Check if generate_summary was called with gemini-3.5-flash-lite
        mock_gen_summary.assert_called()
        any_flash_summary = any(
            call.args[0] == "gemini-3.5-flash-lite"
            for call in mock_gen_summary.call_args_list
        )
        self.assertTrue(any_flash_summary)

        # Check if process_tts was called with gemini-3.8-flash-lite-tts-Kore
        mock_tts.assert_called()
        self.assertEqual(mock_tts.call_args.args[1], "gemini-3.8-flash-lite-tts-Kore")

        # Check if generate_infographic was called with gemini-3-pro-image
        mock_gen_info.assert_called()
        self.assertEqual(mock_gen_info.call_args.args[0], "gemini-3-pro-image")

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_one_sentence_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_one_sentence_summary_column(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_one_sentence,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_gen_one_sentence.return_value = ("One Sentence Summary 1", 10, 10)
        mock_get_pricing.return_value = (0.0, 0.0)

        with patch(
            "sys.argv",
            ["main.py", "vid1", "-o", self.outfile, "-m", "gemini-test", "--verbose"],
        ):
            with patch("builtins.open", mock_open()):
                main.main()

        self.assertTrue(os.path.exists(self.outfile))
        df = pl.read_csv(self.outfile)
        self.assertEqual(len(df), 1)
        self.assertEqual(
            df[0, "One Sentence Summary gemini-test from youtube"],
            "One Sentence Summary 1",
        )
        self.assertIn(
            "gemini-test one sentence summary cost from youtube ($)", df.columns
        )

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_tags")
    @patch("os.makedirs")
    def test_null_storage(
        self,
        mock_makedirs,
        mock_gen_tags,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_gen_tags.return_value = ("tag1, tag2", 10, 5)
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
            60.0,
        )
        mock_fetch_trans.return_value = ("Transcript 1", False, "")
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)

        with patch(
            "sys.argv",
            ["main.py", "vid1", "-o", "n", "-m", "gemini-test", "--verbose"],
        ):
            # Capture stdout to ensure it doesn't fail and we can check output if needed
            with patch("youtube_to_docs.main.rprint") as mock_rprint:
                main.main()

        # Verify no file was created (this relies on the fact that self.outfile
        # is NOT used)
        self.assertFalse(os.path.exists("none.csv"))

        # Verify rprint was called (results are printed)
        mock_rprint.assert_called()
        # Check if some expected text was printed
        any_results_header = any(
            "Results (Not Saved)" in str(call) for call in mock_rprint.call_args_list
        )
        self.assertTrue(any_results_header)


@patch("youtube_to_docs.main.process_corrected_captions")
@patch("youtube_to_docs.main.get_provider")
@patch("youtube_to_docs.main.extract_audio")
@patch("youtube_to_docs.main.get_youtube_service")
@patch("youtube_to_docs.main.resolve_video_ids", return_value=["vid1"])
@patch("youtube_to_docs.main.get_video_details")
@patch("youtube_to_docs.main.fetch_transcript", return_value=("Hi.", True, []))
@patch("youtube_to_docs.main.generate_summary", return_value=("Summary", 1, 1))
@patch("youtube_to_docs.main.extract_speakers", return_value=("Jane Doe", 1, 1))
@patch("youtube_to_docs.main.generate_qa", return_value=("nan", 1, 1))
@patch("youtube_to_docs.main.generate_one_sentence_summary", return_value=("S", 1, 1))
@patch("youtube_to_docs.main.generate_tags", return_value=("tag", 1, 1))
class TestCorrectedCaptionsDefaults(unittest.TestCase):
    """`-cc` without a value transcribes with gemini-3.5-transcribe and corrects
    that SRT with gemini-4-argon, unless -t says otherwise."""

    def _run(self, mocks, *args):
        (*_, mock_details, _resolve, _svc, mock_extract, mock_provider, mock_cc) = mocks
        mock_details.return_value = (
            "Title",
            "Desc",
            "2026-10-15",
            "Chan",
            "",
            "0:01:00",
            "url",
            60.0,
        )
        from youtube_to_docs.providers import BaseProvider, STTProvider

        class FakeSTT(BaseProvider, STTProvider):
            def transcribe(self, *a, **k):
                return (
                    "Hi.",
                    "1\n00:00:00,000 --> 00:00:01,000\n[Speaker 1] Hi.\n",
                    1,
                    1,
                )

        with tempfile.TemporaryDirectory() as tmp:
            audio = os.path.join(tmp, "a.m4a")
            with open(audio, "wb") as f:
                f.write(b"audio")
            mock_extract.return_value = audio
            mock_provider.side_effect = lambda name: FakeSTT(name)
            main.main(["vid1", "-o", os.path.join(tmp, "out.csv"), *args])
        return mock_provider, mock_cc

    def test_bare_flag_uses_best_defaults(self, *mocks):
        mock_provider, mock_cc = self._run(mocks, "-cc")
        mock_provider.assert_called_with("gemini-3.5-transcribe")
        kwargs = mock_cc.call_args.kwargs
        self.assertEqual(kwargs["model_name"], "gemini-4-argon")
        self.assertEqual(kwargs["source"], "gemini-3.5-transcribe")
        self.assertEqual(kwargs["transcript_arg"], "gemini-3.5-transcribe")

    def test_explicit_transcript_source_is_kept(self, *mocks):
        mock_provider, mock_cc = self._run(mocks, "-t", "youtube", "-cc")
        mock_provider.assert_not_called()
        self.assertEqual(mock_cc.call_args.kwargs["transcript_arg"], "youtube")

    @patch("youtube_to_docs.main.write_player_page")
    @patch("youtube_to_docs.main.process_accessible_video")
    @patch("youtube_to_docs.main.process_audio_description")
    @patch("youtube_to_docs.main.save_original_video")
    @patch("youtube_to_docs.main.download_video")
    def test_corrected_captions_do_the_whole_pipeline(
        self, mock_download, mock_original, mock_ad, mock_video, mock_page, *mocks
    ):
        files = CaptionFiles("cc/out.srt", "gemini-3.5-transcribe", "stt.srt", None)
        mocks[-1].return_value = files
        described = [object()]
        seen: list = []
        with tempfile.TemporaryDirectory():

            def fake_download(video_id, directory):
                os.makedirs(directory, exist_ok=True)
                path = os.path.join(directory, f"{video_id}.mp4")
                with open(path, "wb") as f:
                    f.write(b"video")
                seen.append(path)
                return path

            mock_download.side_effect = fake_download
            mock_original.side_effect = lambda **kwargs: kwargs["get_video"]()
            mock_ad.side_effect = lambda **kwargs: (kwargs["get_video"](), described)[1]
            mock_video.side_effect = lambda **kwargs: kwargs["get_video"]()
            _, mock_cc = self._run(mocks, "-cc")
            self.assertFalse(os.path.exists(seen[0]))  # temporary copy removed
        mock_download.assert_called_once()  # shared by every step
        self.assertEqual(mock_cc.call_args.kwargs["model_name"], "gemini-4-argon")
        self.assertTrue(
            mock_original.call_args.kwargs["path"].endswith(" - original video.mp4")
        )
        self.assertIs(mock_ad.call_args.kwargs["captions"], files)
        self.assertEqual(mock_ad.call_args.kwargs["model_name"], "gemini-4-argon")
        video = mock_video.call_args.kwargs
        self.assertIs(video["descriptions"], described)
        self.assertEqual(video["captions_path"], "cc/out.srt")
        self.assertEqual(video["source"], "gemini-3.5-transcribe")
        self.assertTrue(mock_page.call_args.kwargs["path"].endswith(" - player.html"))

    @patch("youtube_to_docs.main.write_player_page")
    @patch("youtube_to_docs.main.process_accessible_video")
    @patch("youtube_to_docs.main.process_audio_description")
    @patch("youtube_to_docs.main.download_video", return_value=None)
    def test_failed_video_download_skips_the_video_steps(
        self, mock_download, mock_ad, mock_video, mock_page, *mocks
    ):
        mocks[-1].return_value = CaptionFiles(
            "cc/out.srt", "gemini-3.5-transcribe", "stt.srt", None
        )
        with patch("youtube_to_docs.main.rprint") as mock_print:
            self._run(mocks, "-cc")
        mock_download.assert_called_once()
        mock_ad.assert_not_called()
        mock_video.assert_not_called()
        mock_page.assert_not_called()  # a page with no video to play
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn("Couldn't download the video", printed)

    def test_youtube_source_needs_no_transcription(self, *mocks):
        mock_provider, mock_cc = self._run(mocks, "-cc", "gemini-4-argon-youtube")
        mock_provider.assert_not_called()
        self.assertEqual(mock_cc.call_args.kwargs["source"], "youtube")


if __name__ == "__main__":
    unittest.main()
