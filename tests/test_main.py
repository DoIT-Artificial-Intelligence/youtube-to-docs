import os
import unittest
from unittest.mock import mock_open, patch

import polars as pl

from youtube_to_docs import main


class TestMain(unittest.TestCase):
    def setUp(self):
        self.outfile = "test_output_main.csv"
        if os.path.exists(self.outfile):
            os.remove(self.outfile)
        self.sleep_patcher = patch("youtube_to_docs.main.time.sleep")
        self.mock_sleep = self.sleep_patcher.start()

        # Create dummy audio file for tests that need it
        self.dummy_audio = os.path.abspath("dummy_audio.m4a")
        with open(self.dummy_audio, "wb") as f:
            f.write(b"dummy audio content")

    def tearDown(self):
        self.sleep_patcher.stop()
        if os.path.exists(self.outfile):
            os.remove(self.outfile)
        if os.path.exists(self.dummy_audio):
            os.remove(self.dummy_audio)

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("os.makedirs")
    def test_create_new_file(
        self,
        mock_makedirs,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcript 1", False)
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
    @patch("os.makedirs")
    def test_no_verbose_no_cost_columns(
        self,
        mock_makedirs,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcript 1", False)
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
    @patch("os.makedirs")
    def test_append_new_video(
        self,
        mock_makedirs,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
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
        )
        mock_fetch_trans.return_value = ("Transcript 2", False)
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
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_add_new_summary_column(
        self,
        mock_makedirs,
        mock_exists,
        mock_gen_summary,
        mock_get_pricing,
        mock_resolve,
        mock_svc,
    ):
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
    @patch("os.makedirs")
    def test_column_ordering(
        self,
        mock_makedirs,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcript 1", False)
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

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_infographic")
    @patch("os.makedirs")
    def test_infographic_storage(
        self,
        mock_makedirs,
        mock_gen_info,
        mock_get_pricing,
        mock_gen_summary,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcript 1", False)
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
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("os.makedirs")
    def test_translation_columns(
        self,
        mock_makedirs,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcripción 1", False)
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
                "--language",
                "es",
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
    @patch("os.makedirs")
    def test_multiple_languages(
        self,
        mock_makedirs,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcript", False)
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
                "--language",
                "en,es",
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
    @patch("os.makedirs")
    def test_all_gemini_flash_flag(
        self,
        mock_makedirs,
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
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcript 1", False)
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)
        mock_gen_info.return_value = (b"fake_image_bytes", 100, 1290)
        mock_extract_speakers.return_value = ("Speaker 1", 10, 10)
        mock_gen_qa.return_value = ("Q&A", 20, 20)
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

        # Check if generate_summary was called with gemini-3-flash-preview
        mock_gen_summary.assert_called()
        any_flash_summary = any(
            call.args[0] == "gemini-3-flash-preview"
            for call in mock_gen_summary.call_args_list
        )
        self.assertTrue(any_flash_summary)

        # Check if process_tts was called with gemini-2.5-flash-preview-tts-Kore
        mock_tts.assert_called()
        self.assertEqual(
            mock_tts.call_args.args[1], "gemini-2.5-flash-preview-tts-Kore"
        )

        # Check if generate_infographic was called with gemini-2.5-flash-image
        mock_gen_info.assert_called()
        self.assertEqual(mock_gen_info.call_args.args[0], "gemini-2.5-flash-image")

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
    @patch("os.makedirs")
    def test_all_gemini_pro_flag(
        self,
        mock_makedirs,
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
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcript 1", False)
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)
        mock_gen_info.return_value = (b"fake_image_bytes", 100, 1290)
        mock_extract_speakers.return_value = ("Speaker 1", 10, 10)
        mock_gen_qa.return_value = ("Q&A", 20, 20)
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

        # Check if generate_summary was called with gemini-3-pro-preview
        mock_gen_summary.assert_called()
        any_pro_summary = any(
            call.args[0] == "gemini-3-pro-preview"
            for call in mock_gen_summary.call_args_list
        )
        self.assertTrue(any_pro_summary)

        # Check if process_tts was called with gemini-2.5-pro-preview-tts-Kore
        mock_tts.assert_called()
        self.assertEqual(mock_tts.call_args.args[1], "gemini-2.5-pro-preview-tts-Kore")

        # Check if generate_infographic was called with gemini-3-pro-image-preview
        mock_gen_info.assert_called()
        self.assertEqual(mock_gen_info.call_args.args[0], "gemini-3-pro-image-preview")

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
    @patch("os.makedirs")
    def test_all_gemini_flash_pro_image_flag(
        self,
        mock_makedirs,
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
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcript 1", False)
        mock_gen_summary.return_value = ("Summary 1", 100, 50)
        mock_get_pricing.return_value = (0.0, 0.0)
        mock_gen_info.return_value = (b"fake_image_bytes", 100, 1290)
        mock_extract_speakers.return_value = ("Speaker 1", 10, 10)
        mock_gen_qa.return_value = ("Q&A", 20, 20)
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

        # Check if generate_summary was called with gemini-3-flash-preview
        mock_gen_summary.assert_called()
        any_flash_summary = any(
            call.args[0] == "gemini-3-flash-preview"
            for call in mock_gen_summary.call_args_list
        )
        self.assertTrue(any_flash_summary)

        # Check if process_tts was called with gemini-2.5-flash-preview-tts-Kore
        mock_tts.assert_called()
        self.assertEqual(
            mock_tts.call_args.args[1], "gemini-2.5-flash-preview-tts-Kore"
        )

        # Check if generate_infographic was called with gemini-3-pro-image-preview
        mock_gen_info.assert_called()
        self.assertEqual(mock_gen_info.call_args.args[0], "gemini-3-pro-image-preview")

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.get_model_pricing")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("youtube_to_docs.main.generate_one_sentence_summary")
    @patch("os.makedirs")
    def test_one_sentence_summary_column(
        self,
        mock_makedirs,
        mock_gen_one_sentence,
        mock_gen_summary,
        mock_get_pricing,
        mock_fetch_trans,
        mock_details,
        mock_resolve,
        mock_svc,
    ):
        mock_resolve.return_value = ["vid1"]
        mock_details.return_value = (
            "Title 1",
            "Desc",
            "2023-01-01",
            "Chan",
            "Tags",
            "0:01:00",
            "url1",
        )
        mock_fetch_trans.return_value = ("Transcript 1", False)
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


if __name__ == "__main__":
    unittest.main()
