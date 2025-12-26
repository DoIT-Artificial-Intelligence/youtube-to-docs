import os
import unittest
from unittest.mock import mock_open, patch

import polars as pl

from youtube_to_docs import main


class TestCSVAppendLogic(unittest.TestCase):
    def setUp(self):
        self.outfile = "test_output.csv"
        if os.path.exists(self.outfile):
            os.remove(self.outfile)

    def tearDown(self):
        if os.path.exists(self.outfile):
            os.remove(self.outfile)

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("os.makedirs")
    def test_create_new_file(
        self,
        mock_makedirs,
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
        mock_gen_summary.return_value = "Summary 1"

        with patch(
            "sys.argv", ["main.py", "vid1", "-o", self.outfile, "-m", "gemini-test"]
        ):
            with patch("builtins.open", mock_open()):
                main.main()

        self.assertTrue(os.path.exists(self.outfile))
        df = pl.read_csv(self.outfile)
        self.assertEqual(len(df), 1)
        self.assertEqual(df[0, "URL"], "https://www.youtube.com/watch?v=vid1")
        self.assertEqual(df[0, "Summary Text gemini-test"], "Summary 1")
        self.assertIn("Summary File gemini-test", df.columns)
        self.assertIn("Transcript File youtube generated", df.columns)
        self.assertNotIn("Is Generated", df.columns)

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.get_video_details")
    @patch("youtube_to_docs.main.fetch_transcript")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("os.makedirs")
    def test_append_new_video(
        self,
        mock_makedirs,
        mock_gen_summary,
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
                "Transcript characters": [12],
                "Transcript File youtube generated": ["path1"],
                "Summary File gemini-test": ["spath1"],
                "Summary Text gemini-test": ["Summary 1"],
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
        mock_gen_summary.return_value = "Summary 2"

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
                "Transcript characters": [12],
                "Transcript File youtube generated": ["path1"],
                "Summary File gemini-test": ["spath1"],
                "Summary Text gemini-test": ["Summary 1"],
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

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.resolve_video_ids")
    @patch("youtube_to_docs.main.generate_summary")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_add_new_summary_column(
        self, mock_makedirs, mock_exists, mock_gen_summary, mock_resolve, mock_svc
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
                "Transcript characters": [12],
                "Transcript File youtube generated": ["transcript_vid1.txt"],
                "Summary File gemini-test": ["spath1"],
                "Summary Text gemini-test": ["Summary Gemini"],
            }
        )
        initial_data.write_csv(self.outfile)

        mock_resolve.return_value = ["vid1"]
        mock_gen_summary.return_value = "Summary Haiku"

        # Mock transcript file existence and reading
        def side_effect(path):
            if path == self.outfile:
                return True
            if "transcript_vid1.txt" in path:
                return True
            return False

        mock_exists.side_effect = side_effect

        with patch("sys.argv", ["main.py", "vid1", "-o", self.outfile, "-m", "haiku"]):
            with patch("builtins.open", mock_open(read_data="Transcript Content")):
                main.main()

        df = pl.read_csv(self.outfile)
        self.assertIn("Summary Text haiku", df.columns)
        self.assertIn("Summary File haiku", df.columns)
        self.assertIn("Summary Text gemini-test", df.columns)
        self.assertIn("Summary File gemini-test", df.columns)
        self.assertIn("Transcript File youtube generated", df.columns)
        self.assertEqual(
            df.filter(pl.col("URL").str.contains("vid1"))[0, "Summary Text haiku"],
            "Summary Haiku",
        )
        self.assertEqual(
            df.filter(pl.col("URL").str.contains("vid1"))[
                0, "Summary Text gemini-test"
            ],
            "Summary Gemini",
        )


if __name__ == "__main__":
    unittest.main()
