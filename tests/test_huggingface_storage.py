import unittest
from unittest.mock import MagicMock, patch

import polars as pl


class TestHuggingFaceStorage(unittest.TestCase):
    def setUp(self):
        self.api_patcher = patch("huggingface_hub.HfApi")
        mock_hfapi_cls = self.api_patcher.start()
        self.addCleanup(self.api_patcher.stop)

        self.mock_api = MagicMock()
        self.mock_api.whoami.return_value = {"name": "tester"}
        mock_hfapi_cls.return_value = self.mock_api

        self.env_patcher = patch.dict("os.environ", {"HF_TOKEN": "secret-token"})
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

        from youtube_to_docs.storage import HuggingFaceStorage

        self.HuggingFaceStorage = HuggingFaceStorage
        self.storage = HuggingFaceStorage("Code for America Summit 2026 Recap")

    def test_requires_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                self.HuggingFaceStorage("some-dataset")

    def test_requires_dataset(self):
        with self.assertRaises(ValueError):
            self.HuggingFaceStorage("")

    def test_slugify_and_namespace(self):
        # A plain name is slugified and namespaced under the authenticated user.
        self.assertEqual(
            self.storage.repo_id, "tester/Code-for-America-Summit-2026-Recap"
        )

    def test_full_repo_id_preserved(self):
        storage = self.HuggingFaceStorage("someone/my-dataset")
        self.assertEqual(storage.repo_id, "someone/my-dataset")

    def test_create_repo_called(self):
        self.mock_api.create_repo.assert_called_with(
            repo_id="tester/Code-for-America-Summit-2026-Recap",
            repo_type="dataset",
            exist_ok=True,
            token="secret-token",
        )

    def test_write_text_stages_and_returns_url(self):
        import os

        url = self.storage.write_text("summary-files/foo.md", "hello")
        # Writes are staged locally, not committed one-per-file.
        self.mock_api.upload_file.assert_not_called()
        staged = os.path.join(self.storage._staging_dir, "summary-files/foo.md")
        self.assertTrue(os.path.exists(staged))
        with open(staged, "rb") as f:
            self.assertEqual(f.read(), b"hello")
        self.assertTrue(
            url.startswith(
                "https://huggingface.co/datasets/"
                "tester/Code-for-America-Summit-2026-Recap/blob/main/"
            )
        )

    def test_read_back_staged_write(self):
        # A staged write is readable without hitting the Hub.
        self.storage.write_text("summary-files/foo.md", "hello")
        self.assertEqual(self.storage.read_text("summary-files/foo.md"), "hello")

    def test_write_caches_existence(self):
        self.storage.write_bytes("infographic-files/foo.png", b"\x89PNG")
        # exists() should short-circuit via the cache without an API call.
        self.assertTrue(self.storage.exists("infographic-files/foo.png"))
        self.mock_api.file_exists.assert_not_called()

    def test_save_dataframe_flushes_folder(self):
        df = pl.DataFrame({"a": [1, 2]})
        url = self.storage.save_dataframe(df, "youtube-docs.csv")
        # save_dataframe is the checkpoint: it commits the staged folder once.
        self.mock_api.upload_folder.assert_called_once()
        kwargs = self.mock_api.upload_folder.call_args.kwargs
        self.assertEqual(kwargs["folder_path"], self.storage._staging_dir)
        self.assertEqual(kwargs["repo_type"], "dataset")
        self.assertIn("youtube-docs.csv", url)

    def test_flush_noop_when_clean(self):
        # Nothing staged -> no commit.
        self.storage.flush()
        self.mock_api.upload_folder.assert_not_called()

    def test_flush_batches_multiple_writes_into_one_commit(self):
        self.storage.write_text("summary-files/a.md", "a")
        self.storage.write_text("summary-files/b.md", "b")
        self.storage.write_bytes("infographic-files/c.png", b"\x89PNG")
        self.storage.flush()
        # All three artifacts go up in a single commit.
        self.mock_api.upload_folder.assert_called_once()
        # A second flush with nothing new staged is a no-op.
        self.storage.flush()
        self.mock_api.upload_folder.assert_called_once()

    def test_get_full_path_returns_url(self):
        path = self.storage.get_full_path("srt-files/foo.srt")
        self.assertIn("srt-files/foo.srt", path)
        self.assertTrue(path.startswith("https://huggingface.co/datasets/"))

    def test_get_full_path_passthrough_for_url(self):
        url = "https://huggingface.co/datasets/tester/x/blob/main/a.md"
        self.assertEqual(self.storage.get_full_path(url), url)

    def test_get_name(self):
        self.assertEqual(self.storage.get_name("summary-files/foo.md"), "foo.md")
        url = "https://huggingface.co/datasets/tester/x/blob/main/dir/bar.png"
        self.assertEqual(self.storage.get_name(url), "bar.png")

    def test_load_dataframe_missing_returns_none(self):
        self.mock_api.file_exists.return_value = False
        self.assertIsNone(self.storage.load_dataframe("youtube-docs.csv"))


if __name__ == "__main__":
    unittest.main()
