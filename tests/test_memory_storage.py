import os
import tempfile
import unittest

import polars as pl

from youtube_to_docs.storage import MemoryStorage


class TestMemoryStorage(unittest.TestCase):
    def setUp(self):
        self.storage = MemoryStorage()

    # -- exists --

    def test_exists_returns_false_for_unknown(self):
        self.assertFalse(self.storage.exists("no/such/path"))

    def test_exists_returns_true_after_write_text(self):
        self.storage.write_text("a.txt", "hello")
        self.assertTrue(self.storage.exists("a.txt"))

    def test_exists_returns_true_after_write_bytes(self):
        self.storage.write_bytes("b.bin", b"\x00\x01")
        self.assertTrue(self.storage.exists("b.bin"))

    # -- read / write text --

    def test_write_text_returns_path(self):
        result = self.storage.write_text("dir/file.md", "# Title")
        self.assertEqual(result, "dir/file.md")

    def test_read_text_roundtrip(self):
        self.storage.write_text("f.txt", "content")
        self.assertEqual(self.storage.read_text("f.txt"), "content")

    def test_read_text_raises_for_missing(self):
        with self.assertRaises(FileNotFoundError):
            self.storage.read_text("missing.txt")

    # -- read / write bytes --

    def test_write_bytes_returns_path(self):
        result = self.storage.write_bytes("img.png", b"\x89PNG")
        self.assertEqual(result, "img.png")

    def test_read_bytes_roundtrip(self):
        self.storage.write_bytes("img.png", b"\x89PNG")
        self.assertEqual(self.storage.read_bytes("img.png"), b"\x89PNG")

    def test_read_bytes_falls_back_to_text(self):
        self.storage.write_text("f.txt", "abc")
        self.assertEqual(self.storage.read_bytes("f.txt"), b"abc")

    def test_read_bytes_raises_for_missing(self):
        with self.assertRaises(FileNotFoundError):
            self.storage.read_bytes("missing.bin")

    # -- dataframe --

    def test_load_dataframe_returns_none_when_empty(self):
        self.assertIsNone(self.storage.load_dataframe("data.csv"))

    def test_save_and_load_dataframe_roundtrip(self):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = self.storage.save_dataframe(df, "out.csv")
        self.assertEqual(path, "out.csv")

        loaded = self.storage.load_dataframe("out.csv")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.shape, (2, 2))
        self.assertEqual(loaded["a"].to_list(), [1, 2])

    # -- ensure_directory (no-op) --

    def test_ensure_directory_does_not_raise(self):
        self.storage.ensure_directory("some/dir")

    # -- upload_file --

    def test_upload_file_reads_local_into_memory(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as f:
            f.write(b"audio-data")
            tmp_path = f.name
        try:
            result = self.storage.upload_file(tmp_path, "audio/vid.m4a")
            self.assertEqual(result, "audio/vid.m4a")
            self.assertTrue(self.storage.exists("audio/vid.m4a"))
            self.assertEqual(self.storage.read_bytes("audio/vid.m4a"), b"audio-data")
        finally:
            os.unlink(tmp_path)

    # -- get_full_path / get_name --

    def test_get_full_path_returns_same(self):
        self.assertEqual(self.storage.get_full_path("a/b.txt"), "a/b.txt")

    def test_get_name_returns_basename(self):
        self.assertEqual(self.storage.get_name("dir/sub/file.md"), "file.md")

    # -- get_local_file --

    def test_get_local_file_returns_none_for_missing(self):
        self.assertIsNone(self.storage.get_local_file("nope.txt"))

    def test_get_local_file_writes_bytes_to_download_dir(self):
        self.storage.write_bytes("audio/vid.m4a", b"audio-bytes")
        with tempfile.TemporaryDirectory() as dl_dir:
            local = self.storage.get_local_file("audio/vid.m4a", download_dir=dl_dir)
            self.assertIsNotNone(local)
            assert local is not None
            self.assertTrue(os.path.exists(local))
            with open(local, "rb") as f:
                self.assertEqual(f.read(), b"audio-bytes")

    def test_get_local_file_writes_text_to_temp(self):
        self.storage.write_text("f.txt", "hello")
        local = self.storage.get_local_file("f.txt")
        self.assertIsNotNone(local)
        assert local is not None
        try:
            with open(local, "rb") as f:
                self.assertEqual(f.read(), b"hello")
        finally:
            os.unlink(local)

    # -- get_artifacts --

    def test_get_artifacts_empty(self):
        self.assertEqual(self.storage.get_artifacts(), [])

    def test_get_artifacts_lists_all(self):
        self.storage.write_text("summary-files/s.md", "# Summary")
        self.storage.write_bytes("infographic-files/i.png", b"\x89PNG")
        artifacts = self.storage.get_artifacts()
        self.assertEqual(len(artifacts), 2)
        paths = {a["path"] for a in artifacts}
        self.assertEqual(paths, {"summary-files/s.md", "infographic-files/i.png"})
        for a in artifacts:
            self.assertIn("name", a)
            self.assertIn("directory", a)
            self.assertIn("size", a)

    # -- serve_artifact --

    def test_serve_artifact_text(self):
        self.storage.write_text("qa-files/qa.md", "## Q&A")
        content, media_type = self.storage.serve_artifact("qa-files/qa.md")
        self.assertEqual(content, b"## Q&A")
        self.assertIn("text/markdown", media_type)

    def test_serve_artifact_binary(self):
        self.storage.write_bytes("infographic-files/i.png", b"\x89PNG")
        content, media_type = self.storage.serve_artifact("infographic-files/i.png")
        self.assertEqual(content, b"\x89PNG")
        self.assertEqual(media_type, "image/png")

    def test_serve_artifact_raises_for_missing(self):
        with self.assertRaises(FileNotFoundError):
            self.storage.serve_artifact("nope.txt")


if __name__ == "__main__":
    unittest.main()
