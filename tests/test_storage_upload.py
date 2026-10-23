import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs.storage import (
    GoogleDriveStorage,
    _pandoc_format,
    _pandoc_input,
)


class DriveTestCase(unittest.TestCase):
    @patch("youtube_to_docs.storage.build")
    @patch("google.oauth2.credentials.Credentials")
    @patch("youtube_to_docs.storage.Path")
    def setUp(self, mock_path, mock_creds, mock_build):
        self.mock_service = MagicMock()
        mock_build.return_value = self.mock_service

        # Mock Path.home() and other path ops
        mock_path_obj = MagicMock()
        mock_path.home.return_value = mock_path_obj
        mock_path_obj.__truediv__.return_value = mock_path_obj
        mock_path_obj.exists.return_value = True

        self.storage = GoogleDriveStorage("workspace")
        # Mock _resolve_root_folder_id to avoid API call in init
        self.storage.root_folder_id = "root_id"


class TestGoogleDriveStorage(DriveTestCase):
    @patch("youtube_to_docs.storage.MediaFileUpload")
    def test_upload_file_uses_media_file_upload(self, mock_media_file_upload):
        # Setup mocks
        with (
            patch.object(self.storage, "_get_parent_id", return_value="parent_id"),
            patch.object(self.storage, "_get_file_id", return_value=None),
        ):
            local_path = "/tmp/local_file.mp4"
            target_path = "video-files/remote_file.mp4"

            # Execute
            self.storage.upload_file(local_path, target_path, content_type="video/mp4")

        # Assert MediaFileUpload was used, not MediaIoBaseUpload
        mock_media_file_upload.assert_called_once_with(
            local_path, mimetype="video/mp4", resumable=True
        )

        # Assert service create call
        self.mock_service.files().create.assert_called_once()


class TestGoogleDrivePlainTextFiles(DriveTestCase):
    """Machine-readable text files are plain Drive files, not Google Docs."""

    DOC = {"id": "doc_id", "mimeType": "application/vnd.google-apps.document"}
    FILE = {"id": "file_id", "mimeType": "text/plain"}

    def _write(self, path, existing=None):
        files = self.mock_service.files.return_value
        files.create.return_value.execute.return_value = {
            "id": "new_id",
            "webViewLink": "https://drive.google.com/file/d/new_id/view",
        }
        with (
            patch.object(self.storage, "_get_parent_id", return_value="parent_id"),
            patch.object(self.storage, "_get_file_metadata", return_value=existing),
            patch("youtube_to_docs.storage.MediaIoBaseUpload") as mock_media,
        ):
            link = self.storage.write_text(path, "1\n00:00:01,000 --> 00:00:02,000\nHi")
        return link, files, mock_media

    def test_srt_is_uploaded_as_plain_text(self):
        link, files, mock_media = self._write("srt-files/x.srt")
        self.assertEqual(link, "https://drive.google.com/file/d/new_id/view")
        body = files.create.call_args.kwargs["body"]
        self.assertNotIn("mimeType", body)
        self.assertEqual(mock_media.call_args.kwargs["mimetype"], "text/plain")
        self.assertEqual(
            mock_media.call_args.args[0].getvalue(),
            b"1\n00:00:01,000 --> 00:00:02,000\nHi",
        )

    def test_old_google_doc_is_replaced(self):
        _, files, _ = self._write("vtt-files/x.vtt", existing=self.DOC)
        files.update.assert_called_once_with(fileId="doc_id", body={"trashed": True})
        files.create.assert_called_once()

    def test_old_google_doc_track_counts_as_missing(self):
        for metadata, expected in ((self.DOC, False), (self.FILE, True)):
            with patch.object(
                self.storage, "_get_file_metadata", return_value=metadata
            ):
                self.assertIs(self.storage.exists("vtt-files/x.vtt"), expected)
                self.assertTrue(self.storage.exists("summary-files/x.md"))

    def test_google_doc_is_exported_not_downloaded(self):
        with patch.object(self.storage, "read_text", return_value="text") as read:
            data = self.storage.read_bytes(
                "https://docs.google.com/document/d/abc/edit?usp=drivesdk"
            )
        self.assertEqual(data, b"text")
        read.assert_called_once()
        self.mock_service.files.return_value.get_media.assert_not_called()


class TestPandocFormat(unittest.TestCase):
    def test_text_files_keep_line_breaks(self):
        self.assertEqual(_pandoc_format("a - tags.txt"), "markdown+hard_line_breaks")
        self.assertEqual(_pandoc_format("a - summary.md"), "md")

    def test_text_files_keep_blank_lines(self):
        self.assertEqual(
            _pandoc_input("d.txt", "One.\n\n\nTwo\nthree\n"),
            "One.\n\n\u00a0\n\nTwo\nthree",
        )
        self.assertEqual(_pandoc_input("s.md", "One.\n\nTwo"), "One.\n\nTwo")


if __name__ == "__main__":
    unittest.main()
