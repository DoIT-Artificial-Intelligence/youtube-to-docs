import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs.storage import GoogleDriveStorage


class TestGoogleDriveStorage(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
