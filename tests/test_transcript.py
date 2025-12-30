import os
import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs import transcript


class TestTranscript(unittest.TestCase):
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict(
            os.environ,
            {
                "YOUTUBE_DATA_API_KEY": "fake_youtube_key",
            },
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("youtube_to_docs.transcript.build")
    def test_get_youtube_service(self, mock_build):
        service = transcript.get_youtube_service()
        self.assertIsNotNone(service)
        mock_build.assert_called_with("youtube", "v3", developerKey="fake_youtube_key")

    def test_get_youtube_service_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            service = transcript.get_youtube_service()
            self.assertIsNone(service)

    def test_resolve_video_ids_single(self):
        ids = transcript.resolve_video_ids("KuPc06JgI_A", None)
        self.assertEqual(ids, ["KuPc06JgI_A"])

    def test_resolve_video_ids_list(self):
        ids = transcript.resolve_video_ids("KuPc06JgI_A,GalhDyf3F8g", None)
        self.assertEqual(ids, ["KuPc06JgI_A", "GalhDyf3F8g"])

    def test_resolve_video_ids_playlist_no_service(self):
        with self.assertRaises(SystemExit):
            transcript.resolve_video_ids("PL8ZxoInteClyHaiReuOHpv6Z4SPrXtYtW", None)

    @patch("youtube_to_docs.transcript.build")
    def test_resolve_video_ids_playlist(self, mock_build):
        mock_service = MagicMock()
        mock_request = MagicMock()
        mock_response = {
            "items": [
                {"contentDetails": {"videoId": "vid1"}},
                {"contentDetails": {"videoId": "vid2"}},
            ]
        }
        mock_request.execute.return_value = mock_response
        # Mock list_next to return None to stop iteration
        mock_service.playlistItems().list.return_value = mock_request
        mock_service.playlistItems().list_next.return_value = None

        ids = transcript.resolve_video_ids("PL123", mock_service)
        self.assertEqual(ids, ["vid1", "vid2"])

    @patch("youtube_to_docs.transcript.build")
    def test_resolve_video_ids_channel_handle(self, mock_build):
        mock_service = MagicMock()

        # Mock channel list response
        mock_channel_req = MagicMock()
        mock_channel_resp = {
            "items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU123"}}}]
        }
        mock_channel_req.execute.return_value = mock_channel_resp
        mock_service.channels().list.return_value = mock_channel_req

        # Mock playlist items response (since it calls resolve_video_ids internally
        # with the playlist ID)
        mock_playlist_req = MagicMock()
        mock_playlist_resp = {
            "items": [{"contentDetails": {"videoId": "vid_from_channel"}}]
        }
        mock_playlist_req.execute.return_value = mock_playlist_resp
        mock_service.playlistItems().list.return_value = mock_playlist_req
        mock_service.playlistItems().list_next.return_value = None

        ids = transcript.resolve_video_ids("@channel", mock_service)
        self.assertEqual(ids, ["vid_from_channel"])

    def test_get_video_details_none(self):
        details = transcript.get_video_details("vid1", None)
        self.assertEqual(
            details, ("", "", "", "", "", "", "https://www.youtube.com/watch?v=vid1")
        )

    def test_get_video_details_success(self):
        mock_service = MagicMock()
        mock_req = MagicMock()
        mock_resp = {
            "items": [
                {
                    "snippet": {
                        "title": "Test Video",
                        "description": "Desc",
                        "publishedAt": "2023-01-01",
                        "channelTitle": "Test Channel",
                        "tags": ["tag1", "tag2"],
                    },
                    "contentDetails": {"duration": "PT1M10S"},
                }
            ]
        }
        mock_req.execute.return_value = mock_resp
        mock_service.videos().list.return_value = mock_req

        details = transcript.get_video_details("vid1", mock_service)
        self.assertIsNotNone(details)
        assert details is not None
        self.assertEqual(details[0], "Test Video")
        self.assertEqual(details[5], "0:01:10")  # Duration

    @patch("youtube_to_docs.transcript.YouTubeTranscriptApi.list")
    def test_fetch_transcript(self, mock_list):
        mock_transcript_list = MagicMock()
        mock_transcript_obj = MagicMock()

        snippet1 = MagicMock()
        snippet1.text = "Hello"
        snippet2 = MagicMock()
        snippet2.text = "world"

        mock_transcript_obj.fetch.return_value = [snippet1, snippet2]
        mock_transcript_obj.is_generated = False
        mock_transcript_obj.translation_languages = []

        # Setup the chain:
        # list -> find_manually_created_transcript -> mock_transcript_obj
        mock_list.return_value = mock_transcript_list
        mock_transcript_list.find_manually_created_transcript.return_value = (
            mock_transcript_obj
        )

        result = transcript.fetch_transcript("vid1")
        self.assertIsNotNone(result)
        assert result is not None
        text, is_generated = result
        self.assertEqual(text, "Hello world")
        self.assertFalse(is_generated)

    @patch("youtube_to_docs.transcript.YouTubeTranscriptApi.list")
    def test_fetch_transcript_error(self, mock_list):
        mock_list.side_effect = Exception("Transcript disabled")
        result = transcript.fetch_transcript("vid1")
        self.assertIsNone(result)

    @patch("youtube_to_docs.transcript.YouTubeTranscriptApi.list")
    def test_fetch_transcript_translation(self, mock_list_transcripts):
        mock_transcript_list = MagicMock()
        mock_en_transcript = MagicMock()
        mock_es_transcript = MagicMock()

        mock_en_transcript.translate.return_value = mock_es_transcript

        snippet1 = MagicMock()
        snippet1.text = "Hola"
        snippet2 = MagicMock()
        snippet2.text = "mundo"

        mock_es_transcript.fetch.return_value = [snippet1, snippet2]
        mock_es_transcript.is_generated = False
        mock_es_transcript.translation_languages = [
            {"language": "Spanish", "language_code": "es"}
        ]

        mock_list_transcripts.return_value = mock_transcript_list
        # fail find 'es'
        mock_transcript_list.find_manually_created_transcript.side_effect = [
            Exception("Not found es"),  # 1st call for 'es'
            mock_en_transcript,  # 2nd call for 'en'
        ]
        mock_transcript_list.find_generated_transcript.side_effect = Exception(
            "Not found generated"
        )

        result = transcript.fetch_transcript("vid1", language="es")
        self.assertIsNotNone(result)
        assert result is not None
        text, _ = result
        self.assertEqual(text, "Hola mundo")
        mock_en_transcript.translate.assert_called_with("es")


if __name__ == "__main__":
    unittest.main()
