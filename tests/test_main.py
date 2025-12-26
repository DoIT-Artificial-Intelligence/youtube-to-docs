import os
import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs import llms, transcript


class TestYoutubeToDocs(unittest.TestCase):
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict(
            os.environ,
            {
                "YOUTUBE_DATA_API_KEY": "fake_youtube_key",
                "GEMINI_API_KEY": "fake_gemini_key",
                "PROJECT_ID": "fake_project_id",
                "AWS_BEARER_TOKEN_BEDROCK": "fake_bedrock_token",
                "AZURE_FOUNDRY_ENDPOINT": "https://fake.openai.azure.com/",
                "AZURE_FOUNDRY_API_KEY": "fake_foundry_key",
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

    @patch("youtube_to_docs.transcript.ytt_api")
    def test_fetch_transcript(self, mock_ytt_api):
        mock_transcript_obj = MagicMock()
        mock_transcript_obj.to_raw_data.return_value = [
            {"text": "Hello"},
            {"text": "world"},
        ]
        mock_ytt_api.fetch.return_value = mock_transcript_obj

        result = transcript.fetch_transcript("vid1")
        self.assertIsNotNone(result)
        assert result is not None
        text, is_generated = result
        self.assertEqual(text, "Hello world")

    @patch("youtube_to_docs.transcript.ytt_api")
    def test_fetch_transcript_error(self, mock_ytt_api):
        mock_ytt_api.fetch.side_effect = Exception("Transcript disabled")
        result = transcript.fetch_transcript("vid1")
        self.assertIsNone(result)

    @patch("youtube_to_docs.llms.genai.Client")
    def test_generate_summary_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()
        mock_resp.text = "Gemini Summary"
        mock_client.models.generate_content.return_value = mock_resp

        summary = llms.generate_summary("gemini-pro", "transcript", "Title", "url")
        self.assertEqual(summary, "Gemini Summary")

    @patch("youtube_to_docs.llms.requests.post")
    @patch("google.auth.default")
    def test_generate_summary_vertex(self, mock_auth, mock_post):
        mock_creds = MagicMock()
        mock_creds.token = "fake_token"
        mock_creds.expired = False
        mock_auth.return_value = (mock_creds, "proj")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": [{"text": "Vertex Summary"}]}
        mock_post.return_value = mock_resp

        summary = llms.generate_summary(
            "vertex-claude-3-5", "transcript", "Title", "url"
        )
        self.assertEqual(summary, "Vertex Summary")

    @patch("youtube_to_docs.llms.requests.post")
    def test_generate_summary_bedrock(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "output": {"message": {"content": [{"text": "Bedrock Summary"}]}}
        }
        mock_post.return_value = mock_resp

        summary = llms.generate_summary(
            "bedrock-claude-3-5", "transcript", "Title", "url"
        )
        self.assertEqual(summary, "Bedrock Summary")

    @patch("youtube_to_docs.llms.OpenAI")
    def test_generate_summary_foundry(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Foundry Summary"
        mock_client.chat.completions.create.return_value = mock_completion

        summary = llms.generate_summary("foundry-gpt-4", "transcript", "Title", "url")
        self.assertEqual(summary, "Foundry Summary")


if __name__ == "__main__":
    unittest.main()
