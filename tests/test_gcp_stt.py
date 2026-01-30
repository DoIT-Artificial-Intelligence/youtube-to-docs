import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from youtube_to_docs.llms import generate_transcript


class TestGCPSTT(unittest.TestCase):
    @patch("youtube_to_docs.llms._transcribe_gcp")
    def test_gcp_dispatch(self, mock_transcribe):
        """Test that gcp- models are dispatched to _transcribe_gcp."""
        mock_transcribe.return_value = ("transcript", "srt_content", 0, 0)

        generate_transcript("gcp-chirp3", "audio.m4a", "http://url")

        mock_transcribe.assert_called_once()
        args, _ = mock_transcribe.call_args
        self.assertEqual(args[0], "gcp-chirp3")

    @patch.dict(
        os.environ,
        {"GOOGLE_CLOUD_PROJECT": "test-project", "YTD_GCS_BUCKET_NAME": "test-bucket"},
    )
    @patch("uuid.uuid4", return_value="1234")
    def test_transcribe_gcp_imports(self, mock_uuid):
        """Test the logic inside _transcribe_gcp with mocked cloud libs."""

        # Mocks for Google Cloud libraries
        mock_speech_module = MagicMock()
        mock_storage_module = MagicMock()
        mock_types_module = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "google.cloud.speech_v2": mock_speech_module,
                "google.cloud.storage": mock_storage_module,
                "google.cloud.speech_v2.types": mock_types_module,
                # We don't mock 'google.cloud' parent to avoid namespace confusion,
                # let the specific submodules be returned by import.
            },
        ):
            # Re-import to ensure we are testing logic using these mocks?
            # Actually code calls `from google.cloud import speech_v2`
            # If `google.cloud.speech_v2` is in sys.modules, that works.

            # Need to access the private function
            from youtube_to_docs import llms

            # Mock Client
            mock_client_instance = MagicMock()
            mock_speech_module.SpeechClient.return_value = mock_client_instance

            # Mock Operation / Response
            mock_op = MagicMock()
            mock_client_instance.batch_recognize.return_value = mock_op

            mock_result = MagicMock()
            mock_op.result.return_value = mock_result

            # Mock Storage for transcript JSON
            mock_storage_client = MagicMock()
            mock_storage_module.Client.return_value = mock_storage_client
            mock_bucket = MagicMock()
            mock_storage_client.bucket.return_value = mock_bucket
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob

            # Mock Batch Result
            mock_batch_result = MagicMock()
            mock_batch_result.uri = (
                "gs://test-bucket/temp/transcripts/ytd_audio_1234_transcript.json"
            )
            mock_batch_result.error = None  # Ensure no error is triggered
            mock_batch_result.inline_result = None  # Ensure GCS fallback

            # Setup transcript JSON content
            mock_json_content = """
            {
              "results": [
                {
                  "alternatives": [
                    {
                      "transcript": "Test Transcript"
                    }
                  ]
                }
              ]
            }
            """
            mock_blob.download_as_text.return_value = mock_json_content

            # Setup results dict with EXPECTED key
            mock_result.results = {
                "gs://test-bucket/temp/ytd_audio_1234.m4a": mock_batch_result
            }

            # Run
            transcript, srt_content, _, _ = llms._transcribe_gcp(
                "gcp-chirp3", "audio.m4a", "http://url"
            )

            self.assertEqual(transcript, "Test Transcript")

            self.assertEqual(transcript, "Test Transcript")

    @patch.dict(
        os.environ,
        {"GOOGLE_CLOUD_PROJECT": "test-project", "YTD_GCS_BUCKET_NAME": "test-bucket"},
    )
    @patch("uuid.uuid4", return_value="5678")
    def test_transcribe_gcp_inline(self, mock_uuid):
        """Test the inline logic inside _transcribe_gcp for short videos."""

        # Mocks for Google Cloud libraries
        mock_speech_module = MagicMock()
        mock_storage_module = MagicMock()
        mock_types_module = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "google.cloud.speech_v2": mock_speech_module,
                "google.cloud.storage": mock_storage_module,
                "google.cloud.speech_v2.types": mock_types_module,
            },
        ):
            from youtube_to_docs import llms

            # Mock Client
            mock_client_instance = MagicMock()
            mock_speech_module.SpeechClient.return_value = mock_client_instance

            # Mock Operation / Response
            mock_op = MagicMock()
            mock_client_instance.batch_recognize.return_value = mock_op

            mock_result = MagicMock()
            mock_op.result.return_value = mock_result

            # Inline Result Mock
            mock_batch_result = MagicMock()
            mock_batch_result.error = None

            # Setup the nested structure for inline_result.transcript.results
            mock_inline_transcript = MagicMock()
            mock_batch_result.inline_result.transcript = mock_inline_transcript

            # Mock the transcript result object
            # The code uses: result.alternatives[0].transcript
            mock_transcript_item = MagicMock()
            mock_alt = MagicMock()
            mock_alt.transcript = "Inline Transcript"
            # Handles both obj and dict, here we mock obj behavior
            mock_transcript_item.alternatives = [mock_alt]

            mock_inline_transcript.results = [mock_transcript_item]

            # Add to response results map
            # key is gcs_uri
            gcs_uri = "gs://test-bucket/temp/ytd_audio_5678.m4a"
            mock_result.results = {gcs_uri: mock_batch_result}

            # Mock Storage (for upload - still happens)
            mock_storage_client = MagicMock()
            mock_storage_module.Client.return_value = mock_storage_client
            mock_bucket = MagicMock()
            mock_storage_client.bucket.return_value = mock_bucket
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob

            # Run with short duration
            transcript, srt_content, _, _ = llms._transcribe_gcp(
                "gcp-chirp3", "audio.m4a", "http://url", duration_seconds=100.0
            )

            self.assertEqual(transcript, "Inline Transcript")
            # Verify NO download_as_text called
            mock_blob.download_as_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
