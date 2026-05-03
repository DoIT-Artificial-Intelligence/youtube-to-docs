import os
import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from youtube_to_docs.providers import BaseProvider, TTSProvider
from youtube_to_docs.tts import (
    generate_speech,
    generate_speech_aws_polly,
    generate_speech_gcp,
    is_aws_polly_model,
    is_gcp_tts_model,
    parse_tts_arg,
    process_tts,
)


class TestTTS(unittest.TestCase):
    def test_parse_tts_arg(self):
        # Test with hyphen
        model, voice = parse_tts_arg("gemini-2.5-flash-preview-tts-Kore")
        self.assertEqual(model, "gemini-2.5-flash-preview-tts")
        self.assertEqual(voice, "Kore")

        # Test with simple hyphen
        model, voice = parse_tts_arg("model-voice")
        self.assertEqual(model, "model")
        self.assertEqual(voice, "voice")

        # Test without hyphen (default)
        model, voice = parse_tts_arg("singlemodelname")
        self.assertEqual(model, "singlemodelname")
        self.assertEqual(voice, "Kore")

    @patch("google.genai", create=True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"})
    def test_generate_speech_success(self, mock_genai):
        # Setup mock client and response
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"fake_audio_data"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        mock_client.models.generate_content.return_value = mock_response

        # Execute
        audio_data, rate = generate_speech("Hello world", "model", "voice")

        # Verify
        self.assertEqual(audio_data, b"fake_audio_data")
        self.assertEqual(rate, 24000)
        mock_client.models.generate_content.assert_called_once()

    @patch("google.genai", create=True)
    @patch.dict(os.environ, {}, clear=True)
    def test_generate_speech_no_api_key(self, mock_genai):
        # Execute
        audio_data, rate = generate_speech("Hello world", "model", "voice")

        # Verify
        self.assertEqual(audio_data, b"")
        self.assertEqual(rate, 0)
        mock_genai.Client.assert_not_called()

    @patch("google.genai", create=True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"})
    def test_generate_speech_api_error(self, mock_genai):
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("API Error")

        # Execute
        audio_data, rate = generate_speech("Hello world", "model", "voice")

        # Verify
        self.assertEqual(audio_data, b"")
        self.assertEqual(rate, 0)

    @patch("youtube_to_docs.providers.get_provider")
    def test_process_tts(
        self,
        mock_get_provider,
    ):
        # Setup DataFrame
        df = pl.DataFrame(
            {
                "Summary File 1": ["/path/to/summary1.md", "/path/to/summary2.md"],
                "Other Col": [1, 2],
            }
        )

        # Setup Mock Storage
        mock_storage = MagicMock()

        # storage.exists behavior:
        # 1. target_path check for row 1 -> False (generate)
        # 2. target_path check for row 2 -> True (skip)
        # (It checks if summary exists first? process_tts logic:
        #  for row:
        #   target_path = ...
        #   if storage.exists(target_path): skip
        #   ...
        #   read_text
        # )

        # Let's map side_effect based on path
        def exists_side_effect(path):
            if path.endswith(".md"):
                return True
            if path.endswith("summary2 - tts-arg.wav"):
                return True
            return False

        mock_storage.exists.side_effect = exists_side_effect
        mock_storage.read_text.return_value = "Summary text"
        mock_storage.write_bytes.return_value = "/saved/path.wav"
        mock_storage.get_full_path.return_value = "/full/path/summary2 - tts-arg.wav"

        from typing import Any

        provider_instance: Any = MockTTSProvider("tts")
        provider_instance.generate_speech = MagicMock(return_value=(b"1234", 16000))
        mock_get_provider.return_value = provider_instance

        # Execute
        updated_df = process_tts(df, "tts-arg", mock_storage, "/tmp")

        # Verify
        self.assertIn("Summary Audio File 1 tts-arg File", updated_df.columns)

        # Check generate_speech called once (for the first row only)
        provider_instance.generate_speech.assert_called_once_with(
            "Summary text", "arg", "en-US"
        )

        # Check storage writes
        mock_storage.write_bytes.assert_called_once()
        args, _ = mock_storage.write_bytes.call_args
        self.assertTrue(args[0].endswith("summary1 - tts-arg.wav"))

        # Verify content is a valid WAV container
        wav_content = args[1]
        self.assertTrue(wav_content.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav_content)

        # Check DataFrame content
        new_col = updated_df["Summary Audio File 1 tts-arg File"]
        self.assertEqual(new_col[0], "/saved/path.wav")
        # For existing file, process_tts appends storage.get_full_path(target_path)
        # if available
        self.assertEqual(new_col[1], "/full/path/summary2 - tts-arg.wav")

    @patch("youtube_to_docs.providers.get_provider")
    def test_process_tts_with_language(
        self,
        mock_get_provider,
    ):
        # Setup DataFrame with language-suffixed column
        df = pl.DataFrame(
            {
                "Summary File (es)": ["/path/to/summary_es.md"],
            }
        )

        # Mock Storage
        mock_storage = MagicMock()

        def exists_side_effect(path):
            if path.endswith(".md"):
                return True
            return False

        mock_storage.exists.side_effect = exists_side_effect
        mock_storage.read_text.return_value = "Texto resumen"
        mock_storage.write_bytes.return_value = "/saved/es.wav"

        from typing import Any

        provider_instance: Any = MockTTSProvider("tts")
        provider_instance.generate_speech = MagicMock(return_value=(b"1234", 16000))
        mock_get_provider.return_value = provider_instance

        # Execute
        process_tts(df, "tts-arg", mock_storage, "/tmp")

        # Verify generate_speech called with 'es-US'
        provider_instance.generate_speech.assert_called_once_with(
            "Texto resumen", "arg", "es-US"
        )

        # Verify write was called
        mock_storage.write_bytes.assert_called_once()
        args, _ = mock_storage.write_bytes.call_args
        self.assertTrue(args[1].startswith(b"RIFF"))


class TestGCPTTS(unittest.TestCase):
    """Tests for GCP Chirp3 TTS functionality."""

    def test_parse_tts_arg_gcp_simple(self):
        """Test parsing gcp-chirp3 without a voice (defaults to Kore)."""
        model, voice = parse_tts_arg("gcp-chirp3")
        self.assertEqual(model, "gcp-chirp3")
        self.assertEqual(voice, "Kore")

    def test_parse_tts_arg_gcp_with_voice(self):
        """Test parsing gcp-chirp3 with explicit voice."""
        model, voice = parse_tts_arg("gcp-chirp3-Kore")
        self.assertEqual(model, "gcp-chirp3")
        self.assertEqual(voice, "Kore")

        model, voice = parse_tts_arg("gcp-chirp3-Orus")
        self.assertEqual(model, "gcp-chirp3")
        self.assertEqual(voice, "Orus")

    def test_is_gcp_tts_model(self):
        """Test GCP model detection helper."""
        self.assertTrue(is_gcp_tts_model("gcp-chirp3"))
        self.assertTrue(is_gcp_tts_model("gcp-chirp3-Kore"))
        self.assertFalse(is_gcp_tts_model("gemini-2.5-flash-preview-tts"))
        self.assertFalse(is_gcp_tts_model("some-other-model"))

    @patch("youtube_to_docs.tts.texttospeech", create=True)
    def test_generate_speech_gcp_success(self, mock_tts_module):
        """Test successful GCP TTS generation."""
        # Import the function to test with mocked module
        with patch.dict("sys.modules", {"google.cloud.texttospeech": mock_tts_module}):
            # Setup mock client and response
            mock_client = MagicMock()
            mock_tts_module.TextToSpeechClient.return_value = mock_client
            mock_tts_module.SynthesisInput.return_value = MagicMock()
            mock_tts_module.VoiceSelectionParams.return_value = MagicMock()
            mock_tts_module.AudioConfig.return_value = MagicMock()
            mock_tts_module.AudioEncoding.LINEAR16 = "LINEAR16"

            mock_response = MagicMock()
            mock_response.audio_content = b"fake_pcm_audio_data"
            mock_client.synthesize_speech.return_value = mock_response

            # Execute
            audio_data, rate = generate_speech_gcp("Hello world", "Kore", "en-US")

            # Verify
            self.assertEqual(audio_data, b"fake_pcm_audio_data")
            self.assertEqual(rate, 24000)

    def test_generate_speech_gcp_import_error(self):
        """Test GCP TTS when google-cloud-texttospeech is not installed."""
        with patch.dict("sys.modules", {"google.cloud": None}):
            # This should catch the import error and return empty bytes
            audio_data, rate = generate_speech_gcp("Hello", "Kore", "en-US")
            self.assertEqual(audio_data, b"")
            self.assertEqual(rate, 0)
            # Note: Due to how the import works, this may not trigger cleanly
            # in all test environments

    @patch("youtube_to_docs.providers.get_provider")
    def test_process_tts_with_gcp_model(self, mock_get_provider):
        """Test process_tts routes to GCP TTS for gcp-chirp3 models."""
        df = pl.DataFrame(
            {
                "Summary File 1": ["/path/to/summary.md"],
            }
        )

        mock_storage = MagicMock()

        def exists_side_effect(path):
            if path.endswith(".md"):
                return True
            return False

        mock_storage.exists.side_effect = exists_side_effect
        mock_storage.read_text.return_value = "Summary text"
        mock_storage.write_bytes.return_value = "/saved/path.wav"

        from typing import Any

        provider_instance: Any = MockGCPTTSProvider("gcp-chirp3")
        provider_instance.generate_speech = MagicMock(return_value=(b"1234", 24000))
        mock_get_provider.return_value = provider_instance

        # Execute with GCP model
        updated_df = process_tts(df, "gcp-chirp3-Kore", mock_storage, "/tmp")

        # Verify GCP function was called
        provider_instance.generate_speech.assert_called_once_with(
            "Summary text", "Kore", "en-US"
        )

        # Verify new column was created
        self.assertIn("Summary Audio File 1 gcp-chirp3-Kore File", updated_df.columns)

        # Verify WAV file was written
        mock_storage.write_bytes.assert_called_once()
        args, _ = mock_storage.write_bytes.call_args
        self.assertTrue(args[0].endswith(".wav"))
        self.assertTrue(args[1].startswith(b"RIFF"))


class TestAWSPolly(unittest.TestCase):
    """Tests for AWS Polly TTS functionality."""

    def test_parse_tts_arg_aws_simple(self):
        """Test parsing aws-polly without a voice (defaults to Ruth)."""
        model, voice = parse_tts_arg("aws-polly")
        self.assertEqual(model, "aws-polly")
        self.assertEqual(voice, "Ruth")

    def test_parse_tts_arg_aws_with_voice(self):
        """Test parsing aws-polly with explicit voice."""
        model, voice = parse_tts_arg("aws-polly-Joanna")
        self.assertEqual(model, "aws-polly")
        self.assertEqual(voice, "Joanna")

    def test_is_aws_polly_model(self):
        """Test AWS Polly model detection helper."""
        self.assertTrue(is_aws_polly_model("aws-polly"))
        self.assertTrue(is_aws_polly_model("aws-polly-Ruth"))
        self.assertFalse(is_aws_polly_model("gcp-chirp3"))
        self.assertFalse(is_aws_polly_model("gemini-2.5-flash-preview-tts"))

    @patch("youtube_to_docs.tts.boto3", create=True)
    def test_generate_speech_aws_polly_success(self, mock_boto3):
        """Test successful AWS Polly TTS generation."""
        # Setup mock client and response
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_response = {"AudioStream": MagicMock()}
        # Simulate stream read
        mock_response["AudioStream"].__enter__.return_value = MagicMock()
        mock_response[
            "AudioStream"
        ].__enter__.return_value.read.return_value = b"fake_pcm_data"

        mock_client.synthesize_speech.return_value = mock_response

        # Need to ensure import inside function uses the mock
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            # Execute
            audio_data, rate = generate_speech_aws_polly("Hello world", "Ruth")

        # Verify
        self.assertEqual(audio_data, b"fake_pcm_data")
        self.assertEqual(rate, 16000)
        mock_client.synthesize_speech.assert_called_once()
        args, kwargs = mock_client.synthesize_speech.call_args
        self.assertEqual(kwargs["Text"], "Hello world")
        self.assertEqual(kwargs["VoiceId"], "Ruth")
        self.assertEqual(kwargs["Engine"], "long-form")
        self.assertEqual(kwargs["OutputFormat"], "pcm")

    def test_generate_speech_aws_polly_import_error(self):
        """Test AWS Polly TTS when boto3 is not installed."""
        with patch.dict("sys.modules", {"boto3": None}):
            audio_data, rate = generate_speech_aws_polly("Hello", "Ruth")
            self.assertEqual(audio_data, b"")
            self.assertEqual(rate, 0)

    @patch("youtube_to_docs.providers.get_provider")
    def test_process_tts_with_aws_polly_model(self, mock_get_provider):
        """Test process_tts routes to AWS Polly for aws-polly models."""
        df = pl.DataFrame(
            {
                "Summary File 1": ["/path/to/summary.md"],
            }
        )

        mock_storage = MagicMock()

        def exists_side_effect(path):
            if path.endswith(".md"):
                return True
            return False

        mock_storage.exists.side_effect = exists_side_effect
        mock_storage.read_text.return_value = "Summary text"
        mock_storage.write_bytes.return_value = "/saved/path.wav"

        from typing import Any

        provider_instance: Any = MockAWSTTSProvider("aws-polly")
        provider_instance.generate_speech = MagicMock(return_value=(b"1234", 16000))
        mock_get_provider.return_value = provider_instance

        # Execute with AWS Polly model
        updated_df = process_tts(df, "aws-polly", mock_storage, "/tmp")

        # Verify AWS function was called
        # Note: process_tts now passes (text, voice_name, lang_code)
        provider_instance.generate_speech.assert_called_once_with(
            "Summary text", "Ruth", "en-US"
        )

        # Verify new column was created
        self.assertIn("Summary Audio File 1 aws-polly File", updated_df.columns)

        # Verify WAV file was written
        mock_storage.write_bytes.assert_called_once()
        args, _ = mock_storage.write_bytes.call_args
        self.assertTrue(args[0].endswith(".wav"))
        self.assertTrue(args[1].startswith(b"RIFF"))


class MockTTSProvider(BaseProvider, TTSProvider):
    def generate_speech(self, text, voice, language_code=None, **kwargs):
        return b"audio", 16000


class MockGCPTTSProvider(BaseProvider, TTSProvider):
    def generate_speech(self, text, voice, language_code=None, **kwargs):
        return b"audio", 24000


class MockAWSTTSProvider(BaseProvider, TTSProvider):
    def generate_speech(self, text, voice, language_code=None, **kwargs):
        return b"audio", 16000


if __name__ == "__main__":
    unittest.main()
