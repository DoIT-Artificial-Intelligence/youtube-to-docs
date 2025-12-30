import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import polars as pl

from youtube_to_docs.tts import generate_speech, parse_tts_arg, process_tts


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

    @patch("youtube_to_docs.tts.genai")
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
        audio_data = generate_speech("Hello world", "model", "voice")

        # Verify
        self.assertEqual(audio_data, b"fake_audio_data")
        mock_client.models.generate_content.assert_called_once()

    @patch("youtube_to_docs.tts.genai")
    @patch.dict(os.environ, {}, clear=True)
    def test_generate_speech_no_api_key(self, mock_genai):
        # Execute
        audio_data = generate_speech("Hello world", "model", "voice")

        # Verify
        self.assertEqual(audio_data, b"")
        mock_genai.Client.assert_not_called()

    @patch("youtube_to_docs.tts.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"})
    def test_generate_speech_api_error(self, mock_genai):
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("API Error")

        # Execute
        audio_data = generate_speech("Hello world", "model", "voice")

        # Verify
        self.assertEqual(audio_data, b"")

    @patch("youtube_to_docs.tts.wave_file")
    @patch("youtube_to_docs.tts.generate_speech")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open, read_data="Summary text")
    def test_process_tts(
        self,
        mock_open,
        mock_makedirs,
        mock_getsize,
        mock_exists,
        mock_generate_speech,
        mock_wave_file,
    ):
        # Setup DataFrame
        df = pl.DataFrame(
            {
                "Summary File 1": ["/path/to/summary1.md", "/path/to/summary2.md"],
                "Other Col": [1, 2],
            }
        )

        # Setup Mocks
        # Default behavior for os.path.exists:
        # 1. Summary file check (True)
        # 2. Audio file check (False - needs generation)
        # 3. Summary file check (True)
        # 4. Audio file check (True - already exists)

        # However, it's called multiple times.
        # Let's verify by logic:
        # Row 1: summary exists? -> True. audio exists? -> False. -> generate.
        # Row 2: summary exists? -> True. audio exists? -> True. -> skip.

        def exists_side_effect(path):
            if path.endswith(".md"):
                return True
            if path.endswith("summary1 - tts-arg.wav"):
                return False
            if path.endswith("summary2 - tts-arg.wav"):
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_getsize.return_value = 100  # For the existing file check

        mock_generate_speech.return_value = b"audio_bytes"

        # Execute
        updated_df = process_tts(df, "tts-arg", "/tmp")

        # Verify
        self.assertIn("Summary Audio File 1 tts-arg File", updated_df.columns)

        # Check generate_speech called once (for the first row only)
        mock_generate_speech.assert_called_once_with(
            "Summary text", "tts", "arg", "en-US"
        )

        # Check wave_file called once
        mock_wave_file.assert_called_once()
        args, _ = mock_wave_file.call_args
        self.assertTrue(args[0].endswith("summary1 - tts-arg.wav"))
        self.assertEqual(args[1], b"audio_bytes")

        # Check DataFrame content
        # Row 1 should have new audio path
        # Row 2 should have existing audio path (from logic)
        new_col = updated_df["Summary Audio File 1 tts-arg File"]
        self.assertTrue(new_col[0].endswith("summary1 - tts-arg.wav"))
        self.assertTrue(new_col[1].endswith("summary2 - tts-arg.wav"))

    @patch("youtube_to_docs.tts.wave_file")
    @patch("youtube_to_docs.tts.generate_speech")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open, read_data="Texto resumen")
    def test_process_tts_with_language(
        self,
        mock_open,
        mock_makedirs,
        mock_getsize,
        mock_exists,
        mock_generate_speech,
        mock_wave_file,
    ):
        # Setup DataFrame with language-suffixed column
        df = pl.DataFrame(
            {
                "Summary File (es)": ["/path/to/summary_es.md"],
            }
        )

        # Mock that audio file doesn't exist
        def exists_side_effect(path):
            if path.endswith(".md"):
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_generate_speech.return_value = b"audio_bytes"

        # Execute
        process_tts(df, "tts-arg", "/tmp")

        # Verify generate_speech called with 'es-US'
        mock_generate_speech.assert_called_once_with(
            "Texto resumen", "tts", "arg", "es-US"
        )


if __name__ == "__main__":
    unittest.main()
