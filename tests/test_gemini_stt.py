import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google.genai import types

from youtube_to_docs import llms
from youtube_to_docs.llms import (
    GEMINI_TRANSCRIBE_API_VERSION,
    generate_transcript_with_srt,
    is_gemini_transcribe_model,
)


def _transcription_part(text, speaker_label, words):
    """Builds a response part shaped like a dedicated Gemini STT segment."""
    return types.Part(
        text=text,
        audio_transcription=types.Transcription(
            text=text,
            speaker_label=speaker_label,
            words=[
                types.WordInfo(word=word, start_offset=start, end_offset=end)
                for word, start, end in words
            ],
        ),
    )


def _response(parts, prompt_tokens=376, candidate_tokens=None):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidate_tokens,
        ),
    )


TWO_SPEAKER_PARTS = [
    _transcription_part(
        "Good morning. This is the review.",
        "spk:0",
        [
            ("Good", "0.300s", "0.400s"),
            ("morning.", "0.400s", "1.100s"),
            ("This", "1.200s", "1.500s"),
            ("is", "1.500s", "1.600s"),
            ("the", "1.600s", "1.700s"),
            ("review.", "1.700s", "3.600s"),
        ],
    ),
    _transcription_part(
        "Thanks.",
        "spk:1",
        [("Thanks.", "4.200s", "4.800s")],
    ),
    _transcription_part(
        "Excellent.",
        "spk:0",
        [("Excellent.", "8.600s", "9.400s")],
    ),
]


class TestGeminiTranscribeDetection(unittest.TestCase):
    def test_dedicated_stt_models_detected(self):
        self.assertTrue(is_gemini_transcribe_model("gemini-3.5-transcribe"))
        self.assertTrue(is_gemini_transcribe_model("gemini-3.5-transcribe-live"))

    def test_other_models_not_detected(self):
        self.assertFalse(is_gemini_transcribe_model("gemini-3.5-flash-lite"))
        self.assertFalse(is_gemini_transcribe_model("gcp-chirp3"))
        self.assertFalse(is_gemini_transcribe_model("aws-transcribe"))

    @patch("youtube_to_docs.llms._transcribe_gemini")
    def test_gemini_transcribe_dispatch(self, mock_transcribe):
        """gemini-*-transcribe is dispatched to _transcribe_gemini, not the prompt."""
        mock_transcribe.return_value = ("transcript", "srt_content", 0, 0)

        generate_transcript_with_srt("gemini-3.5-transcribe", "audio.m4a", "http://url")

        mock_transcribe.assert_called_once()
        args, _ = mock_transcribe.call_args
        self.assertEqual(args[0], "gemini-3.5-transcribe")

    @patch("youtube_to_docs.llms._transcribe_gemini")
    def test_prompt_based_gemini_models_not_dispatched(self, mock_transcribe):
        """Regular Gemini models keep using the prompt-based transcribe path."""
        with patch("google.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
                generate_transcript_with_srt(
                    "gemini-3.5-flash-lite", "audio.m4a", "http://url"
                )

        mock_transcribe.assert_not_called()


class TestProcessGeminiTranscription(unittest.TestCase):
    def test_text_joins_segments_without_labels(self):
        text, _ = llms._process_gemini_transcription(_response(TWO_SPEAKER_PARTS))

        self.assertEqual(text, "Good morning. This is the review. Thanks. Excellent.")

    def test_srt_uses_word_offsets(self):
        _, srt = llms._process_gemini_transcription(_response(TWO_SPEAKER_PARTS))

        self.assertIn("00:00:00,300 --> 00:00:01,100", srt)
        self.assertIn("00:00:04,200 --> 00:00:04,800", srt)

    def test_srt_labels_speaker_changes(self):
        _, srt = llms._process_gemini_transcription(_response(TWO_SPEAKER_PARTS))

        self.assertIn("[Speaker 1] Good morning.", srt)
        self.assertIn("[Speaker 2] Thanks.", srt)
        # The first speaker keeps their number when they return.
        self.assertIn("[Speaker 1] Excellent.", srt)
        # Continuation entries within a turn are not re-labelled.
        self.assertIn("This is the review.", srt)
        self.assertNotIn("[Speaker 1] This is the review.", srt)

    def test_srt_entries_are_numbered_sequentially(self):
        _, srt = llms._process_gemini_transcription(_response(TWO_SPEAKER_PARTS))

        numbers = [line for line in srt.splitlines() if line.strip().isdigit()]
        self.assertEqual(numbers, ["1", "2", "3", "4"])

    def test_long_turn_is_split_across_entries(self):
        words = [(f"word{i}", f"{i}s", f"{i + 1}s") for i in range(30)]
        part = _transcription_part("long turn", "spk:0", words)

        _, srt = llms._process_gemini_transcription(_response([part]))

        self.assertGreater(srt.count(" --> "), 1)

    def test_empty_response_returns_empty_strings(self):
        self.assertEqual(
            llms._process_gemini_transcription(types.GenerateContentResponse()),
            ("", ""),
        )

    def test_parts_without_transcription_still_contribute_text(self):
        response = _response([types.Part(text="Plain text only.")])

        text, srt = llms._process_gemini_transcription(response)

        self.assertEqual(text, "Plain text only.")
        self.assertEqual(srt, "")


class TestTranscribeGemini(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"})
        self.env_patcher.start()

        tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
        tmp.write(b"fake audio bytes")
        tmp.close()
        self.audio_path = tmp.name

    def tearDown(self):
        self.env_patcher.stop()
        os.unlink(self.audio_path)

    def _run(self, mock_client_cls, **kwargs):
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.return_value = _response(TWO_SPEAKER_PARTS)
        return (
            llms._transcribe_gemini(
                "gemini-3.5-transcribe", self.audio_path, "http://url", **kwargs
            ),
            mock_client,
        )

    @patch("google.genai.Client")
    def test_returns_transcript_srt_and_tokens(self, mock_client_cls):
        (transcript, srt, in_tok, out_tok), _ = self._run(mock_client_cls)

        self.assertEqual(
            transcript, "Good morning. This is the review. Thanks. Excellent."
        )
        self.assertIn("[Speaker 2] Thanks.", srt)
        self.assertEqual(in_tok, 376)
        # The API does not report transcript output tokens for this model.
        self.assertEqual(out_tok, 0)

    @patch("google.genai.Client")
    def test_uses_v1alpha_api_version(self, mock_client_cls):
        """Word timestamps and speaker labels are dropped on v1beta."""
        self._run(mock_client_cls)

        _, kwargs = mock_client_cls.call_args
        self.assertEqual(
            kwargs["http_options"].api_version, GEMINI_TRANSCRIBE_API_VERSION
        )

    @patch("google.genai.Client")
    def test_requests_word_timestamps_and_diarization(self, mock_client_cls):
        _, mock_client = self._run(mock_client_cls)

        _, kwargs = mock_client.models.generate_content.call_args
        transcription_config = kwargs["config"].audio_transcription_config
        self.assertTrue(transcription_config.word_timestamp)
        self.assertTrue(transcription_config.diarization)
        self.assertEqual(transcription_config.language_codes, ["en-US"])

    @patch("google.genai.Client")
    def test_language_code_passed_through(self, mock_client_cls):
        _, mock_client = self._run(mock_client_cls, language="es")

        _, kwargs = mock_client.models.generate_content.call_args
        self.assertEqual(
            kwargs["config"].audio_transcription_config.language_codes, ["es"]
        )

    @patch("google.genai.Client")
    def test_small_audio_is_sent_inline(self, mock_client_cls):
        _, mock_client = self._run(mock_client_cls)

        mock_client.files.upload.assert_not_called()
        _, kwargs = mock_client.models.generate_content.call_args
        part = kwargs["contents"][0].parts[0]
        self.assertIsNotNone(part.inline_data)

    @patch("os.path.getsize", return_value=llms.GEMINI_INLINE_AUDIO_LIMIT_BYTES + 1)
    @patch("google.genai.Client")
    def test_large_audio_uses_files_api_and_cleans_up(
        self, mock_client_cls, mock_getsize
    ):
        mock_client = mock_client_cls.return_value
        uploaded = MagicMock()
        uploaded.uri = "https://generativelanguage.googleapis.com/v1beta/files/abc"
        uploaded.name = "files/abc"
        mock_client.files.upload.return_value = uploaded
        mock_client.models.generate_content.return_value = _response(TWO_SPEAKER_PARTS)

        llms._transcribe_gemini("gemini-3.5-transcribe", self.audio_path, "http://url")

        mock_client.files.upload.assert_called_once()
        _, kwargs = mock_client.models.generate_content.call_args
        part = kwargs["contents"][0].parts[0]
        self.assertEqual(part.file_data.file_uri, uploaded.uri)
        mock_client.files.delete.assert_called_once_with(name="files/abc")

    @patch("google.genai.Client")
    def test_api_error_is_reported(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.side_effect = Exception("boom")

        transcript, srt, in_tok, out_tok = llms._transcribe_gemini(
            "gemini-3.5-transcribe", self.audio_path, "http://url"
        )

        self.assertIn("boom", transcript)
        self.assertEqual((srt, in_tok, out_tok), ("", 0, 0))

    @patch("google.genai.Client")
    def test_empty_transcript_is_reported(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.return_value = _response([])

        transcript, srt, _, _ = llms._transcribe_gemini(
            "gemini-3.5-transcribe", self.audio_path, "http://url"
        )

        self.assertIn("no transcript", transcript)
        self.assertEqual(srt, "")

    def test_live_model_is_rejected(self):
        transcript, srt, in_tok, out_tok = llms._transcribe_gemini(
            "gemini-3.5-transcribe-live", self.audio_path, "http://url"
        )

        self.assertIn("Live API", transcript)
        self.assertEqual((srt, in_tok, out_tok), ("", 0, 0))

    def test_missing_api_key_is_reported(self):
        with patch.dict(os.environ, {}, clear=True):
            transcript, _, _, _ = llms._transcribe_gemini(
                "gemini-3.5-transcribe", self.audio_path, "http://url"
            )

        self.assertEqual(transcript, "Error: GEMINI_API_KEY not found")


class TestGeminiTranscribePricing(unittest.TestCase):
    def test_pricing_is_listed(self):
        input_price, output_price = llms.get_model_pricing("gemini-3.5-transcribe")

        self.assertEqual(input_price, 2.0)
        self.assertEqual(output_price, 12.0)


if __name__ == "__main__":
    unittest.main()
