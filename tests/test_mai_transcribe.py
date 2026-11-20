import json
import os
import unittest
from unittest.mock import patch

import requests

from youtube_to_docs.llms import get_model_pricing
from youtube_to_docs.mai_transcribe import (
    MAI_TRANSCRIBE_CHUNK_SECONDS,
    MaiTranscribeProvider,
    transcribe_mai,
)
from youtube_to_docs.providers import get_provider
from youtube_to_docs.translate import parse_corrected_captions_arg

ENV = {
    "AZURE_TRANSCRIPTION_ENDPOINT": "https://example.cognitiveservices.azure.com/",
    "AZURE_TRANSCRIPTION_API_KEY": "key",
}


def _word(text, start_ms, duration_ms):
    return {
        "text": text,
        "offsetMilliseconds": start_ms,
        "durationMilliseconds": duration_ms,
    }


def _response(phrases, duration_ms=60_000):
    return {
        "durationMilliseconds": duration_ms,
        "combinedPhrases": [{"text": " ".join(p["text"] for p in phrases)}],
        "phrases": phrases,
    }


GOOD_MORNING = _response(
    [
        {
            "speaker": 0,
            "offsetMilliseconds": 1000,
            "durationMilliseconds": 1000,
            "text": "Good morning.",
            "words": [_word("Good", 1000, 400), _word("morning.", 1500, 500)],
        },
        {
            "speaker": 1,
            "offsetMilliseconds": 2500,
            "durationMilliseconds": 700,
            "text": "Thank you.",
            "words": [_word("Thank", 2500, 300), _word("you.", 2900, 300)],
        },
    ]
)


@patch.dict(os.environ, ENV)
class TestTranscribe(unittest.TestCase):
    def setUp(self):
        def fake_ffmpeg(cmd, check):
            with open(cmd[-1], "wb") as f:
                f.write(b"flac-bytes")

        patcher = patch(
            "youtube_to_docs.mai_transcribe.subprocess.run", side_effect=fake_ffmpeg
        )
        self.mock_run = patcher.start()
        self.addCleanup(patcher.stop)

    @patch("youtube_to_docs.mai_transcribe._transcribe_chunk")
    def test_speakers_become_labels(self, mock_chunk):
        mock_chunk.return_value = GOOD_MORNING
        text, srt, in_tok, out_tok = transcribe_mai(
            "mai-transcribe-2", "a.m4a", "url", "en", 60
        )

        self.assertEqual(text, "Good morning. Thank you.")
        self.assertEqual(
            srt,
            "1\n00:00:01,000 --> 00:00:02,000\n[Speaker 1] Good morning.\n\n"
            "2\n00:00:02,500 --> 00:00:03,200\n[Speaker 2] Thank you.\n",
        )
        # Billed per minute: 200 pseudo-tokens a minute, each side.
        self.assertEqual((in_tok, out_tok), (200, 200))
        self.assertNotIn("-ss", self.mock_run.call_args.args[0])

    @patch("youtube_to_docs.mai_transcribe._transcribe_chunk")
    def test_same_speaker_keeps_one_label(self, mock_chunk):
        phrases = [
            {"speaker": 0, "text": "One.", "words": [_word("One.", 0, 500)]},
            {"speaker": 0, "text": "Two.", "words": [_word("Two.", 1000, 500)]},
        ]
        mock_chunk.return_value = _response(phrases)
        _, srt, _, _ = transcribe_mai("mai-transcribe-2", "a.m4a", "url", "en", 5)
        self.assertEqual(srt.count("[Speaker"), 1)

    @patch("youtube_to_docs.mai_transcribe._transcribe_chunk")
    def test_long_audio_is_chunked(self, mock_chunk):
        one = {"speaker": 0, "text": "One.", "words": [_word("One.", 1000, 500)]}
        two = {"speaker": 0, "text": "Two.", "words": [_word("Two.", 2000, 500)]}
        mock_chunk.side_effect = [
            _response([one], MAI_TRANSCRIBE_CHUNK_SECONDS * 1000),
            _response([two], 60_000),
        ]
        text, srt, in_tok, _ = transcribe_mai(
            "mai-transcribe-2", "a.m4a", "url", "en", MAI_TRANSCRIBE_CHUNK_SECONDS + 60
        )

        self.assertEqual(text, "One. Two.")
        # The second chunk's times are offset, and its speaker 0 is a new label.
        self.assertIn("00:30:02,000 --> 00:30:02,500\n[Speaker 2] Two.", srt)
        self.assertEqual(in_tok, 31 * 200)
        second_cmd = self.mock_run.call_args_list[1].args[0]
        self.assertEqual(second_cmd[second_cmd.index("-ss") + 1], "1800")

    @patch("youtube_to_docs.mai_transcribe.requests.post")
    def test_request(self, mock_post):
        mock_post.return_value.json.return_value = GOOD_MORNING
        transcribe_mai("mai-transcribe-2", "a.m4a", "url", "en", 60)

        args, kwargs = mock_post.call_args
        self.assertEqual(
            args[0],
            "https://example.cognitiveservices.azure.com"
            "/speechtotext/transcriptions:transcribe",
        )
        self.assertEqual(kwargs["params"], {"api-version": "2025-10-15"})
        self.assertEqual(kwargs["headers"], {"Ocp-Apim-Subscription-Key": "key"})
        definition = json.loads(kwargs["files"]["definition"][1])
        self.assertEqual(
            definition,
            {
                "diarization": {"enabled": True},
                "enhancedMode": {
                    "enabled": True,
                    "model": "mai-transcribe-2",
                    "modelOptions": {"timestamps": "word"},
                },
                "locales": ["en"],
            },
        )
        self.assertEqual(kwargs["files"]["audio"][2], "audio/flac")

    @patch(
        "youtube_to_docs.mai_transcribe._transcribe_chunk",
        side_effect=requests.ConnectionError("refused"),
    )
    def test_api_error(self, _):
        text, srt, _, _ = transcribe_mai("mai-transcribe-2", "a.m4a", "url", "en", 10)
        self.assertEqual(text, "Error: mai-transcribe-2 failed: refused")
        self.assertEqual(srt, "")

    @patch("youtube_to_docs.mai_transcribe._transcribe_chunk")
    def test_empty_transcript(self, mock_chunk):
        mock_chunk.return_value = _response([])
        text, _, _, _ = transcribe_mai("mai-transcribe-2", "a.m4a", "url", "en", 10)
        self.assertTrue(text.startswith("Error: mai-transcribe-2 returned no"))


class TestSetup(unittest.TestCase):
    def test_missing_env(self):
        with patch.dict(os.environ, {}, clear=True):
            text, srt, _, _ = transcribe_mai("mai-transcribe-2", "a.m4a", "url")
        self.assertEqual(
            text,
            "Error: AZURE_TRANSCRIPTION_ENDPOINT and AZURE_TRANSCRIPTION_API_KEY "
            "not set",
        )

    def test_provider_routing(self):
        self.assertIsInstance(get_provider("mai-transcribe-2"), MaiTranscribeProvider)

    def test_pricing(self):
        # $0.54 an hour.
        input_price, output_price = get_model_pricing("mai-transcribe-2")
        assert input_price is not None and output_price is not None
        cost = (200 * input_price + 200 * output_price) / 1_000_000 * 60
        self.assertAlmostEqual(cost, 0.54)

    def test_corrected_captions_source(self):
        self.assertEqual(
            parse_corrected_captions_arg("gemini-4-carbon-mai-transcribe-2"),
            ("gemini-4-carbon", "mai-transcribe-2"),
        )


if __name__ == "__main__":
    unittest.main()
