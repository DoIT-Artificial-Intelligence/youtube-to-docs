import os
import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs import video_understanding as vu


def _interaction(id_="int-1", status="completed", text="ok", usage=None):
    interaction = MagicMock()
    interaction.id = id_
    interaction.status = status
    interaction.output_text = text
    interaction.usage = usage or MagicMock(
        total_input_tokens=10,
        total_tool_use_tokens=5,
        total_output_tokens=20,
        total_thought_tokens=100,
    )
    return interaction


class TestParsing(unittest.TestCase):
    def test_parse_video_modes(self):
        self.assertEqual(vu.parse_video_modes(None), [])
        self.assertEqual(vu.parse_video_modes("agentic"), ["agentic"])
        self.assertEqual(vu.parse_video_modes("Agentic, static"), ["agentic", "static"])
        with self.assertRaises(ValueError):
            vu.parse_video_modes("agentic,dynamic")

    def test_video_mode_error(self):
        self.assertIsNone(vu.video_mode_error("gemini-3.5-flash-lite"))
        self.assertIsNone(vu.video_mode_error("gemini-4-argon"))
        error = vu.video_mode_error("vertex-claude-opus-5-5")
        assert error is not None
        self.assertIn("Gemini API model", error)

    def test_usage_tokens_counts_thought_and_tool_use(self):
        """Thought tokens bill as output and tool-use tokens as input."""
        self.assertEqual(vu._usage_tokens(_interaction()), (15, 120))
        self.assertEqual(vu._usage_tokens(MagicMock(usage=None)), (0, 0))


@patch.dict(os.environ, {"GEMINI_API_KEY": "fake"})
@patch("google.genai.Client")
class TestVideoSession(unittest.TestCase):
    def test_first_request_sends_video_then_reuses_session(self, mock_client_cls):
        client = mock_client_cls.return_value
        client.interactions.create.side_effect = [
            _interaction("int-1", text="summary"),
            _interaction("int-2", text="speakers"),
        ]
        session = vu.VideoSession("gemini-3.5-flash-lite", "https://yt/x", "agentic")

        self.assertEqual(session.ask("Summarize")[0], "summary")
        first = client.interactions.create.call_args_list[0].kwargs
        self.assertEqual(
            first["input"][0],
            {"type": "video", "uri": "https://yt/x", "processing": "agentic"},
        )
        self.assertEqual(first["input"][1]["text"], "Summarize")
        self.assertTrue(first["background"])
        self.assertNotIn("previous_interaction_id", first)

        self.assertEqual(session.ask("Speakers?")[0], "speakers")
        second = client.interactions.create.call_args_list[1].kwargs
        self.assertEqual(second["previous_interaction_id"], "int-1")
        self.assertEqual([p["type"] for p in second["input"]], ["text"])

    @patch("youtube_to_docs.video_understanding.time.sleep")
    def test_polls_until_complete(self, mock_sleep, mock_client_cls):
        client = mock_client_cls.return_value
        client.interactions.create.return_value = _interaction(status="in_progress")
        client.interactions.get.side_effect = [
            _interaction(status="in_progress"),
            _interaction(status="completed", text="done"),
        ]
        session = vu.VideoSession("gemini-3.5-flash-lite", "https://yt/x", "static")
        self.assertEqual(session.ask("Summarize")[0], "done")
        self.assertEqual(client.interactions.get.call_count, 2)

    def test_failed_status_returns_error(self, mock_client_cls):
        client = mock_client_cls.return_value
        client.interactions.create.return_value = _interaction(status="failed")
        session = vu.VideoSession("gemini-3.5-flash-lite", "https://yt/x", "static")
        text, in_tok, out_tok = session.ask("Summarize")
        self.assertTrue(text.startswith("Error"))
        self.assertEqual((in_tok, out_tok), (0, 0))

    @patch("youtube_to_docs.video_understanding.VIDEO_TIMEOUT_SECONDS", -1)
    def test_timeout_returns_error(self, mock_client_cls):
        client = mock_client_cls.return_value
        client.interactions.create.return_value = _interaction(status="in_progress")
        session = vu.VideoSession("gemini-3.5-flash-lite", "https://yt/x", "agentic")
        self.assertIn("timed out", session.ask("Summarize")[0])


class TestProcessFromVideo(unittest.TestCase):
    def _run(self, model_name="gemini-3.5-flash-lite", modes=("agentic",), row=None):
        storage = MagicMock()
        storage.write_text.side_effect = lambda path, text: path
        row = {} if row is None else row
        vu.process_from_video(
            row=row,
            storage=storage,
            model_name=model_name,
            modes=list(modes),
            video_id="vid1",
            video_title="Title",
            safe_title="Title",
            url="https://www.youtube.com/watch?v=vid1",
            language="en",
            col_suffix="",
            lang_str="",
            dirs={
                "summaries": "s",
                "summary_prompts": "p",
                "one_sentence": "o",
                "speakers": "spk",
                "qa": "qa",
                "video_context": "ctx",
            },
            verbose=False,
            vprint=lambda *a, **k: None,
        )
        return row, storage

    @patch("youtube_to_docs.video_understanding.generate_one_sentence_summary")
    @patch("youtube_to_docs.video_understanding.VideoSession")
    def test_fills_video_columns(self, mock_session_cls, mock_one_sentence):
        session = mock_session_cls.return_value
        session.ask.side_effect = [
            ("Video summary", 1, 1),
            ("Jane Doe (Chair)", 1, 1),
            ("## People\n- Jane Doe (Chair) — speaking at 00:10", 1, 1),
            ("| questioner(s) | question |\n|---|---|\n| A | Q? |", 1, 1),
        ]
        mock_one_sentence.return_value = ("One sentence.", 1, 1)

        row, _ = self._run()

        m = "gemini-3.5-flash-lite"
        self.assertEqual(row[f"Summary Text {m} from video-agentic"], "Video summary")
        self.assertIn(
            "summary (from video-agentic).md",
            row[f"Summary File {m} from video-agentic"],
        )
        self.assertEqual(
            row[f"One Sentence Summary {m} from video-agentic"], "One sentence."
        )
        self.assertEqual(row[f"Speakers {m} from video-agentic"], "Jane Doe (Chair)")
        self.assertIn("question number", row[f"QA Text {m} from video-agentic"])
        self.assertIn("speaking at 00:10", row[f"Video Context {m} from video-agentic"])
        self.assertIn(
            "video-context (from video-agentic).md",
            row[f"Video Context File {m} from video-agentic"],
        )
        mock_session_cls.assert_called_once_with(
            m, "https://www.youtube.com/watch?v=vid1", "agentic"
        )

    @patch("youtube_to_docs.video_understanding.VideoSession")
    def test_skips_non_gemini_model(self, mock_session_cls):
        row, _ = self._run(model_name="bedrock-claude-opus-5-5", modes=("agentic",))
        self.assertEqual(row, {})
        mock_session_cls.assert_not_called()

    @patch("youtube_to_docs.video_understanding.generate_one_sentence_summary")
    @patch("youtube_to_docs.video_understanding.VideoSession")
    def test_existing_outputs_are_not_regenerated(
        self, mock_session_cls, mock_one_sentence
    ):
        m = "gemini-3.5-flash-lite"
        row = {
            f"Summary Text {m} from video-static": "s",
            f"One Sentence Summary {m} from video-static": "o",
            f"Speakers {m} from video-static": "spk",
            f"Video Context {m} from video-static": "ctx",
            f"QA Text {m} from video-static": "qa",
        }
        self._run(modes=("static",), row=row)
        mock_session_cls.assert_not_called()
        mock_one_sentence.assert_not_called()

    @patch("youtube_to_docs.video_understanding.VideoSession")
    def test_summary_error_stops_mode(self, mock_session_cls):
        mock_session_cls.return_value.ask.return_value = ("Error: boom", 0, 0)
        row, _ = self._run()
        self.assertNotIn("Summary Text gemini-3.5-flash-lite from video-agentic", row)
        self.assertEqual(mock_session_cls.return_value.ask.call_count, 1)


if __name__ == "__main__":
    unittest.main()
