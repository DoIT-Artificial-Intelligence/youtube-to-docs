import os
import unittest
from unittest.mock import MagicMock, patch

from youtube_to_docs import llms


class TestLLMs(unittest.TestCase):
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict(
            os.environ,
            {
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

    def test_build_summary_prompt(self):
        prompt = llms.build_summary_prompt(
            "transcript text",
            "Video Title",
            "https://youtu.be/abc",
            language="es",
        )
        self.assertIn("https://youtu.be/abc (Video Title)", prompt)
        self.assertIn("summarize this in es?", prompt)
        self.assertIn("transcript text", prompt)

    def test_build_summary_prompt_default_language(self):
        prompt = llms.build_summary_prompt("transcript", "Title", "url")
        self.assertIn("summarize this in en?", prompt)

    @patch("youtube_to_docs.llms._query_llm")
    def test_generate_summary_uses_build_prompt(self, mock_query):
        # The prompt sent to the model must match build_summary_prompt
        # so the persisted prompt file is faithful.
        mock_query.return_value = ("Summary", 1, 2)

        llms.generate_summary("gemini-pro", "transcript", "Title", "url", language="fr")

        sent_prompt = mock_query.call_args[0][1]
        expected = llms.build_summary_prompt(
            "transcript", "Title", "url", language="fr"
        )
        self.assertEqual(sent_prompt, expected)

    @patch("google.genai.Client")
    def test_generate_summary_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()
        mock_resp.text = "Gemini Summary"
        mock_resp.usage_metadata.prompt_token_count = 100
        mock_resp.usage_metadata.candidates_token_count = 50
        mock_client.models.generate_content.return_value = mock_resp

        summary, in_tokens, out_tokens = llms.generate_summary(
            "gemini-pro", "transcript", "Title", "url"
        )
        self.assertEqual(summary, "Gemini Summary")
        self.assertEqual(in_tokens, 100)
        self.assertEqual(out_tokens, 50)

        # Thinking tokens are billed as output, so they count towards it.
        mock_resp.usage_metadata.thoughts_token_count = 30
        _, _, out_tokens = llms.generate_summary("gemini-pro", "t", "Title", "url")
        self.assertEqual(out_tokens, 80)

    @patch("anthropic.AnthropicVertex")
    @patch("google.auth.default")
    def test_generate_summary_vertex(self, mock_auth, mock_anthropic_vertex):
        mock_creds = MagicMock()
        mock_auth.return_value = (mock_creds, "proj")

        mock_client = MagicMock()
        mock_anthropic_vertex.return_value = mock_client

        mock_message = MagicMock()
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Vertex Summary"
        mock_message.content = [mock_block]
        mock_message.usage.input_tokens = 100
        mock_message.usage.output_tokens = 50

        mock_client.messages.create.return_value = mock_message

        summary, in_tokens, out_tokens = llms.generate_summary(
            "vertex-claude-3-5", "transcript", "Title", "url"
        )
        self.assertEqual(summary, "Vertex Summary")
        self.assertEqual(in_tokens, 100)
        self.assertEqual(out_tokens, 50)

    @patch("youtube_to_docs.llms.requests.post")
    def test_generate_summary_bedrock(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "output": {"message": {"content": [{"text": "Bedrock Summary"}]}},
            "usage": {"inputTokens": 100, "outputTokens": 50},
        }
        mock_post.return_value = mock_resp

        summary, in_tokens, out_tokens = llms.generate_summary(
            "bedrock-claude-3-5", "transcript", "Title", "url"
        )
        self.assertEqual(summary, "Bedrock Summary")
        self.assertEqual(in_tokens, 100)
        self.assertEqual(out_tokens, 50)

    @patch("youtube_to_docs.llms.requests.post")
    def test_bedrock_claude_model_ids(self, mock_post):
        """Dated Claude IDs get a ':0' suffix; newer undated IDs don't."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        mock_post.return_value = mock_resp

        cases = {
            "bedrock-claude-opus-5-5": "us.anthropic.claude-opus-5-5",
            "bedrock-claude-haiku-4-5-20251001-v1": (
                "us.anthropic.claude-haiku-4-5-20251001-v1:0"
            ),
        }
        for model, expected_id in cases.items():
            with self.subTest(model=model):
                llms.generate_summary(model, "transcript", "Title", "url")
                url = mock_post.call_args.args[0]
                self.assertIn(f"/model/{expected_id}/", url)

    @patch("youtube_to_docs.llms.requests.post")
    def test_bedrock_openai_uses_mantle(self, mock_post):
        """OpenAI GPT models on Bedrock go to the Mantle chat completions API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Mantle Summary"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock_post.return_value = mock_resp

        for model in ["bedrock-gpt-6-luna", "bedrock-openai.gpt-6-luna"]:
            with self.subTest(model=model):
                summary, in_tokens, out_tokens = llms.generate_summary(
                    model, "transcript", "Title", "url"
                )
                self.assertEqual(summary, "Mantle Summary")
                self.assertEqual((in_tokens, out_tokens), (100, 50))
                self.assertEqual(
                    mock_post.call_args.args[0],
                    "https://bedrock-mantle.us-east-1.api.aws/openai/v1/chat/completions",
                )
                self.assertEqual(
                    mock_post.call_args.kwargs["json"]["model"], "openai.gpt-6-luna"
                )

    @patch("youtube_to_docs.llms.requests.post")
    def test_bedrock_timeout(self, mock_post):
        """Bedrock calls set a timeout and report a stalled request as an error."""
        import requests

        mock_post.side_effect = requests.exceptions.ReadTimeout("read timed out")
        for model in ["bedrock-claude-haiku-5-5", "bedrock-gpt-6-luna"]:
            with self.subTest(model=model):
                summary, in_tokens, out_tokens = llms.generate_summary(
                    model, "transcript", "Title", "url"
                )
                self.assertIn("read timed out", summary)
                self.assertEqual((in_tokens, out_tokens), (0, 0))
                self.assertEqual(
                    mock_post.call_args.kwargs["timeout"],
                    llms.BEDROCK_TIMEOUT_SECONDS,
                )

    @patch("youtube_to_docs.llms.requests.post")
    def test_bedrock_http_error_is_error_response(self, mock_post):
        """A retired model's 404 is reported in the shared "Error:" format, so
        main() doesn't save it as a summary or reuse it on the next run."""
        from youtube_to_docs.utils import is_error_response

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = (
            '{"message":"This model version has reached the end of its life."}'
        )
        mock_post.return_value = mock_resp
        for model in ["bedrock-nova-premier-v1", "bedrock-gpt-6-luna"]:
            with self.subTest(model=model):
                summary, in_tokens, out_tokens = llms.generate_summary(
                    model, "transcript", "Title", "url"
                )
                self.assertTrue(is_error_response(summary), summary)
                self.assertIn("404", summary)
                self.assertEqual((in_tokens, out_tokens), (0, 0))

    def test_is_bedrock_openai_model(self):
        self.assertTrue(llms._is_bedrock_openai_model("gpt-6-luna"))
        self.assertTrue(llms._is_bedrock_openai_model("openai.gpt-6.1-sol"))
        self.assertFalse(llms._is_bedrock_openai_model("openai.gpt-oss-120b-1:0"))
        self.assertFalse(llms._is_bedrock_openai_model("claude-opus-5-5"))

    @patch("anthropic.AnthropicVertex")
    @patch("google.auth.default")
    def test_vertex_claude_default_region(self, mock_auth, mock_anthropic_vertex):
        """Undated Claude models default to the global endpoint on Vertex."""
        mock_auth.return_value = (MagicMock(), "proj")
        mock_message = MagicMock()
        mock_message.content = []
        mock_anthropic_vertex.return_value.messages.create.return_value = mock_message

        cases = {
            "vertex-claude-opus-5-5": "global",
            "vertex-claude-haiku-4-5@20251001": "us-east5",
        }
        env = {k: v for k, v in os.environ.items() if k != "VERTEX_LOCATION"}
        for model, expected_region in cases.items():
            with self.subTest(model=model), patch.dict(os.environ, env, clear=True):
                llms.generate_summary(model, "transcript", "Title", "url")
                self.assertEqual(
                    mock_anthropic_vertex.call_args.kwargs["region"], expected_region
                )

    def test_get_model_pricing_claude_5_5(self):
        """Claude 5.5 models resolve to prices via Bedrock and Vertex names."""
        cases = {
            "bedrock-claude-opus-5-5": (4, 20),
            "vertex-claude-opus-5-5": (4, 20),
            "bedrock-claude-sonnet-5-5": (2, 10),
            "vertex-claude-sonnet-5-5": (2, 10),
            "bedrock-claude-haiku-5-5": (0.1, 0.5),
            "vertex-claude-haiku-5-5": (0.1, 0.5),
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(llms.get_model_pricing(model), expected)

    @patch("openai.OpenAI")
    def test_generate_summary_foundry(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Foundry Summary"
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_completion

        summary, in_tokens, out_tokens = llms.generate_summary(
            "foundry-gpt-4", "transcript", "Title", "url"
        )
        self.assertEqual(summary, "Foundry Summary")
        self.assertEqual(in_tokens, 100)
        self.assertEqual(out_tokens, 50)

    @patch("google.genai.Client")
    def test_extract_speakers_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()
        mock_resp.text = "Speaker 1 (Expert)\nSpeaker 2 (UNKNOWN)"
        mock_resp.usage_metadata.prompt_token_count = 120
        mock_resp.usage_metadata.candidates_token_count = 30
        mock_client.models.generate_content.return_value = mock_resp

        speakers, in_tokens, out_tokens = llms.extract_speakers(
            "gemini-pro", "transcript content"
        )
        self.assertEqual(speakers, "Speaker 1 (Expert)\nSpeaker 2 (UNKNOWN)")
        self.assertEqual(in_tokens, 120)
        self.assertEqual(out_tokens, 30)

    @patch("google.genai.Client")
    def test_generate_qa_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()
        mock_resp.text = "| Q | A |\n|---|---|\n| Q1 | A1 |"
        mock_resp.usage_metadata.prompt_token_count = 150
        mock_resp.usage_metadata.candidates_token_count = 60
        mock_client.models.generate_content.return_value = mock_resp

        qa, in_tokens, out_tokens = llms.generate_qa(
            "gemini-pro",
            "transcript content",
            "Speaker 1, Speaker 2",
            "https://youtu.be/vid1",
        )
        # Expecting the added column
        expected_qa = "| question number | Q | A |\n|---|---|---|\n| 1 | Q1 | A1 |"
        self.assertEqual(qa, expected_qa)
        self.assertEqual(in_tokens, 150)
        self.assertEqual(out_tokens, 60)

    @patch("google.genai.Client")
    def test_generate_tags_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()
        mock_resp.text = "Tag1, Tag2, Tag3, Tag4, Tag5"
        mock_resp.usage_metadata.prompt_token_count = 80
        mock_resp.usage_metadata.candidates_token_count = 20
        mock_client.models.generate_content.return_value = mock_resp

        tags, in_tokens, out_tokens = llms.generate_tags(
            "gemini-pro", "summary content", "en"
        )
        self.assertEqual(tags, "Tag1, Tag2, Tag3, Tag4, Tag5")
        self.assertEqual(in_tokens, 80)
        self.assertEqual(out_tokens, 20)

    @patch("google.genai.Client")
    def test_generate_summary_gemma(self, mock_client_cls):
        """Gemma models should route to the Google GenAI client (same as Gemini)."""
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()
        mock_resp.text = "Gemma Summary"
        mock_resp.usage_metadata.prompt_token_count = 100
        mock_resp.usage_metadata.candidates_token_count = 50
        mock_client.models.generate_content.return_value = mock_resp

        summary, in_tokens, out_tokens = llms.generate_summary(
            "gemma-4-27b-it", "transcript", "Title", "url"
        )
        self.assertEqual(summary, "Gemma Summary")
        self.assertEqual(in_tokens, 100)
        self.assertEqual(out_tokens, 50)

    def test_vertex_gemma_raises_not_implemented(self):
        """vertex-gemma-* should raise NotImplementedError (requires deployment)."""
        with self.assertRaises(NotImplementedError):
            llms._query_llm("vertex-gemma-4-27b-it", "test prompt")

    @patch("google.genai.Client")
    def test_generate_alt_text_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()
        mock_resp.text = "Alt text: A beautiful infographic description."
        mock_resp.usage_metadata.prompt_token_count = 110
        mock_resp.usage_metadata.candidates_token_count = 40
        mock_client.models.generate_content.return_value = mock_resp

        alt_text, in_tokens, out_tokens = llms.generate_alt_text(
            "gemini-pro", b"fake_image_bytes", "en"
        )
        self.assertEqual(alt_text, "A beautiful infographic description.")
        self.assertEqual(in_tokens, 110)
        self.assertEqual(out_tokens, 40)


class TestPricing(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "fake_gemini_key",
                "AWS_BEARER_TOKEN_BEDROCK": "fake_bedrock_token",
            },
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch(
        "youtube_to_docs.llms.PRICES",
        {"prices": [{"id": "gpt-4", "input": 30.0, "output": 60.0}]},
    )
    def test_get_model_pricing_found(self):
        inp, outp = llms.get_model_pricing("gpt-4")
        self.assertEqual(inp, 30.0)
        self.assertEqual(outp, 60.0)

    @patch(
        "youtube_to_docs.llms.PRICES",
        {"prices": [{"id": "gpt-4", "input": 30.0, "output": 60.0}]},
    )
    def test_get_model_pricing_normalized(self):
        inp, outp = llms.get_model_pricing("vertex-gpt-4")
        self.assertEqual(inp, 30.0)
        self.assertEqual(outp, 60.0)

    @patch(
        "youtube_to_docs.llms.PRICES",
        {
            "prices": [{"id": "claude-4.5-haiku", "input": 1.0, "output": 5.0}],
            "aliases": {"claude-haiku-4-5": "claude-4.5-haiku"},
        },
    )
    def test_get_model_pricing_aliased(self):
        """Test that aliases (like claude-haiku-4-5 -> claude-4.5-haiku) work."""
        # This model name normalizes to 'claude-haiku-4-5'
        # which should alias to 'claude-4.5-haiku'
        inp, outp = llms.get_model_pricing("bedrock-claude-haiku-4-5-20251001-v1")
        self.assertEqual(inp, 1.0)
        self.assertEqual(outp, 5.0)

    @patch(
        "youtube_to_docs.llms.PRICES",
        {
            "prices": [
                {"id": "claude-4.5-haiku", "input": 1.0, "output": 5.0},
                {"id": "gemini-3.1-flash-lite", "input": 0.5, "output": 3.0},
                {"id": "gpt-5-mini", "input": 0.25, "output": 2.0},
                {
                    "id": "amazon-nova-2-lite",
                    "input": 0.3,
                    "output": 2.5,
                },
                {
                    "id": "gemini-nano-banana-2.1",
                    "input": 1.5,
                    "output": 30.0,
                },
            ],
            "aliases": {
                "claude-haiku-4-5": "claude-4.5-haiku",
                "nova-2-lite": "amazon-nova-2-lite",
            },
        },
    )
    def test_get_model_pricing_specific_models(self):
        """Test getting pricing for specific models requested by user."""
        models = [
            "gemini-3.1-flash-lite",
            "vertex-claude-haiku-4-5@20251001",
            "bedrock-claude-haiku-4-5-20251001-v1",
            "bedrock-nova-2-lite-v1",
            "foundry-gpt-5-mini",
            "gemini-nano-banana-2.1",
        ]

        for model in models:
            with self.subTest(model=model):
                inp, outp = llms.get_model_pricing(model)
                self.assertIsNotNone(inp, f"Input price for {model} should not be None")
                self.assertIsNotNone(
                    outp, f"Output price for {model} should not be None"
                )

    @patch("youtube_to_docs.llms.PRICES", {"prices": [{"id": "gpt-4"}]})
    def test_get_model_pricing_not_found(self):
        inp, outp = llms.get_model_pricing("non-existent-model")
        self.assertIsNone(inp)
        self.assertIsNone(outp)

    def test_get_model_pricing_unlisted_model_returns_none(self):
        """Models absent from prices.py (e.g. gemini-2.0-flash-lite) must
        return (None, None) so downstream cost calculations skip cleanly."""
        for model in [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "claude-3-opus",
            "gpt-4o",
            "totally-fake-model-xyz",
        ]:
            with self.subTest(model=model):
                inp, outp = llms.get_model_pricing(model)
                self.assertIsNone(inp)
                self.assertIsNone(outp)

    def test_cost_calc_skipped_when_pricing_missing(self):
        """Mirrors the guard pattern used throughout main.py — no crash, no cost."""
        input_price, output_price = llms.get_model_pricing("gemini-2.0-flash-lite")
        cost = None
        if input_price is not None and output_price is not None:
            cost = (1000 / 1_000_000) * input_price + (500 / 1_000_000) * output_price
        self.assertIsNone(cost)

    def test_get_model_pricing_openai_gpt_6(self):
        """OpenAI models resolve to prices via Foundry names and old aliases."""
        cases = {
            "foundry-gpt-6-astra": (10.0, 50.0),
            "foundry-gpt-6.1-sol": (2.0, 10.0),
            "foundry-gpt-6-luna": (0.1, 0.5),
            "gpt-5.5": (10.0, 50.0),
            "gpt-5.4": (2.0, 10.0),
            "gpt-5.4-mini": (0.1, 0.5),
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(llms.get_model_pricing(model), expected)

    def test_get_model_pricing_claude_fable_5_1(self):
        """Test that get_model_pricing retrieves the correct prices for
        claude-fable-5.1 and its aliases.
        """
        # Test exact match
        inp, outp = llms.get_model_pricing("claude-fable-5.1")
        self.assertEqual(inp, 10.0)
        self.assertEqual(outp, 50.0)

        # Test alias matches
        for alias in [
            "bedrock-claude-fable-5-1",
            "vertex-claude-fable-5-1",
            "claude-fable-5",
            "anthropic-5-fable",
            "anthrophic-5-fable",
        ]:
            with self.subTest(alias=alias):
                inp_alias, outp_alias = llms.get_model_pricing(alias)
                self.assertEqual(inp_alias, 10.0)
                self.assertEqual(outp_alias, 50.0)


if __name__ == "__main__":
    unittest.main()
