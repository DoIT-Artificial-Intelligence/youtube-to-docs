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
        # Converse only honours the output limit inside inferenceConfig.
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["inferenceConfig"], {"maxTokens": 64_000})
        self.assertNotIn("max_tokens", payload)

    @patch("youtube_to_docs.llms.requests.post")
    def test_generate_summary_bedrock_skips_reasoning_block(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "hmm"}}},
                        {"text": "Bedrock Summary"},
                    ]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 50},
        }
        mock_post.return_value = mock_resp

        summary, _, _ = llms.generate_summary(
            "bedrock-claude-opus-5-5", "transcript", "Title", "url"
        )
        self.assertEqual(summary, "Bedrock Summary")

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

    def _nvidia_response(self, status=200, body=None, text=""):
        resp = MagicMock()
        resp.status_code = status
        resp.text = text
        resp.json.return_value = body or {
            "choices": [
                {
                    "message": {
                        "content": "NVIDIA Summary",
                        "reasoning_content": "thinking",
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        return resp

    @patch("youtube_to_docs.llms._nvidia_model_ids", None)
    @patch("youtube_to_docs.llms.requests.get")
    @patch("youtube_to_docs.llms.requests.post")
    def test_generate_summary_nvidia(self, mock_post, mock_get):
        """nvidia-<model> resolves the publisher from NVIDIA's catalog."""
        mock_get.return_value.json.return_value = {
            "data": [
                {"id": "nvidia/nemotron-3-ultra-550b-a55b"},
                {"id": "moonshotai/kimi-k3.1-pro"},
            ]
        }
        mock_post.return_value = self._nvidia_response()

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "fake_nvidia_key"}):
            summary, in_tokens, out_tokens = llms.generate_summary(
                "nvidia-kimi-k3.1-pro", "transcript", "Title", "url"
            )

        self.assertEqual(summary, "NVIDIA Summary")
        self.assertEqual(in_tokens, 100)
        self.assertEqual(out_tokens, 50)
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "moonshotai/kimi-k3.1-pro")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer fake_nvidia_key")

    @patch("youtube_to_docs.llms._nvidia_model_ids", None)
    @patch("youtube_to_docs.llms.requests.get", side_effect=Exception("offline"))
    def test_resolve_nvidia_model_falls_back_to_nvidia_publisher(self, _mock_get):
        self.assertEqual(
            llms.resolve_nvidia_model("nvidia-nemotron-3-ultra-550b-a55b", "key"),
            "nvidia/nemotron-3-ultra-550b-a55b",
        )

    @patch(
        "youtube_to_docs.llms._nvidia_model_ids",
        ["nvidia/nemotron-3.5-lightning-30b-a3b"],
    )
    @patch("youtube_to_docs.llms.time.sleep")
    @patch("youtube_to_docs.llms.requests.post")
    def test_nvidia_retries_when_rate_limited(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            self._nvidia_response(status=429, text="Too Many Requests"),
            self._nvidia_response(),
        ]

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "fake_nvidia_key"}):
            text, _, _ = llms._query_llm(
                "nvidia-nemotron-3.5-lightning-30b-a3b", "prompt"
            )

        self.assertEqual(text, "NVIDIA Summary")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(llms.NVIDIA_RETRY_WAITS_SECONDS[0])

    @patch(
        "youtube_to_docs.llms._nvidia_model_ids",
        ["nvidia/nemotron-3.5-lightning-30b-a3b"],
    )
    @patch("youtube_to_docs.llms.requests.post")
    def test_nvidia_api_error_is_returned_as_error_text(self, mock_post):
        mock_post.return_value = self._nvidia_response(status=404, text="Not Found")

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "fake_nvidia_key"}):
            text, in_tokens, out_tokens = llms._query_llm(
                "nvidia-nemotron-3.5-lightning-30b-a3b", "prompt"
            )

        self.assertTrue(text.startswith("Error: NVIDIA API Error 404"))
        self.assertEqual((in_tokens, out_tokens), (0, 0))

    def test_nvidia_requires_api_key(self):
        env = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            text, _, _ = llms._query_llm(
                "nvidia-nemotron-3.5-lightning-30b-a3b", "prompt"
            )
        self.assertEqual(text, "Error: NVIDIA_API_KEY not found")

    def test_nvidia_models_are_free(self):
        self.assertEqual(llms.get_model_pricing("nvidia-any-catalog-model"), (0, 0))

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
            "foundry-gpt-6.2-astra": (8.0, 40.0),
            "foundry-gpt-6.2-sol": (2.0, 10.0),
            "foundry-gpt-6.1-terra": (1.0, 5.0),
            "foundry-gpt-6-luna": (0.1, 0.5),
            "foundry-gpt-6-astra": (8.0, 40.0),
            "foundry-gpt-6.1-sol": (2.0, 10.0),
            "gpt-5.5": (8.0, 40.0),
            "gpt-5.4": (2.0, 10.0),
            "gpt-5.4-mini": (0.1, 0.5),
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(llms.get_model_pricing(model), expected)

    def test_get_model_pricing_claude_fable_5_2(self):
        """Test that get_model_pricing retrieves the correct prices for
        claude-fable-5.2 and its aliases.
        """
        # Test exact match
        inp, outp = llms.get_model_pricing("claude-fable-5.2")
        self.assertEqual(inp, 8.0)
        self.assertEqual(outp, 40.0)

        # Test alias matches
        for alias in [
            "bedrock-claude-fable-5-2",
            "vertex-claude-fable-5-2",
            "bedrock-claude-fable-5-1",
            "claude-fable-5.1",
            "claude-fable-5",
            "anthropic-5-fable",
            "anthrophic-5-fable",
        ]:
            with self.subTest(alias=alias):
                inp_alias, outp_alias = llms.get_model_pricing(alias)
                self.assertEqual(inp_alias, 8.0)
                self.assertEqual(outp_alias, 40.0)


class TestDescription(unittest.TestCase):
    def test_parse_chapters_normalizes_youtube_format(self):
        text = (
            "Here are the chapters:\n"
            "- 00:05 Introduction and Webinar Overview\n"
            "**05:33** - Maryland State Perspective\n"
            "02:50 Learning Objectives\n"
            "02:55 Too close to the last one\n"
            "01:06:53 Panel Q&A And Closing Remarks\n"
        )
        self.assertEqual(
            llms.parse_chapters(text),
            "00:00 Introduction and Webinar Overview\n"
            "02:50 Learning Objectives\n"
            "05:33 Maryland State Perspective\n"
            "1:06:53 Panel Q&A And Closing Remarks",
        )

    def test_parse_chapters_needs_three(self):
        self.assertEqual(llms.parse_chapters("00:00 Intro\n01:00 End"), "")

    @patch("youtube_to_docs.llms._query_llm")
    def test_generate_chapters(self, mock_query):
        mock_query.return_value = ("0:00 A\n0:30 B\n1:00 C", 10, 5)
        text, inp, out = llms.generate_chapters("m", "1\n00:00:00,000 --> ...")
        self.assertEqual(text, "00:00 A\n00:30 B\n01:00 C")
        self.assertEqual((inp, out), (10, 5))
        self.assertIn("SRT", mock_query.call_args.args[1])

    def test_tags_to_hashtags(self):
        self.assertEqual(
            llms.tags_to_hashtags(
                "economic development, Appalachian Regional Commission, "
                "grant-implementation"
            ),
            "#economicdevelopment #AppalachianRegionalCommission #grantimplementation",
        )

    def test_to_plain_text(self):
        self.assertEqual(
            llms.to_plain_text(
                "# **Title**\n\n---\n1. **Role:**\n   * ARISE is [big](http://x).\n"
                "   - Item `two`\n\n\n\nEnd"
            ),
            "Title\n\nRole:\nARISE is big.\nItem two\n\nEnd",
        )

    @patch("youtube_to_docs.llms._query_llm")
    def test_generate_description_text_is_plain(self, mock_query):
        mock_query.return_value = ("**Bold** start.\n\n* A point", 3, 2)
        text, inp, out = llms.generate_description_text("m", "summary")
        self.assertEqual(text, "Bold start.\n\nA point")
        self.assertIn("plain text", mock_query.call_args.args[1])

    def test_build_description(self):
        self.assertEqual(
            llms.build_description(
                "The summary.\n", "00:00 Intro\n01:00 End", "a b, c"
            ),
            "The summary.\n\nChapters:\n00:00 Intro\n01:00 End\n\n#ab #c\n",
        )
        self.assertEqual(llms.build_description("S", "", ""), "S\n")


if __name__ == "__main__":
    unittest.main()
