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

    @patch("google.auth.transport.requests.AuthorizedSession")
    @patch("google.auth.default")
    def test_generate_summary_vertex(self, mock_auth, mock_authed_session):
        mock_creds = MagicMock()
        mock_creds.token = "fake_token"
        mock_creds.expired = False
        mock_auth.return_value = (mock_creds, "proj")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"text": "Vertex Summary"}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        # Mock the post method of the AuthorizedSession instance.
        mock_session_instance = mock_authed_session.return_value
        mock_session_instance.post.return_value = mock_resp

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
            "gemini-pro", "transcript content", "Speaker 1, Speaker 2"
        )
        # Expecting the added column
        expected_qa = "| question number | Q | A |\n|---|---|---|\n| 1 | Q1 | A1 |"
        self.assertEqual(qa, expected_qa)
        self.assertEqual(in_tokens, 150)
        self.assertEqual(out_tokens, 60)


class TestPricing(unittest.TestCase):
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
                {"id": "gemini-3-flash-preview", "input": 0.5, "output": 3.0},
                {"id": "gpt-5-mini", "input": 0.25, "output": 2.0},
                {
                    "id": "amazon-nova-2-lite",
                    "input": 0.3,
                    "output": 2.5,
                },
                {
                    "id": "imagen-4",
                    "input": 0.0,
                    "output": 40.0,
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
            "gemini-3-flash-preview",
            "vertex-claude-haiku-4-5@20251001",
            "bedrock-claude-haiku-4-5-20251001-v1",
            "bedrock-nova-2-lite-v1",
            "foundry-gpt-5-mini",
            "imagen-4",
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


if __name__ == "__main__":
    unittest.main()
