import os
from unittest import TestCase, mock
from unittest.mock import MagicMock, patch

from youtube_to_docs import infographic


class TestInfographic(TestCase):
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "fake_gemini_key",
                "AWS_BEARER_TOKEN_BEDROCK": "fake_bedrock_token",
                "AZURE_FOUNDRY_ENDPOINT": "fake_endpoint",
                "AZURE_FOUNDRY_API_KEY": "fake_foundry_key",
            },
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("google.genai.Client")
    def test_generate_infographic_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value

        # Mocking the stream response for generate_content_stream
        mock_chunk = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"fake_gemini_bytes"
        mock_chunk.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_chunk.usage_metadata.prompt_token_count = 10
        mock_chunk.usage_metadata.candidates_token_count = 20

        mock_client.models.generate_content_stream.return_value = [mock_chunk]

        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "gemini-nano-banana-2.1", "Summary text", "Video Title"
        )

        self.assertEqual(image_bytes, b"fake_gemini_bytes")
        self.assertEqual(in_tok, 10)
        self.assertEqual(out_tok, 20)
        mock_client.models.generate_content_stream.assert_called_once()
        config = mock_client.models.generate_content_stream.call_args.kwargs["config"]
        self.assertEqual(config.image_config.aspect_ratio, "16:9")

    def test_image_file_extension(self):
        self.assertEqual(infographic.image_file_extension(b"\xff\xd8\xff\xe0"), ".jpg")
        self.assertEqual(infographic.image_file_extension(b"\x89PNG\r\n\x1a\n"), ".png")
        self.assertEqual(
            infographic.image_file_extension(b"RIFF\x00\x00\x00\x00WEBPVP8 "), ".webp"
        )

    def test_build_infographic_prompt(self):
        prompt = infographic.build_infographic_prompt(
            "Summary text", "Video Title", language="es"
        )
        self.assertIn("Video Title: Video Title", prompt)
        self.assertIn("Summary:\nSummary text", prompt)
        self.assertIn("in es.", prompt)
        self.assertIn("Do not include any people", prompt)

    def test_build_infographic_prompt_default_language(self):
        prompt = infographic.build_infographic_prompt("Summary text", "Video Title")
        self.assertIn("in en.", prompt)

    @patch("google.genai.Client")
    def test_generate_infographic_uses_build_prompt(self, mock_client_cls):
        # The prompt sent to the model must match build_infographic_prompt
        # so the persisted prompt file is faithful.
        mock_client = mock_client_cls.return_value
        mock_chunk = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"fake_gemini_bytes"
        mock_chunk.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_chunk.usage_metadata.prompt_token_count = 10
        mock_chunk.usage_metadata.candidates_token_count = 20
        mock_client.models.generate_content_stream.return_value = [mock_chunk]

        infographic.generate_infographic(
            "gemini-nano-banana-2.1", "Summary text", "Video Title"
        )

        _, kwargs = mock_client.models.generate_content_stream.call_args
        sent_text = kwargs["contents"][0].parts[0].text
        expected = infographic.build_infographic_prompt("Summary text", "Video Title")
        self.assertEqual(sent_text, expected)

    def test_generate_infographic_none_model(self):
        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            None, "Summary text", "Video Title"
        )
        self.assertIsNone(image_bytes)
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 0)

    def test_generate_infographic_unsupported_model(self):
        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "unsupported-model", "Summary text", "Video Title"
        )
        self.assertIsNone(image_bytes)
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 0)

    @patch("google.genai.Client")
    def test_generate_infographic_gemini_no_data(self, mock_client_cls):
        mock_client = mock_client_cls.return_value

        # Mocking a stream with no inline data
        mock_chunk = MagicMock()
        mock_chunk.candidates = [
            MagicMock(content=MagicMock(parts=[MagicMock(inline_data=None)]))
        ]
        mock_chunk.usage_metadata = None

        mock_client.models.generate_content_stream.return_value = [mock_chunk]

        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "gemini-nano-banana-2.1", "Summary text", "Video Title"
        )
        self.assertIsNone(image_bytes)
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 0)

    def test_generate_infographic_retired_bedrock_models(self):
        # Titan Image Generator and Nova Canvas reached end of life on Bedrock.
        for model in ["bedrock-titan-image-generator-v2:0", "bedrock-nova-canvas-v1"]:
            with self.subTest(model=model):
                self.assertEqual(
                    infographic.generate_infographic(
                        model, "Summary text", "Video Title"
                    ),
                    (None, 0, 0),
                )

    @patch("openai.OpenAI")
    def test_generate_infographic_foundry(self, mock_openai_cls):
        mock_client = mock_openai_cls.return_value
        mock_resp = MagicMock()
        mock_image = MagicMock()
        mock_image.b64_json = "ZmFrZV9ieXRlcw=="
        mock_resp.data = [mock_image]
        mock_client.images.generate.return_value = mock_resp

        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "foundry-gpt-image-1.5", "Summary text", "Video Title"
        )

        self.assertEqual(image_bytes, b"fake_bytes")
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 3400)
        mock_client.images.generate.assert_called_once_with(
            model="gpt-image-1.5",
            prompt=mock.ANY,
            n=1,
            size="1536x1024",
            response_format="b64_json",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
