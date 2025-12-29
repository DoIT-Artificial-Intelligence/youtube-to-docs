import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from youtube_to_docs import infographic


class TestInfographic(unittest.TestCase):
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

    @patch("youtube_to_docs.infographic.genai.Client")
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
            "gemini-2.5-flash-image", "Summary text", "Video Title"
        )

        self.assertEqual(image_bytes, b"fake_gemini_bytes")
        self.assertEqual(in_tok, 10)
        self.assertEqual(out_tok, 20)
        mock_client.models.generate_content_stream.assert_called_once()

    @patch("youtube_to_docs.infographic.genai.Client")
    def test_generate_infographic_imagen(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()

        # Mocking the response structure for generate_images
        mock_image = MagicMock()
        mock_image.image.image_bytes = b"fake_imagen_bytes"
        mock_resp.generated_images = [mock_image]

        mock_client.models.generate_images.return_value = mock_resp

        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "imagen-4.0-generate-001", "Summary text", "Video Title"
        )

        self.assertEqual(image_bytes, b"fake_imagen_bytes")
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 1000)
        mock_client.models.generate_images.assert_called_once()

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

    @patch("youtube_to_docs.infographic.genai.Client")
    def test_generate_infographic_imagen_no_images(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_resp = MagicMock()
        mock_resp.generated_images = []
        mock_client.models.generate_images.return_value = mock_resp

        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "imagen-4.0-generate-001", "Summary text", "Video Title"
        )
        self.assertIsNone(image_bytes)
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 0)

    @patch("youtube_to_docs.infographic.genai.Client")
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
            "gemini-2.5-flash-image", "Summary text", "Video Title"
        )
        self.assertIsNone(image_bytes)
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 0)

    @patch("youtube_to_docs.infographic.requests.post")
    def test_generate_infographic_bedrock(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # "fake_bytes" in base64 is "ZmFrZV9ieXRlcw=="
        mock_resp.json.return_value = {"images": ["ZmFrZV9ieXRlcw=="]}
        mock_post.return_value = mock_resp

        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "bedrock-titan-image-generator-v2", "Summary text", "Video Title"
        )

        self.assertEqual(image_bytes, b"fake_bytes")
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 1000)
        mock_post.assert_called_once()
        # Check if actual_model_id was mapped correctly
        args, kwargs = mock_post.call_args
        self.assertIn("amazon.titan-image-generator-v2:0", args[0])

    @patch("youtube_to_docs.infographic.requests.post")
    def test_generate_infographic_bedrock_nova(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"images": ["ZmFrZV9ieXRlcw=="]}
        mock_post.return_value = mock_resp

        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "bedrock-nova-canvas-v1", "Summary text", "Video Title"
        )

        self.assertEqual(image_bytes, b"fake_bytes")
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 4000)
        mock_post.assert_called_once()
        # Check if actual_model_id was mapped correctly
        args, kwargs = mock_post.call_args
        self.assertIn("amazon.nova-canvas-v1:0", args[0])

    @patch("youtube_to_docs.infographic.requests.post")
    def test_generate_infographic_bedrock_with_suffix(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"images": ["ZmFrZV9ieXRlcw=="]}
        mock_post.return_value = mock_resp

        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "bedrock-nova-canvas-v1:0", "Summary text", "Video Title"
        )

        self.assertEqual(image_bytes, b"fake_bytes")
        mock_post.assert_called_once()
        # Check that it didn't double the :0
        args, kwargs = mock_post.call_args
        self.assertIn("amazon.nova-canvas-v1:0", args[0])
        self.assertNotIn("amazon.nova-canvas-v1:0:0", args[0])

    @patch("youtube_to_docs.infographic.requests.post")
    def test_generate_infographic_bedrock_skip_long_prompt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"images": ["ZmFrZV9ieXRlcw=="]}
        mock_post.return_value = mock_resp

        long_summary = "A" * 2000
        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            "amazon.nova-canvas-v1:0", long_summary, "Video Title"
        )

        self.assertIsNone(image_bytes)
        self.assertEqual(in_tok, 0)
        self.assertEqual(out_tok, 0)
        mock_post.assert_not_called()

    @patch("youtube_to_docs.infographic.OpenAI")
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
    unittest.main()
