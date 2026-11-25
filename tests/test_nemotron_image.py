import base64
import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from youtube_to_docs import infographic, main
from youtube_to_docs.nemotron_image import (
    NEMOTRON_IMAGE_DEFAULT_SIZE,
    NEMOTRON_IMAGE_HF_ID,
    NEMOTRON_IMAGE_MODEL,
    NemotronSetupError,
    check_nemotron_image_hardware,
    check_nemotron_image_server,
    generate_nemotron_image,
)
from youtube_to_docs.nemotron_transcribe import GpuInfo

H100 = GpuInfo("NVIDIA H100 80GB HBM3", 81559, 9.0)
A100_80 = GpuInfo("NVIDIA A100-SXM4-80GB", 81920, 8.0)
A100_40 = GpuInfo("NVIDIA A100-SXM4-40GB", 40960, 8.0)
L4 = GpuInfo("NVIDIA L4", 23034, 8.9)
T4 = GpuInfo("Tesla T4", 15360, 7.5)


class TestCheckHardware(unittest.TestCase):
    def test_single_80gb_gpu_ok(self):
        ok, message = check_nemotron_image_hardware([H100], platform="linux")
        self.assertTrue(ok)
        self.assertIn("H100", message)
        self.assertIn("TP=1", message)

    def test_two_a100_40gb_ok(self):
        ok, message = check_nemotron_image_hardware(
            [A100_40, A100_40], platform="linux"
        )
        self.assertTrue(ok)
        self.assertIn("2x NVIDIA A100-SXM4-40GB", message)
        self.assertIn("TP=2", message)

    def test_four_l4_ok(self):
        ok, message = check_nemotron_image_hardware([L4, L4, L4, L4], platform="linux")
        self.assertTrue(ok)
        self.assertIn("4x NVIDIA L4", message)
        self.assertIn("TP=4", message)

    def test_single_40gb_too_small(self):
        ok, message = check_nemotron_image_hardware([A100_40], platform="linux")
        self.assertFalse(ok)
        self.assertIn("too small", message)

    def test_two_l4_too_small(self):
        ok, message = check_nemotron_image_hardware([L4, L4], platform="linux")
        self.assertFalse(ok)
        self.assertIn("too small", message)

    def test_old_architecture_rejected(self):
        ok, message = check_nemotron_image_hardware([T4] * 8, platform="linux")
        self.assertFalse(ok)
        self.assertIn("bfloat16", message)

    def test_no_driver(self):
        with patch("youtube_to_docs.nemotron_image.detect_gpus", return_value=None):
            ok, message = check_nemotron_image_hardware(platform="linux")
        self.assertFalse(ok)
        self.assertIn("nvidia-smi", message)

    def test_no_gpus(self):
        ok, message = check_nemotron_image_hardware([], platform="linux")
        self.assertFalse(ok)
        self.assertIn("found no GPUs", message)

    def test_needs_linux(self):
        ok, message = check_nemotron_image_hardware([H100], platform="win32")
        self.assertFalse(ok)
        self.assertIn("Linux", message)


class TestServer(unittest.TestCase):
    @patch("youtube_to_docs.nemotron_image.requests.get")
    def test_server_ok(self, mock_get):
        mock_get.return_value.json.return_value = {
            "data": [{"id": NEMOTRON_IMAGE_HF_ID}]
        }
        with patch.dict(os.environ, {"NEMOTRON_IMAGE_URL": "http://box:8001/v1/"}):
            ok, message = check_nemotron_image_server()
        self.assertTrue(ok)
        mock_get.assert_called_once_with("http://box:8001/v1/models", timeout=10)

    @patch("youtube_to_docs.nemotron_image.requests.get")
    def test_server_serves_another_model(self, mock_get):
        mock_get.return_value.json.return_value = {"data": [{"id": "other"}]}
        ok, message = check_nemotron_image_server()
        self.assertFalse(ok)
        self.assertIn("serves other", message)

    @patch(
        "youtube_to_docs.nemotron_image.requests.get",
        side_effect=requests.ConnectionError("refused"),
    )
    def test_server_down(self, _):
        ok, message = check_nemotron_image_server()
        self.assertFalse(ok)
        self.assertIn("refused", message)


class TestGenerateImage(unittest.TestCase):
    @patch("youtube_to_docs.nemotron_image.requests.post")
    def test_generate_nemotron_image_success(self, mock_post):
        png_bytes = b"\x89PNG\r\n\x1a\nfake"
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"b64_json": base64.b64encode(png_bytes).decode("ascii")}],
            "usage": {"input_tokens": 120, "output_tokens": 4120},
        }
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"NEMOTRON_IMAGE_URL": "http://box:8001/v1"}):
            image_bytes, in_tok, out_tok = generate_nemotron_image("Draw a chart")

        self.assertEqual(image_bytes, png_bytes)
        self.assertEqual((in_tok, out_tok), (120, 4120))
        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args.args[0], "http://box:8001/v1/images/generations"
        )
        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(body["model"], NEMOTRON_IMAGE_HF_ID)
        self.assertEqual(body["prompt"], "Draw a chart")
        self.assertEqual(body["size"], NEMOTRON_IMAGE_DEFAULT_SIZE)

    @patch("youtube_to_docs.nemotron_image.requests.post")
    def test_infographic_routing(self, mock_post):
        png_bytes = b"\x89PNG\r\n\x1a\ninfographic"
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"b64_json": base64.b64encode(png_bytes).decode("ascii")}],
            "usage": {"input_tokens": 50, "output_tokens": 4056},
        }
        mock_post.return_value = mock_resp

        image_bytes, in_tok, out_tok = infographic.generate_infographic(
            NEMOTRON_IMAGE_MODEL, "Summary text", "Video Title"
        )
        self.assertEqual(image_bytes, png_bytes)
        self.assertEqual((in_tok, out_tok), (50, 4056))
        sent_prompt = mock_post.call_args.kwargs["json"]["prompt"]
        self.assertEqual(
            sent_prompt,
            infographic.build_infographic_prompt("Summary text", "Video Title"),
        )

    @patch(
        "youtube_to_docs.nemotron_image.requests.post",
        side_effect=requests.ConnectionError("refused"),
    )
    def test_generate_error_returns_none(self, _):
        self.assertEqual(generate_nemotron_image("Draw a chart"), (None, 0, 0))

    @patch("youtube_to_docs.nemotron_image.requests.post")
    def test_generate_empty_data_returns_none(self, mock_post):
        mock_post.return_value.json.return_value = {"data": []}
        self.assertEqual(generate_nemotron_image("Draw a chart"), (None, 0, 0))


class TestMainPreflightCheck(unittest.TestCase):
    args = ["dQw4w9WgXcQ", "-i", "nemotron-image", "-o", "none"]

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.check_nemotron_image_server")
    @patch(
        "youtube_to_docs.main.check_nemotron_image_hardware",
        return_value=(False, "nvidia-smi found no GPUs."),
    )
    def test_stops_before_any_work(self, _, mock_server, mock_youtube):
        with (
            patch.dict(os.environ, {}, clear=False),
            self.assertRaisesRegex(NemotronSetupError, "found no GPUs"),
        ):
            os.environ.pop("NEMOTRON_IMAGE_URL", None)
            main.main(self.args)
        mock_server.assert_not_called()
        mock_youtube.assert_not_called()

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch(
        "youtube_to_docs.main.check_nemotron_image_server",
        return_value=(False, "No Nemotron image server at http://127.0.0.1:8001/v1"),
    )
    @patch(
        "youtube_to_docs.main.check_nemotron_image_hardware",
        return_value=(True, "1x NVIDIA H100 can run it."),
    )
    def test_needs_a_server(self, _, __, mock_youtube):
        with (
            patch.dict(os.environ, {}, clear=False),
            self.assertRaisesRegex(NemotronSetupError, "No Nemotron image server"),
        ):
            os.environ.pop("NEMOTRON_IMAGE_URL", None)
            main.main(self.args)
        mock_youtube.assert_not_called()

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch(
        "youtube_to_docs.main.check_nemotron_image_server",
        return_value=(False, "No Nemotron image server at http://gpu-box:8001/v1"),
    )
    @patch("youtube_to_docs.main.check_nemotron_image_hardware")
    def test_remote_server_skips_hardware_check(self, mock_hardware, _, __):
        with (
            patch.dict(os.environ, {"NEMOTRON_IMAGE_URL": "http://gpu-box:8001/v1"}),
            self.assertRaises(NemotronSetupError),
        ):
            main.main(self.args)
        mock_hardware.assert_not_called()

    @patch(
        "youtube_to_docs.main.check_nemotron_image_hardware",
        return_value=(False, "nvidia-smi found no GPUs."),
    )
    def test_cli_exits_cleanly(self, _):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("sys.argv", ["youtube-to-docs", *self.args]),
            self.assertRaises(SystemExit) as cm,
        ):
            os.environ.pop("NEMOTRON_IMAGE_URL", None)
            main.main()
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
