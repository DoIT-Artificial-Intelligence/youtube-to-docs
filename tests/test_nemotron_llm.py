import os
import unittest
from unittest.mock import patch

import requests

from youtube_to_docs import llms, main, utils
from youtube_to_docs.nemotron_llm import (
    NEMOTRON_LIGHTNING_MODEL,
    NEMOTRON_SUPER_MODEL,
    NemotronLlmProvider,
    NemotronSetupError,
    _served_model_cache,
    best_llm_precision,
    check_nemotron_llm_hardware,
    check_nemotron_llm_server,
    generate_nemotron_llm,
    get_nemotron_llm_spec,
)
from youtube_to_docs.nemotron_transcribe import GpuInfo
from youtube_to_docs.providers import get_provider

B200 = GpuInfo("NVIDIA B200", 183359, 10.0)
H200 = GpuInfo("NVIDIA H200", 143771, 9.0)
RTX_PRO_6000 = GpuInfo("NVIDIA RTX PRO 6000 Blackwell", 97887, 12.0)
H100 = GpuInfo("NVIDIA H100 80GB HBM3", 81559, 9.0)
A100_80 = GpuInfo("NVIDIA A100-SXM4-80GB", 81920, 8.0)
A100_40 = GpuInfo("NVIDIA A100-SXM4-40GB", 40960, 8.0)
L40S = GpuInfo("NVIDIA L40S", 46068, 8.9)
RTX_5090 = GpuInfo("NVIDIA GeForce RTX 5090", 32607, 12.0)
L4 = GpuInfo("NVIDIA L4", 23034, 8.9)
T4 = GpuInfo("Tesla T4", 15360, 7.5)


class TestBestLlmPrecision(unittest.TestCase):
    def test_supported_gpus(self):
        spec = get_nemotron_llm_spec(NEMOTRON_LIGHTNING_MODEL)
        self.assertEqual(best_llm_precision(H100, spec), "BF16")
        self.assertEqual(best_llm_precision(A100_80, spec), "BF16")
        self.assertEqual(best_llm_precision(L40S, spec), "NVFP4")
        self.assertEqual(best_llm_precision(RTX_5090, spec), "NVFP4")
        self.assertEqual(best_llm_precision(L4, spec), "NVFP4")

    def test_too_small_or_too_old(self):
        spec = get_nemotron_llm_spec(NEMOTRON_LIGHTNING_MODEL)
        self.assertIsNone(best_llm_precision(A100_40, spec))
        self.assertIsNone(best_llm_precision(T4, spec))
        self.assertIsNone(best_llm_precision(GpuInfo("old", 81920, 7.0), spec))


class TestCheckLlmHardware(unittest.TestCase):
    def test_ok_l4(self):
        ok, message = check_nemotron_llm_hardware(
            NEMOTRON_LIGHTNING_MODEL, [L4], platform="linux"
        )
        self.assertTrue(ok)
        self.assertIn("NVIDIA L4", message)
        self.assertIn("NVFP4", message)

    def test_picks_best_gpu(self):
        ok, message = check_nemotron_llm_hardware(
            NEMOTRON_LIGHTNING_MODEL, [T4, L4, H100], platform="linux"
        )
        self.assertTrue(ok)
        self.assertIn("H100", message)
        self.assertIn("BF16", message)

    def test_too_small(self):
        ok, message = check_nemotron_llm_hardware(
            NEMOTRON_LIGHTNING_MODEL, [T4], platform="linux"
        )
        self.assertFalse(ok)
        self.assertIn("Tesla T4 (15 GB, compute capability 7.5)", message)
        self.assertIn("too small", message)

    def test_no_driver(self):
        with patch("youtube_to_docs.nemotron_llm.detect_gpus", return_value=None):
            ok, message = check_nemotron_llm_hardware(
                NEMOTRON_LIGHTNING_MODEL, platform="linux"
            )
        self.assertFalse(ok)
        self.assertIn("nvidia-smi", message)

    def test_no_gpus(self):
        ok, message = check_nemotron_llm_hardware(
            NEMOTRON_LIGHTNING_MODEL, [], platform="linux"
        )
        self.assertFalse(ok)
        self.assertIn("found no GPUs", message)

    def test_needs_linux(self):
        ok, message = check_nemotron_llm_hardware(
            NEMOTRON_LIGHTNING_MODEL, [H100], platform="win32"
        )
        self.assertFalse(ok)
        self.assertIn("Linux", message)


class TestMainLlmHardwareCheck(unittest.TestCase):
    args = ["dQw4w9WgXcQ", "-m", NEMOTRON_LIGHTNING_MODEL, "-o", "none"]

    @patch("youtube_to_docs.main.resolve_video_ids", return_value=[])
    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.check_nemotron_llm_server")
    @patch(
        "youtube_to_docs.main.check_nemotron_llm_hardware",
        return_value=(False, "nvidia-smi found no GPUs."),
    )
    def test_no_gpu_falls_back_to_hosted(
        self, _, mock_server, mock_youtube, _mock_resolve
    ):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEMOTRON_LLM_URL", None)
            main.main(self.args)
        mock_server.assert_not_called()
        mock_youtube.assert_called_once()

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch(
        "youtube_to_docs.main.check_nemotron_llm_server",
        return_value=(False, "No Nemotron LLM server at http://127.0.0.1:8000/v1"),
    )
    @patch(
        "youtube_to_docs.main.check_nemotron_llm_hardware",
        return_value=(True, "NVIDIA L4 can run it in NVFP4."),
    )
    def test_gpu_present_needs_a_server(self, _, __, mock_youtube):
        with (
            patch.dict(os.environ, {}, clear=False),
            self.assertRaisesRegex(NemotronSetupError, "No Nemotron LLM server"),
        ):
            os.environ.pop("NEMOTRON_LLM_URL", None)
            main.main(self.args)
        mock_youtube.assert_not_called()

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch(
        "youtube_to_docs.main.check_nemotron_llm_server",
        return_value=(False, "No Nemotron LLM server at http://gpu-box:8000/v1"),
    )
    @patch("youtube_to_docs.main.check_nemotron_llm_hardware")
    def test_remote_server_skips_hardware_check(self, mock_hardware, _, __):
        with (
            patch.dict(os.environ, {"NEMOTRON_LLM_URL": "http://gpu-box:8000/v1"}),
            self.assertRaises(NemotronSetupError),
        ):
            main.main(self.args)
        mock_hardware.assert_not_called()

    @patch(
        "youtube_to_docs.main.check_nemotron_llm_server",
        return_value=(False, "No Nemotron LLM server at http://127.0.0.1:8000/v1"),
    )
    @patch(
        "youtube_to_docs.main.check_nemotron_llm_hardware",
        return_value=(True, "NVIDIA L4 can run it in NVFP4."),
    )
    def test_cli_exits_cleanly_when_server_missing(self, _, __):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("sys.argv", ["youtube-to-docs", *self.args]),
            self.assertRaises(SystemExit) as cm,
        ):
            os.environ.pop("NEMOTRON_LLM_URL", None)
            main.main()
        self.assertEqual(cm.exception.code, 1)


class TestLlmServer(unittest.TestCase):
    def setUp(self):
        _served_model_cache.clear()

    @patch("youtube_to_docs.nemotron_llm.requests.get")
    def test_server_ok_nvfp4(self, mock_get):
        mock_get.return_value.json.return_value = {
            "data": [{"id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"}]
        }
        with patch.dict(os.environ, {"NEMOTRON_LLM_URL": "http://box:8000/v1/"}):
            ok, message = check_nemotron_llm_server(NEMOTRON_LIGHTNING_MODEL)
        self.assertTrue(ok)
        self.assertIn("NVFP4", message)
        mock_get.assert_called_once_with("http://box:8000/v1/models", timeout=10)

    @patch("youtube_to_docs.nemotron_llm.requests.get")
    def test_server_ok_bf16(self, mock_get):
        mock_get.return_value.json.return_value = {
            "data": [{"id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"}]
        }
        ok, message = check_nemotron_llm_server(NEMOTRON_LIGHTNING_MODEL)
        self.assertTrue(ok)
        self.assertIn("BF16", message)

    @patch("youtube_to_docs.nemotron_llm.requests.get")
    def test_server_serves_another_model(self, mock_get):
        mock_get.return_value.json.return_value = {"data": [{"id": "other"}]}
        ok, message = check_nemotron_llm_server(NEMOTRON_LIGHTNING_MODEL)
        self.assertFalse(ok)
        self.assertIn("serves other", message)

    @patch(
        "youtube_to_docs.nemotron_llm.requests.get",
        side_effect=requests.ConnectionError("refused"),
    )
    def test_server_down(self, _):
        ok, message = check_nemotron_llm_server(NEMOTRON_LIGHTNING_MODEL)
        self.assertFalse(ok)
        self.assertIn("refused", message)


class TestGenerateNemotronLlm(unittest.TestCase):
    def setUp(self):
        _served_model_cache.clear()

    @patch("youtube_to_docs.nemotron_llm.requests.post")
    @patch("youtube_to_docs.nemotron_llm.requests.get")
    def test_generate_content_self_hosted(self, mock_get, mock_post):
        mock_get.return_value.json.return_value = {
            "data": [{"id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"}]
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "thinking...",
                        "content": "Final summary.",
                    }
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 45},
        }
        with patch.dict(os.environ, {"NEMOTRON_LLM_URL": "http://box:8000/v1"}):
            provider = get_provider(NEMOTRON_LIGHTNING_MODEL)
            assert isinstance(provider, NemotronLlmProvider)
            text, in_tok, out_tok = provider.generate_content("Summarize this.")

        self.assertEqual(text, "Final summary.")
        self.assertEqual((in_tok, out_tok), (120, 45))
        self.assertEqual(
            mock_post.call_args.args[0], "http://box:8000/v1/chat/completions"
        )
        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(
            body["model"], "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
        )
        self.assertEqual(
            body["messages"], [{"role": "user", "content": "Summarize this."}]
        )

    @patch(
        "youtube_to_docs.llms._nvidia_model_ids",
        ["nvidia/nemotron-3.5-lightning-30b-a3b"],
    )
    @patch("youtube_to_docs.llms.requests.post")
    @patch(
        "youtube_to_docs.nemotron_llm.check_nemotron_llm_hardware",
        return_value=(False, "No NVIDIA driver found."),
    )
    def test_provider_falls_back_to_build_nvidia_when_no_gpu(self, _, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Hosted summary."}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 30},
        }
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "fake_key"}, clear=False):
            os.environ.pop("NEMOTRON_LLM_URL", None)
            provider = get_provider(NEMOTRON_LIGHTNING_MODEL)
            assert isinstance(provider, NemotronLlmProvider)
            text, in_tok, out_tok = provider.generate_content("Summarize this.")

        self.assertEqual(text, "Hosted summary.")
        self.assertEqual((in_tok, out_tok), (80, 30))
        self.assertEqual(
            mock_post.call_args.args[0],
            f"{llms.NVIDIA_BASE_URL}/chat/completions",
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["model"],
            "nvidia/nemotron-3.5-lightning-30b-a3b",
        )

    @patch("youtube_to_docs.nemotron_llm.requests.post")
    @patch(
        "youtube_to_docs.nemotron_llm.requests.get",
        side_effect=requests.ConnectionError("offline"),
    )
    def test_http_error_returns_error_string(self, _, mock_post):
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Error"
        text, in_tok, out_tok = generate_nemotron_llm(
            NEMOTRON_LIGHTNING_MODEL, "prompt"
        )
        self.assertTrue(text.startswith("Error: Nemotron LLM server error 500"))
        self.assertEqual((in_tok, out_tok), (0, 0))

    @patch(
        "youtube_to_docs.nemotron_llm.requests.post",
        side_effect=requests.ConnectionError("refused"),
    )
    @patch(
        "youtube_to_docs.nemotron_llm.requests.get",
        side_effect=requests.ConnectionError("refused"),
    )
    def test_connection_error(self, _, __):
        text, in_tok, out_tok = generate_nemotron_llm(
            NEMOTRON_LIGHTNING_MODEL, "prompt"
        )
        self.assertEqual(text, "Error: Nemotron LLM server failed: refused")
        self.assertEqual((in_tok, out_tok), (0, 0))

    def test_provider_routing_pricing_and_normalization(self):
        provider = get_provider(NEMOTRON_LIGHTNING_MODEL)
        self.assertIsInstance(provider, NemotronLlmProvider)
        self.assertEqual(llms.get_model_pricing(NEMOTRON_LIGHTNING_MODEL), (0, 0))
        self.assertEqual(
            utils.normalize_model_name(NEMOTRON_LIGHTNING_MODEL),
            "nemotron-3.5-lightning",
        )


class TestNemotronSuper(unittest.TestCase):
    def setUp(self):
        _served_model_cache.clear()

    def test_hardware_single_and_multi_gpu(self):
        ok, msg = check_nemotron_llm_hardware(
            NEMOTRON_SUPER_MODEL, [RTX_PRO_6000], platform="linux"
        )
        self.assertTrue(ok)
        self.assertIn("NVFP4", msg)

        ok, msg = check_nemotron_llm_hardware(
            NEMOTRON_SUPER_MODEL, [H200], platform="linux"
        )
        self.assertTrue(ok)
        self.assertIn("FP8", msg)

        ok, msg = check_nemotron_llm_hardware(
            NEMOTRON_SUPER_MODEL, [B200], platform="linux"
        )
        self.assertTrue(ok)
        self.assertIn("FP8", msg)

        ok, msg = check_nemotron_llm_hardware(
            NEMOTRON_SUPER_MODEL, [H100, H100], platform="linux"
        )
        self.assertTrue(ok)
        self.assertIn("2x", msg)
        self.assertIn("FP8", msg)

        ok, msg = check_nemotron_llm_hardware(
            NEMOTRON_SUPER_MODEL, [H100, H100, H100, H100], platform="linux"
        )
        self.assertTrue(ok)
        self.assertIn("4x", msg)
        self.assertIn("BF16", msg)

        ok, msg = check_nemotron_llm_hardware(
            NEMOTRON_SUPER_MODEL, [H100], platform="linux"
        )
        self.assertFalse(ok)
        self.assertIn("too small", msg)

    @patch("youtube_to_docs.nemotron_llm.requests.get")
    def test_server_ok_fp8(self, mock_get):
        mock_get.return_value.json.return_value = {
            "data": [{"id": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8"}]
        }
        ok, message = check_nemotron_llm_server(NEMOTRON_SUPER_MODEL)
        self.assertTrue(ok)
        self.assertIn("FP8", message)

    @patch(
        "youtube_to_docs.llms._nvidia_model_ids",
        ["nvidia/nemotron-3.5-super-vl-120b-a12b"],
    )
    @patch("youtube_to_docs.llms.requests.post")
    @patch(
        "youtube_to_docs.nemotron_llm.check_nemotron_llm_hardware",
        return_value=(False, "Found NVIDIA H100, which is too small."),
    )
    def test_falls_back_to_build_nvidia_hosted_super(self, _, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Super hosted summary."}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        }
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "fake_key"}, clear=False):
            os.environ.pop("NEMOTRON_LLM_URL", None)
            provider = get_provider(NEMOTRON_SUPER_MODEL)
            assert isinstance(provider, NemotronLlmProvider)
            text, in_tok, out_tok = provider.generate_content("Summarize.")

        self.assertEqual(text, "Super hosted summary.")
        self.assertEqual((in_tok, out_tok), (100, 40))
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["model"],
            "nvidia/nemotron-3.5-super-vl-120b-a12b",
        )

    def test_routing_pricing_and_normalization(self):
        provider = get_provider(NEMOTRON_SUPER_MODEL)
        self.assertIsInstance(provider, NemotronLlmProvider)
        self.assertEqual(llms.get_model_pricing(NEMOTRON_SUPER_MODEL), (0, 0))
        self.assertEqual(
            utils.normalize_model_name(NEMOTRON_SUPER_MODEL),
            "nemotron-3.5-super",
        )


if __name__ == "__main__":
    unittest.main()
