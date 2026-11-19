import os
import subprocess
import unittest
from unittest.mock import patch

import requests

from youtube_to_docs import main
from youtube_to_docs.nemotron_transcribe import (
    NEMOTRON_CHUNK_SECONDS,
    GpuInfo,
    NemotronSetupError,
    NemotronTranscribeProvider,
    best_precision,
    check_nemotron_hardware,
    check_nemotron_server,
    detect_gpus,
    transcribe_nemotron,
)
from youtube_to_docs.providers import get_provider

H100 = GpuInfo("NVIDIA H100 80GB HBM3", 81559, 9.0)
A100_80 = GpuInfo("NVIDIA A100-SXM4-80GB", 81920, 8.0)
A100_40 = GpuInfo("NVIDIA A100-SXM4-40GB", 40960, 8.0)
L40S = GpuInfo("NVIDIA L40S", 46068, 8.9)
RTX_5090 = GpuInfo("NVIDIA GeForce RTX 5090", 32607, 12.0)
L4 = GpuInfo("NVIDIA L4", 23034, 8.9)
T4 = GpuInfo("Tesla T4", 15360, 7.5)


class TestBestPrecision(unittest.TestCase):
    def test_model_card_gpus(self):
        self.assertEqual(best_precision(H100), "BF16")
        self.assertEqual(best_precision(A100_80), "BF16")
        self.assertEqual(best_precision(L40S), "FP8")
        self.assertEqual(best_precision(RTX_5090), "NVFP4")

    def test_too_small_or_too_old(self):
        self.assertIsNone(best_precision(A100_40))  # Ampere has no FP8.
        self.assertIsNone(best_precision(L4))
        self.assertIsNone(best_precision(T4))
        # 80 GB isn't enough without Ampere or newer.
        self.assertIsNone(best_precision(GpuInfo("old", 81920, 7.0)))


class TestCheckHardware(unittest.TestCase):
    def test_ok(self):
        ok, message = check_nemotron_hardware([H100], platform="linux")
        self.assertTrue(ok)
        self.assertIn("H100", message)
        self.assertIn("BF16", message)

    def test_picks_best_gpu(self):
        ok, message = check_nemotron_hardware([T4, L40S, H100], platform="linux")
        self.assertTrue(ok)
        self.assertIn("H100", message)

    def test_too_small(self):
        ok, message = check_nemotron_hardware([L4], platform="linux")
        self.assertFalse(ok)
        self.assertIn("NVIDIA L4 (22 GB, compute capability 8.9)", message)
        self.assertIn("too small", message)

    def test_no_driver(self):
        with patch(
            "youtube_to_docs.nemotron_transcribe.detect_gpus", return_value=None
        ):
            ok, message = check_nemotron_hardware(platform="linux")
        self.assertFalse(ok)
        self.assertIn("nvidia-smi", message)

    def test_no_gpus(self):
        ok, message = check_nemotron_hardware([], platform="linux")
        self.assertFalse(ok)
        self.assertIn("found no GPUs", message)

    def test_needs_linux(self):
        ok, message = check_nemotron_hardware([H100], platform="win32")
        self.assertFalse(ok)
        self.assertIn("Linux", message)


class TestDetectGpus(unittest.TestCase):
    @patch("youtube_to_docs.nemotron_transcribe.shutil.which", return_value=None)
    def test_no_nvidia_smi(self, _):
        self.assertIsNone(detect_gpus())

    @patch("youtube_to_docs.nemotron_transcribe.subprocess.run")
    @patch(
        "youtube_to_docs.nemotron_transcribe.shutil.which",
        return_value="/usr/bin/nvidia-smi",
    )
    def test_parses_csv(self, _, mock_run):
        mock_run.return_value.stdout = (
            "NVIDIA H100 80GB HBM3, 81559, 9.0\nNVIDIA L4, 23034, 8.9\ngarbage\n"
        )
        self.assertEqual(detect_gpus(), [H100, L4])

    @patch(
        "youtube_to_docs.nemotron_transcribe.subprocess.run",
        side_effect=subprocess.CalledProcessError(9, "nvidia-smi"),
    )
    @patch(
        "youtube_to_docs.nemotron_transcribe.shutil.which",
        return_value="/usr/bin/nvidia-smi",
    )
    def test_nvidia_smi_fails(self, _, __):
        self.assertEqual(detect_gpus(), [])


class TestMainHardwareCheck(unittest.TestCase):
    args = ["dQw4w9WgXcQ", "-t", "nvidia-nemotron-3.5-transcribe", "-o", "none"]

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch("youtube_to_docs.main.check_nemotron_server")
    @patch(
        "youtube_to_docs.main.check_nemotron_hardware",
        return_value=(False, "nvidia-smi found no GPUs."),
    )
    def test_stops_before_any_work(self, _, mock_server, mock_youtube):
        with (
            patch.dict(os.environ, {}, clear=False),
            self.assertRaisesRegex(NemotronSetupError, "found no GPUs"),
        ):
            os.environ.pop("NEMOTRON_TRANSCRIBE_URL", None)
            main.main(self.args)
        mock_server.assert_not_called()
        mock_youtube.assert_not_called()

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch(
        "youtube_to_docs.main.check_nemotron_server",
        return_value=(False, "No Nemotron server at http://127.0.0.1:8000/v1"),
    )
    @patch(
        "youtube_to_docs.main.check_nemotron_hardware",
        return_value=(True, "NVIDIA H100 can run it in BF16."),
    )
    def test_needs_a_server(self, _, __, mock_youtube):
        with (
            patch.dict(os.environ, {}, clear=False),
            self.assertRaisesRegex(NemotronSetupError, "No Nemotron server"),
        ):
            os.environ.pop("NEMOTRON_TRANSCRIBE_URL", None)
            main.main(self.args)
        mock_youtube.assert_not_called()

    @patch("youtube_to_docs.main.get_youtube_service")
    @patch(
        "youtube_to_docs.main.check_nemotron_server",
        return_value=(False, "No Nemotron server at http://gpu-box:8000/v1"),
    )
    @patch("youtube_to_docs.main.check_nemotron_hardware")
    def test_remote_server_skips_hardware_check(self, mock_hardware, _, __):
        with (
            patch.dict(
                os.environ, {"NEMOTRON_TRANSCRIBE_URL": "http://gpu-box:8000/v1"}
            ),
            self.assertRaises(NemotronSetupError),
        ):
            main.main(self.args)
        mock_hardware.assert_not_called()

    @patch(
        "youtube_to_docs.main.check_nemotron_hardware",
        return_value=(False, "nvidia-smi found no GPUs."),
    )
    def test_cli_exits_cleanly(self, _):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("sys.argv", ["youtube-to-docs", *self.args]),
            self.assertRaises(SystemExit) as cm,
        ):
            os.environ.pop("NEMOTRON_TRANSCRIBE_URL", None)
            main.main()
        self.assertEqual(cm.exception.code, 1)


class TestServer(unittest.TestCase):
    @patch("youtube_to_docs.nemotron_transcribe.requests.get")
    def test_server_ok(self, mock_get):
        mock_get.return_value.json.return_value = {
            "data": [{"id": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"}]
        }
        with patch.dict(os.environ, {"NEMOTRON_TRANSCRIBE_URL": "http://box:8000/v1/"}):
            ok, message = check_nemotron_server()
        self.assertTrue(ok)
        mock_get.assert_called_once_with("http://box:8000/v1/models", timeout=10)

    @patch("youtube_to_docs.nemotron_transcribe.requests.get")
    def test_server_serves_another_model(self, mock_get):
        mock_get.return_value.json.return_value = {"data": [{"id": "other"}]}
        ok, message = check_nemotron_server()
        self.assertFalse(ok)
        self.assertIn("serves other", message)

    @patch(
        "youtube_to_docs.nemotron_transcribe.requests.get",
        side_effect=requests.ConnectionError("refused"),
    )
    def test_server_down(self, _):
        ok, message = check_nemotron_server()
        self.assertFalse(ok)
        self.assertIn("refused", message)


def _response(content, words):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "ctc_timestamps": {
            "words": [
                {"word": w, "start": s, "end": e, "speaker": spk}
                for w, s, e, spk in words
            ]
        },
    }


class TestTranscribe(unittest.TestCase):
    def setUp(self):
        def fake_ffmpeg(cmd, check):
            with open(cmd[-1], "wb") as f:
                f.write(b"flac-bytes")

        patcher = patch(
            "youtube_to_docs.nemotron_transcribe.subprocess.run",
            side_effect=fake_ffmpeg,
        )
        self.mock_run = patcher.start()
        self.addCleanup(patcher.stop)

    @patch("youtube_to_docs.nemotron_transcribe._transcribe_chunk")
    def test_speakers_become_labels(self, mock_chunk):
        mock_chunk.return_value = _response(
            "<spk:0> Good morning. <spk:1> Thank you.",
            [
                ("Good", 1.0, 1.4, "0"),
                ("morning.", 1.5, 2.0, "0"),
                ("Thank", 2.5, 2.8, "1"),
                ("you.", 2.9, 3.2, "1"),
            ],
        )
        text, srt, in_tok, out_tok = transcribe_nemotron("a.m4a", "url", 30)

        self.assertEqual(text, "Good morning. Thank you.")
        self.assertEqual(
            srt,
            "1\n00:00:01,000 --> 00:00:02,000\n[Speaker 1] Good morning.\n\n"
            "2\n00:00:02,500 --> 00:00:03,200\n[Speaker 2] Thank you.\n",
        )
        self.assertEqual((in_tok, out_tok), (100, 20))
        self.assertNotIn("-ss", self.mock_run.call_args.args[0])

    @patch("youtube_to_docs.nemotron_transcribe._transcribe_chunk")
    def test_long_audio_is_chunked(self, mock_chunk):
        mock_chunk.side_effect = [
            _response("<spk:0> One.", [("One.", 1.0, 1.5, "0")]),
            _response("<spk:0> Two.", [("Two.", 2.0, 2.5, "0")]),
        ]
        text, srt, in_tok, _ = transcribe_nemotron(
            "a.m4a", "url", NEMOTRON_CHUNK_SECONDS + 60
        )

        self.assertEqual(text, "One. Two.")
        # The second chunk's times are offset, and its speaker 0 is a new label.
        self.assertIn("00:45:02,000 --> 00:45:02,500\n[Speaker 2] Two.", srt)
        self.assertEqual(in_tok, 200)
        second_cmd = self.mock_run.call_args_list[1].args[0]
        self.assertEqual(second_cmd[second_cmd.index("-ss") + 1], "2700")

    @patch("youtube_to_docs.nemotron_transcribe.requests.post")
    def test_request_body(self, mock_post):
        mock_post.return_value.json.return_value = _response(
            "<spk:0> Hi.", [("Hi.", 0.1, 0.3, "0")]
        )
        with patch.dict(os.environ, {"NEMOTRON_TRANSCRIBE_URL": "http://box:8000/v1"}):
            transcribe_nemotron("a.m4a", "url", 10)

        self.assertEqual(
            mock_post.call_args.args[0], "http://box:8000/v1/chat/completions"
        )
        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
        self.assertEqual(body["mm_processor_kwargs"], {"capture_ctc_timestamps": True})
        audio = body["messages"][0]["content"][1]["audio_url"]["url"]
        self.assertTrue(audio.startswith("data:audio/flac;base64,"))

    @patch(
        "youtube_to_docs.nemotron_transcribe._transcribe_chunk",
        side_effect=requests.ConnectionError("refused"),
    )
    def test_server_error(self, _):
        text, srt, _, _ = transcribe_nemotron("a.m4a", "url", 10)
        self.assertEqual(text, "Error: Nemotron server failed: refused")
        self.assertEqual(srt, "")

    def test_provider_routing(self):
        provider = get_provider("nvidia-nemotron-3.5-transcribe")
        self.assertIsInstance(provider, NemotronTranscribeProvider)


if __name__ == "__main__":
    unittest.main()
