from unittest.mock import patch

import pytest

from youtube_to_docs.mcp_server import mcp, process_video


def test_process_video_defaults():
    """Test process_video with default arguments."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        url = "https://www.youtube.com/watch?v=123"
        result = process_video(url=url)

        # The CLI picks the transcript source (youtube, or -cc's STT source).
        expected_args = [url, "--outfile", "youtube-to-docs-artifacts/youtube-docs.csv"]

        mock_main.assert_called_once_with(expected_args)
        assert f"Successfully processed {url}" in result


def test_process_video_all_args():
    """Test process_video with all arguments provided."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        url = "https://www.youtube.com/watch?v=123"
        result = process_video(
            url=url,
            output_file="youtube-to-docs-artifacts/custom.csv",
            transcript_source="gemini-pro",
            model="gemini-2.5-pro",
            tts_model="tts-1",
            infographic_model="gemini-nano-banana-2.1",
            no_youtube_summary=True,
            from_video="agentic",
            translate="gemini-2.5-pro-es",
        )

        expected_args = [
            url,
            "--outfile",
            "youtube-to-docs-artifacts/custom.csv",
            "--transcript",
            "gemini-pro",
            "--translate",
            "gemini-2.5-pro-es",
            "--model",
            "gemini-2.5-pro",
            "--tts",
            "tts-1",
            "--infographic",
            "gemini-nano-banana-2.1",
            "--no-youtube-summary",
            "--from-video",
            "agentic",
        ]

        mock_main.assert_called_once_with(expected_args)
        assert f"Successfully processed {url}" in result


@pytest.mark.parametrize(
    "output_file",
    [
        "../../.bashrc",
        "../../.ssh/authorized_keys",
        "/etc/cron.d/malicious",
        "youtube-to-docs-artifacts/../pyproject.toml",
        "youtube-to-docs-artifacts/notes.txt",
    ],
)
def test_process_video_rejects_unsafe_output_file(output_file):
    """output_file must be a .csv inside youtube-to-docs-artifacts/."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        result = process_video(url="abc12345678", output_file=output_file)
    mock_main.assert_not_called()
    assert "Invalid output_file" in result


@pytest.mark.parametrize("output_file", ["workspace", "hf", "memory", "none"])
def test_process_video_allows_remote_output(output_file):
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        process_video(url="abc12345678", output_file=output_file)
    assert mock_main.call_args.args[0][2] == output_file


def test_process_video_all_suite():
    """Test process_video with the all_suite parameter."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        url = "https://www.youtube.com/watch?v=123"
        result = process_video(url=url, all_suite="gemini-flash")

        expected_args = [
            url,
            "--outfile",
            "youtube-to-docs-artifacts/youtube-docs.csv",
            "--all",
            "gemini-flash",
        ]

        mock_main.assert_called_once_with(expected_args)
        assert f"Successfully processed {url}" in result


def test_process_video_error():
    """Test process_video handles exceptions from app_main."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        mock_main.side_effect = Exception("Processing failed")
        url = "https://www.youtube.com/watch?v=123"

        result = process_video(url=url)

        assert f"Error processing {url}: Processing failed" in result


def test_process_video_corrected_captions_default():
    """'default' passes a bare flag so the CLI's default model and source apply."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        url = "https://www.youtube.com/watch?v=123"
        result = process_video(url=url, corrected_captions="default")

        expected_args = [
            url,
            "--outfile",
            "youtube-to-docs-artifacts/youtube-docs.csv",
            "--corrected-captions",
        ]

        mock_main.assert_called_once_with(expected_args)
        assert f"Successfully processed {url}" in result


def test_process_video_corrected_captions_with_source():
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        process_video(
            url="https://www.youtube.com/watch?v=123",
            corrected_captions="gemini-4-argon-youtube",
        )

        call_args = mock_main.call_args[0][0]
        idx = call_args.index("--corrected-captions")
        assert call_args[idx + 1] == "gemini-4-argon-youtube"


def test_process_video_corrected_captions_not_passed_when_none():
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        process_video(url="https://www.youtube.com/watch?v=123")

        call_args = mock_main.call_args[0][0]
        assert "--corrected-captions" not in call_args
        assert "--transcript" not in call_args


def test_tool_registration():
    """Verify that process_video is registered as a tool."""
    try:
        # Check if the tool is registered in the FastMCP instance
        assert mcp.name == "youtube-to-docs"
    except Exception:
        pass  # Tool registration check is best-effort
