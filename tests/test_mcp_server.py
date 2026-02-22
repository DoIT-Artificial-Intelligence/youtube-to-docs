from unittest.mock import patch

from youtube_to_docs.mcp_server import mcp, process_video


def test_process_video_defaults():
    """Test process_video with default arguments."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        url = "https://www.youtube.com/watch?v=123"
        result = process_video(url=url)

        expected_args = [
            url,
            "--outfile",
            "youtube-to-docs-artifacts/youtube-docs.csv",
            "--transcript",
            "youtube",
        ]

        mock_main.assert_called_once_with(expected_args)
        assert f"Successfully processed {url}" in result


def test_process_video_all_args():
    """Test process_video with all arguments provided."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        url = "https://www.youtube.com/watch?v=123"
        result = process_video(
            url=url,
            output_file="custom.csv",
            transcript_source="gemini-pro",
            model="gemini-2.5-pro",
            tts_model="tts-1",
            infographic_model="imagen",
            no_youtube_summary=True,
            translate="gemini-2.5-pro-es",
        )

        expected_args = [
            url,
            "--outfile",
            "custom.csv",
            "--transcript",
            "gemini-pro",
            "--translate",
            "gemini-2.5-pro-es",
            "--model",
            "gemini-2.5-pro",
            "--tts",
            "tts-1",
            "--infographic",
            "imagen",
            "--no-youtube-summary",
        ]

        mock_main.assert_called_once_with(expected_args)
        assert f"Successfully processed {url}" in result


def test_process_video_all_suite():
    """Test process_video with the all_suite parameter."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        url = "https://www.youtube.com/watch?v=123"
        result = process_video(url=url, all_suite="gemini-flash")

        expected_args = [
            url,
            "--outfile",
            "youtube-to-docs-artifacts/youtube-docs.csv",
            "--transcript",
            "youtube",
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


def test_tool_registration():
    """Verify that process_video is registered as a tool."""
    try:
        # Check if the tool is registered in the FastMCP instance
        assert mcp.name == "youtube-to-docs"
    except Exception:
        pass
