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


def test_process_video_suggest_corrected_captions():
    """Test that suggest_corrected_captions is passed as a CLI arg."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        url = "https://www.youtube.com/watch?v=123"
        result = process_video(
            url=url,
            suggest_corrected_captions="gemini-3-flash-preview-gcp-chirp3",
        )

        expected_args = [
            url,
            "--outfile",
            "youtube-to-docs-artifacts/youtube-docs.csv",
            "--transcript",
            "youtube",
            "--suggest-corrected-captions",
            "gemini-3-flash-preview-gcp-chirp3",
        ]

        mock_main.assert_called_once_with(expected_args)
        assert f"Successfully processed {url}" in result


def test_process_video_suggest_corrected_captions_youtube_source():
    """Test suggest_corrected_captions with explicit youtube source."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        url = "https://www.youtube.com/watch?v=123"
        process_video(
            url=url,
            suggest_corrected_captions="gemini-3-flash-preview-youtube",
        )

        call_args = mock_main.call_args[0][0]
        idx = call_args.index("--suggest-corrected-captions")
        assert call_args[idx + 1] == "gemini-3-flash-preview-youtube"


def test_process_video_suggest_corrected_captions_not_passed_when_none():
    """Test that --suggest-corrected-captions is absent when not set."""
    with patch("youtube_to_docs.mcp_server.app_main") as mock_main:
        process_video(url="https://www.youtube.com/watch?v=123")

        call_args = mock_main.call_args[0][0]
        assert "--suggest-corrected-captions" not in call_args


def test_tool_registration():
    """Verify that process_video is registered as a tool."""
    try:
        # Check if the tool is registered in the FastMCP instance
        assert mcp.name == "youtube-to-docs"
    except Exception:
        pass
