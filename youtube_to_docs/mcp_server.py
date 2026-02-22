import contextlib
import io

from mcp.server.fastmcp import FastMCP

from youtube_to_docs.main import main as app_main

mcp = FastMCP("youtube-to-docs")


@mcp.tool()
def process_video(
    url: str,
    output_file: str = "youtube-to-docs-artifacts/youtube-docs.csv",
    transcript_source: str = "youtube",
    model: str | None = None,
    tts_model: str | None = None,
    infographic_model: str | None = None,
    alt_text_model: str | None = None,
    no_youtube_summary: bool = False,
    languages: str = "en",
    combine_infographic_audio: bool = False,
    all_suite: str | None = None,
    verbose: bool = False,
) -> str:
    """
    Process a YouTube video to generate transcripts, summaries, Q&A, and infographics.

    Note: Advanced features (video generation, AI transcription, cloud storage)
    require optional dependencies to be available in the environment. The MCP server
    is pre-configured to handle this via 'uv run --all-extras'.

    Args:
        url: The YouTube URL or video ID. Can also be a playlist ID
            or comma-separated list of IDs.
        output_file: Path to save the output CSV file.
            Defaults to 'youtube-to-docs-artifacts/youtube-docs.csv'.
            Can also be 'workspace' or 'w' to store to Google Drive,
            'sharepoint' or 's' to store to Microsoft SharePoint,
            or 'none' or 'n' to skip saving to a file (results will be in the log).
        transcript_source: The transcript source to use. 'youtube' (default)
            fetches existing transcripts. Provide an AI model name
            (e.g., 'gemini-3-flash-preview') to perform STT on extracted audio.
        model: The LLM model to use for speaker extraction, Q&A, and summarization
            (e.g., 'gemini-3-flash-preview'). Can be a comma-separated list.
        tts_model: The TTS model and voice to use
            (e.g., 'gemini-2.5-flash-preview-tts-Kore', 'gcp-chirp3-Kore',
            'aws-polly-Ruth').
        infographic_model: The image model to use for generating an infographic
            (e.g., 'gemini-2.5-flash-image').
        alt_text_model: The LLM model to use for generating alt text for the
            infographic. Defaults to the summary model.
        no_youtube_summary: If True, skips generating a secondary summary from the
            YouTube transcript when using an AI model for the primary transcript.
        languages: The target language(s) (e.g., 'es', 'fr', 'en'). Defaults to 'en'.
            Format: '{model}-{language}' (e.g., 'gemini-3.1-pro-preview-es' or
            'aws-translate-fr'). Can be a comma-separated list.
        combine_infographic_audio: If True, combines the infographic and audio summary
            into a video file. Requires both tts_model and infographic_model.
        all_suite: Shortcut to use a specific model suite for everything.
            e.g., 'gemini-flash', 'gemini-pro', 'gemini-flash-pro-image', or 'gcp-pro'.
        verbose: If True, enables verbose logging in the output.
    """
    args = [
        url,
        "--outfile",
        output_file,
        "--transcript",
        transcript_source,
        "--language",
        languages,
    ]

    if model:
        args.extend(["--model", model])

    if tts_model:
        args.extend(["--tts", tts_model])

    if infographic_model:
        args.extend(["--infographic", infographic_model])

    if alt_text_model:
        args.extend(["--alt-text-model", alt_text_model])

    if no_youtube_summary:
        args.append("--no-youtube-summary")

    if combine_infographic_audio:
        args.append("--combine-infographic-audio")

    if all_suite:
        args.extend(["--all", all_suite])

    if verbose:
        args.append("--verbose")

    # Capture stdout/stderr to return as tool output
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            app_main(args)
            output = f.getvalue()
            return f"Successfully processed {url}.\n\nOutput Log:\n{output}"
        except Exception as e:
            output = f.getvalue()
            return f"Error processing {url}: {str(e)}\n\nOutput Log:\n{output}"


if __name__ == "__main__":
    mcp.run()
