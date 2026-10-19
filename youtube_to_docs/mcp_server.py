import contextlib
import io

from mcp.server.mcpserver import MCPServer

from youtube_to_docs.main import main as app_main
from youtube_to_docs.utils import (
    build_cli_args,
    get_version,
    validate_output_file,
)

mcp = MCPServer("youtube-to-docs", version=get_version())


@mcp.tool()
def process_video(
    url: str,
    output_file: str = "youtube-to-docs-artifacts/youtube-docs.csv",
    hugging_face_dataset: str | None = None,
    transcript_source: str | None = None,
    model: str | None = None,
    tts_model: str | None = None,
    infographic_model: str | None = None,
    alt_text_model: str | None = None,
    no_youtube_summary: bool = False,
    from_video: str | None = None,
    translate: str | None = None,
    combine_infographic_audio: bool = False,
    all_suite: str | None = None,
    corrected_captions: str | None = None,
    post_process: str | None = None,
    verbose: bool = False,
) -> str:
    """
    Process a YouTube video to generate transcripts, summaries, Q&A, and infographics.

    Note: Advanced features (video generation, AI transcription, cloud storage)
    require optional dependencies to be available in the environment. The MCP server
    is pre-configured to handle this via 'uv run --with youtube-to-docs[all]'.

    Args:
        url: The YouTube URL or video ID. Can also be a playlist ID
            or comma-separated list of IDs.
        output_file: Path to save the output CSV file.
            Defaults to 'youtube-to-docs-artifacts/youtube-docs.csv'.
            Local paths must be a .csv file inside 'youtube-to-docs-artifacts/'.
            Can also be 'workspace' or 'w' to store to Google Drive,
            'sharepoint' or 's' to store to Microsoft SharePoint,
            'hf' to store to a Hugging Face dataset (see hugging_face_dataset),
            'memory' or 'm' to keep artifacts in memory (no files on disk),
            or 'none' or 'n' to skip saving to a file (results will be in the log).
        hugging_face_dataset: The Hugging Face dataset repository to store artifacts
            in when output_file is 'hf'. Can be a full repo id (e.g.
            'username/my-dataset') or a plain name (e.g.
            'Code for America Summit 2026 Recap') which is slugified and namespaced
            under the authenticated user. If omitted and the input is a playlist,
            the playlist's title is used as the dataset name. Requires the HF_TOKEN
            environment variable.
        transcript_source: The transcript source to use. 'youtube' (default)
            fetches existing transcripts. Provide an AI model name
            (e.g., 'gemini-3.5-transcribe', 'gemini-3.5-flash-lite', 'gcp-chirp3')
            to perform STT on extracted audio. 'gemini-3.5-transcribe' is Google's
            dedicated STT model and returns word-level timestamps plus speaker
            diarization, so its SRT has real offsets and '[Speaker N]' labels.
            When corrected_captions corrects an STT transcript, that model is the
            default instead.
        model: The LLM model to use for speaker extraction, Q&A, and summarization
            (e.g., 'gemini-3.5-flash-lite'). Can be a comma-separated list.
            Defaults to None (Transcript only).
            Use 'gemini-3.5-flash-lite' for summarization.
        tts_model: The TTS model and voice to use
            (e.g., 'gemini-3.8-flash-lite-tts-Kore', 'gcp-chirp3-Kore',
            'aws-polly-Ruth').
        infographic_model: The image model to use for generating an infographic
            (e.g., 'gemini-nano-banana-2.1', 'gemini-3-pro-image' or
            'foundry-gpt-image-2.5-sunburst'). Can be a
            comma-separated list to generate one infographic per model.
        alt_text_model: The LLM model to use for generating alt text for the
            infographic. Defaults to the summary model.
        no_youtube_summary: If True, skips generating a secondary summary from the
            YouTube transcript when using an AI model for the primary transcript.
        from_video: Also generate the summary, one sentence summary, speakers and
            Q&A directly from the video with Gemini video understanding. Value is
            the processing mode: 'agentic' (the model navigates the video; Gemini
            3.5 Flash-Lite and later, e.g. gemini-3.8-flash, gemini-4-argon), 'static'
            (frames at 1 FPS; any Gemini model), or 'agentic,static' for both.
        translate: Translate all outputs to a target language after generating in
            English. Format: `{model}-{language}` e.g. `gemini-3.5-flash-lite-es`,
            `bedrock-nova-2-lite-v1-fr`, `aws-translate-es`, or `gcp-translate-es`.
            Use `aws-translate` to use the AWS Translate service, or `gcp-translate`
            to use Google Cloud Translation API, instead of an LLM. When set,
            summaries, Q&A, tags, transcripts, and SRT files are all translated. TTS
            and infographics are generated in both English and the target language.
        combine_infographic_audio: If True, combines the infographic and audio summary
            into a video file. Requires both tts_model and infographic_model.
        all_suite: Shortcut to use a specific model suite for everything.
            e.g., 'gemini-flash', 'gemini-pro', 'gemini-flash-pro-image', 'gcp-pro',
            or 'anthropic-opus'.
        corrected_captions: Make the video accessible (WCAG 2.1 Level AA and
            Section 508): corrected captions (SRT in 'corrected-caption-files/',
            WebVTT in 'vtt-files/') with real speaker names on every change and
            corrected words, sounds and timing; audio description of the slides
            and other visual information (WebVTT descriptions tracks in
            'vtt-files/' and a descriptive transcript in
            'descriptive-transcript-files/'); and in 'video-files/' the original
            video plus a copy with the captions burned in and the descriptions
            narrated (tts_model, default 'gemini-3.8-flash-tts-Kore'). Also
            writes an Able Player page in 'vtt-files/' for checking the WebVTT
            tracks. Use 'default' for the best results (gemini-4-argon on a
            gemini-3.5-transcribe transcript, transcribed in the same run), or
            `{model}` / `{model}-{source}` where source is 'youtube' or a
            transcript model name (e.g. 'gemini-4-argon-youtube'). Takes about
            20 minutes for a 3-hour video.
        post_process: Post-process the transcript with JSON operations.
            Example: '{"word count": "apple"}' counts occurrences of 'apple'.
            Values can be a list for multiple words.
        verbose: If True, enables verbose logging in the output.
    """
    try:
        output_file = validate_output_file(output_file)
    except ValueError as e:
        return f"Error processing {url}: {e}"

    args = build_cli_args(
        url=url,
        output_file=output_file,
        hugging_face_dataset=hugging_face_dataset,
        transcript_source=transcript_source,
        model=model,
        tts_model=tts_model,
        infographic_model=infographic_model,
        alt_text_model=alt_text_model,
        no_youtube_summary=no_youtube_summary,
        from_video=from_video,
        translate=translate,
        combine_infographic_audio=combine_infographic_audio,
        all_suite=all_suite,
        corrected_captions=corrected_captions,
        post_process=post_process,
        verbose=verbose,
    )

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
