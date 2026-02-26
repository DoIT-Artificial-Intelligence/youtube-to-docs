import os
from typing import Any, Optional, Tuple

from youtube_to_docs.constants import KNOWN_SRT_SOURCE_PREFIXES
from youtube_to_docs.llms import _query_llm

try:
    import boto3
except ImportError:
    boto3: Any = None

try:
    from google.cloud import translate_v2 as google_translate
except ImportError:
    google_translate: Any = None


def parse_translate_arg(translate_arg: str) -> Tuple[str, str]:
    """Parse a --translate argument in the format `{model}-{language}`.

    The language code is the last dash-separated component. Examples:
      "gemini-3-flash-preview-es"         -> ("gemini-3-flash-preview", "es")
      "bedrock-nova-2-lite-v1-fr"         -> ("bedrock-nova-2-lite-v1", "fr")
      "gemini-3-flash-preview-zh"         -> ("gemini-3-flash-preview", "zh")
      "aws-translate-es"                  -> ("aws-translate", "es")
      "gcp-translate-es"                  -> ("gcp-translate", "es")

    Returns (model_name, language_code).
    """
    parts = translate_arg.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --translate format: '{translate_arg}'. "
            "Expected '{model}-{language}' e.g. 'gemini-3-flash-preview-es', "
            "'aws-translate-es', or 'gcp-translate-es'."
        )
    return parts[0], parts[1]


def parse_suggest_captions_arg(arg: str) -> Tuple[str, Optional[str]]:
    """Parse a --suggest-corrected-captions argument in the format
    `{model}` or `{model}-{source}`.

    The source is either 'youtube' or a transcript model name
    (e.g. 'gcp-chirp3', 'gemini-3-flash-preview'). Parsing scans
    left-to-right for the first dash-boundary where the remaining suffix
    is 'youtube' or begins with a known source prefix.

    Examples:
      "gemini-3-flash-preview"               -> ("gemini-3-flash-preview", None)
      "gemini-3-flash-preview-youtube"       -> ("gemini-3-flash-preview", "youtube")
      "gemini-3-flash-preview-gcp-chirp3"    -> ("gemini-3-flash-preview", "gcp-chirp3")
      "bedrock-nova-2-lite-v1-youtube"       -> ("bedrock-nova-2-lite-v1", "youtube")

    Returns (model_name, source_or_none).
    """
    parts = arg.split("-")
    for i in range(1, len(parts)):
        candidate_source = "-".join(parts[i:])
        if candidate_source == "youtube" or any(
            candidate_source.startswith(pfx) for pfx in KNOWN_SRT_SOURCE_PREFIXES
        ):
            return "-".join(parts[:i]), candidate_source
    return arg, None


_AWS_TRANSLATE_BYTE_LIMIT = 10_000


def _chunk_text(text: str, max_bytes: int = _AWS_TRANSLATE_BYTE_LIMIT) -> list[str]:
    """Split text into chunks that each fit within max_bytes (UTF-8 encoded).

    Splits on blank lines first, then on single newlines, to keep logical
    blocks (e.g. SRT entries, paragraphs) together wherever possible.
    """
    chunks: list[str] = []
    current_lines: list[str] = []
    current_bytes = 0

    for line in text.splitlines(keepends=True):
        line_bytes = len(line.encode("utf-8"))
        if current_bytes + line_bytes > max_bytes and current_lines:
            chunks.append("".join(current_lines))
            current_lines = []
            current_bytes = 0
        current_lines.append(line)
        current_bytes += line_bytes

    if current_lines:
        chunks.append("".join(current_lines))

    return chunks


def _translate_aws(text: str, target_language: str) -> Tuple[str, int, int]:
    """Translate text using AWS Translate.

    Automatically splits input into chunks to respect the 10,000-byte per-request
    limit, then joins the translated chunks.

    Uses boto3 default credential chain (env vars, ~/.aws/credentials, IAM role, etc.).
    Returns (translated_text, 0, 0) — AWS Translate does not report token counts.
    """
    if boto3 is None:
        return (
            "Error: boto3 is required for AWS Translate. "
            "Install with `pip install boto3`",
            0,
            0,
        )

    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("translate", region_name=region)

    chunks = _chunk_text(text)
    translated_chunks: list[str] = []
    for chunk in chunks:
        response = client.translate_text(
            Text=chunk,
            SourceLanguageCode="en",
            TargetLanguageCode=target_language,
        )
        translated_chunks.append(response["TranslatedText"])

    return "".join(translated_chunks), 0, 0


_GCP_TRANSLATE_CHAR_LIMIT: int = 25_000


def _translate_gcp(text: str, target_language: str) -> Tuple[str, int, int]:
    """Translate text using Google Cloud Translation API (v2/Basic).

    Automatically splits input into chunks to respect the 30,000-character
    per-request limit, then joins the translated chunks.

    Uses Application Default Credentials (ADC): set GOOGLE_APPLICATION_CREDENTIALS
    to a service account key file, or authenticate via `gcloud auth application-default
    login`.
    Returns (translated_text, 0, 0) — Cloud Translation does not report token counts.
    """
    if google_translate is None:
        return (
            "Error: google-cloud-translate is required for GCP Translate. "
            "Install with `pip install google-cloud-translate`",
            0,
            0,
        )

    client = google_translate.Client()

    chunks = _chunk_text(text, max_bytes=_GCP_TRANSLATE_CHAR_LIMIT)
    translated_chunks: list[str] = []
    for chunk in chunks:
        result = client.translate(
            chunk,
            target_language=target_language,
            source_language="en",
            format_="text",
        )
        translated_chunks.append(result["translatedText"])

    return "".join(translated_chunks), 0, 0


def translate_text(
    model_name: str,
    text: str,
    target_language: str,
) -> Tuple[str, int, int]:
    """Translate text to target_language using model_name.

    If model_name is 'aws-translate', uses the AWS Translate service directly.
    If model_name is 'gcp-translate', uses Google Cloud Translation API directly.
    Otherwise, uses the specified LLM via _query_llm.

    Returns (translated_text, input_tokens, output_tokens).
    """
    if model_name == "aws-translate":
        return _translate_aws(text, target_language)

    if model_name == "gcp-translate":
        return _translate_gcp(text, target_language)

    prompt = (
        f"Please translate the following text to {target_language}. "
        "Return only the translated text without any preamble or explanation."
        "\n\n"
        f"{text}"
    )
    return _query_llm(model_name, prompt)


def process_translate(
    row: dict,
    translate_model: str,
    translate_lang: str,
    transcript_arg: str,
    model_names: list[str],
    summaries_dir: str,
    one_sentence_summaries_dir: str,
    qa_dir: str,
    tags_dir: str,
    video_id: str,
    safe_title: str,
    storage,
    verbose: bool = False,
) -> dict:
    """Translate English LLM outputs in row to translate_lang.

    Adds translated columns with suffix ` ({translate_lang})` and saves
    translated files alongside the originals.
    """
    from youtube_to_docs.utils import format_clickable_path

    lang_suffix = f" ({translate_lang})"
    lang_str = f" ({translate_lang})"

    def _translate_and_store(
        en_col: str,
        translated_col: str,
        save_dir: Optional[str],
        filename: str,
        file_col: Optional[str] = None,
    ) -> None:
        en_text = row.get(en_col)
        if not en_text or not isinstance(en_text, str):
            return
        if row.get(translated_col):
            return  # Already translated

        if verbose:
            print(f"Translating '{en_col}' to {translate_lang} using {translate_model}")

        translated, _, _ = translate_text(translate_model, en_text, translate_lang)
        row[translated_col] = translated

        if save_dir and translated and file_col:
            import os

            target_path = os.path.join(save_dir, filename)
            try:
                saved_path = storage.write_text(target_path, translated)
                from rich import print as rprint

                rprint(
                    f"Saved translated {translated_col}: "
                    f"{format_clickable_path(saved_path)}"
                )
                row[file_col] = saved_path
            except Exception as e:
                print(f"Error writing translated file {filename}: {e}")

    for model_name in model_names:
        # Summary
        _translate_and_store(
            en_col=f"Summary Text {model_name} from {transcript_arg}",
            translated_col=(
                f"Summary Text {model_name} from {transcript_arg}{lang_suffix}"
            ),
            save_dir=summaries_dir,
            filename=(
                f"{model_name} - {video_id} - {safe_title} - "
                f"summary (from {transcript_arg}){lang_str}.md"
            ),
            file_col=f"Summary File {model_name} from {transcript_arg}{lang_suffix}",
        )

        # One Sentence Summary
        _translate_and_store(
            en_col=f"One Sentence Summary {model_name} from {transcript_arg}",
            translated_col=(
                f"One Sentence Summary {model_name} from {transcript_arg}{lang_suffix}"
            ),
            save_dir=one_sentence_summaries_dir,
            filename=(
                f"{model_name} - {video_id} - {safe_title} - "
                f"one-sentence-summary (from {transcript_arg}){lang_str}.md"
            ),
            file_col=(
                f"One Sentence Summary File {model_name} from "
                f"{transcript_arg}{lang_suffix}"
            ),
        )

        # Q&A
        _translate_and_store(
            en_col=f"QA Text {model_name} from {transcript_arg}",
            translated_col=f"QA Text {model_name} from {transcript_arg}{lang_suffix}",
            save_dir=qa_dir,
            filename=(
                f"{model_name} - {video_id} - {safe_title} - "
                f"qa (from {transcript_arg}){lang_str}.md"
            ),
            file_col=f"QA File {model_name} from {transcript_arg}{lang_suffix}",
        )

        # Tags
        _translate_and_store(
            en_col=f"Tags {transcript_arg} {model_name} model",
            translated_col=f"Tags {transcript_arg} {model_name} model{lang_suffix}",
            save_dir=tags_dir,
            filename=(
                f"{model_name} - {video_id} - {safe_title} - "
                f"tags (from {transcript_arg}){lang_str}.txt"
            ),
            file_col=f"Tags File {transcript_arg} {model_name} model{lang_suffix}",
        )

        # Secondary (from youtube) summaries
        if f"Summary Text {model_name} from youtube" in row:
            _translate_and_store(
                en_col=f"Summary Text {model_name} from youtube",
                translated_col=f"Summary Text {model_name} from youtube{lang_suffix}",
                save_dir=summaries_dir,
                filename=(
                    f"{model_name} - {video_id} - {safe_title} - "
                    f"summary (from youtube){lang_str}.md"
                ),
                file_col=f"Summary File {model_name} from youtube{lang_suffix}",
            )

        if f"One Sentence Summary {model_name} from youtube" in row:
            _translate_and_store(
                en_col=f"One Sentence Summary {model_name} from youtube",
                translated_col=(
                    f"One Sentence Summary {model_name} from youtube{lang_suffix}"
                ),
                save_dir=one_sentence_summaries_dir,
                filename=(
                    f"{model_name} - {video_id} - {safe_title} - "
                    f"one-sentence-summary (from youtube){lang_str}.md"
                ),
                file_col=(
                    f"One Sentence Summary File {model_name} from youtube{lang_suffix}"
                ),
            )

        if f"QA Text {model_name} from youtube" in row:
            _translate_and_store(
                en_col=f"QA Text {model_name} from youtube",
                translated_col=f"QA Text {model_name} from youtube{lang_suffix}",
                save_dir=qa_dir,
                filename=(
                    f"{model_name} - {video_id} - {safe_title} - "
                    f"qa (from youtube){lang_str}.md"
                ),
                file_col=f"QA File {model_name} from youtube{lang_suffix}",
            )

    return row
