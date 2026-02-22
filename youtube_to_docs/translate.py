from typing import Optional, Tuple

from youtube_to_docs.llms import _query_llm


def parse_translate_arg(translate_arg: str) -> Tuple[str, str]:
    """Parse a --translate argument in the format `{model}-{language}`.

    The language code is the last dash-separated component. Examples:
      "gemini-3-flash-preview-es"         -> ("gemini-3-flash-preview", "es")
      "bedrock-nova-2-lite-v1-fr"         -> ("bedrock-nova-2-lite-v1", "fr")
      "gemini-3-flash-preview-zh"         -> ("gemini-3-flash-preview", "zh")

    Returns (model_name, language_code).
    """
    parts = translate_arg.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --translate format: '{translate_arg}'. "
            "Expected '{model}-{language}' e.g. 'gemini-3-flash-preview-es'."
        )
    return parts[0], parts[1]


def translate_text(
    model_name: str,
    text: str,
    target_language: str,
) -> Tuple[str, int, int]:
    """Translate text to target_language using model_name.

    Returns (translated_text, input_tokens, output_tokens).
    """
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
