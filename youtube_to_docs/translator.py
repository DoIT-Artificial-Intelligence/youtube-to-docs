"""
Language translation module.

Handles parsing of {model}-{language} format and dispatching translation
to the appropriate backend (LLM or cloud service).

All text generation happens in English first. Translation is always
a post-processing step.

Examples:
    gemini-3-flash-preview-es -> LLM translation via gemini-3-flash-preview
    aws-translate-fr -> AWS Translate API
    bedrock-claude-haiku-4-5-20251001-v1-de -> LLM translation via Bedrock Claude
"""

from typing import Any, Tuple

try:
    import boto3
except ImportError:
    boto3: Any = None

# Known cloud translation service prefixes
CLOUD_SERVICES = {"aws-translate"}

# Known language codes (ISO 639-1) used to identify the language suffix
LANGUAGE_CODES = {
    "af",
    "am",
    "ar",
    "az",
    "be",
    "bg",
    "bn",
    "bs",
    "ca",
    "ceb",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "ny",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "rw",
    "sd",
    "si",
    "sk",
    "sl",
    "sm",
    "sn",
    "so",
    "sq",
    "sr",
    "st",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "ug",
    "uk",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
}


def parse_language_arg(lang_str: str) -> Tuple[str, str]:
    """
    Parses a language argument into (translation_model, language_code).

    Format: {model}-{language_code}

    Examples:
        gemini-3-flash-preview-es -> ("gemini-3-flash-preview", "es")
        aws-translate-fr -> ("aws-translate", "fr")
        bedrock-claude-haiku-4-5-20251001-v1-de -> (
            "bedrock-claude-haiku-4-5-20251001-v1", "de"
        )

    The language code is always the last hyphen-delimited segment, validated
    against known ISO 639-1 codes.
    """
    if not lang_str or lang_str.strip() == "":
        raise ValueError("Language argument cannot be empty")

    lang_str = lang_str.strip()

    # Check for cloud service prefixes first
    for service in CLOUD_SERVICES:
        if lang_str.startswith(service):
            # Format: aws-translate-{lang}
            suffix = lang_str[len(service) :]
            if suffix.startswith("-") and suffix[1:] in LANGUAGE_CODES:
                return service, suffix[1:]
            raise ValueError(
                f"Invalid language code in '{lang_str}'. "
                f"Expected format: {service}-{{language_code}}"
            )

    # For LLM models: the last segment after the final hyphen is the language
    if "-" in lang_str:
        parts = lang_str.rsplit("-", 1)
        model, lang_code = parts[0], parts[1]
        if lang_code in LANGUAGE_CODES:
            return model, lang_code
        raise ValueError(
            f"'{lang_code}' is not a recognized language code in '{lang_str}'. "
            f"Expected format: {{model}}-{{language_code}}"
        )

    # Support backward compatibility for plain language codes (e.g., "es")
    # Default to "gemini" as the translation model
    if lang_str in LANGUAGE_CODES:
        return "gemini", lang_str

    raise ValueError(
        f"Invalid language argument '{lang_str}'. "
        f"Expected format: {{model}}-{{language_code}} or {{language_code}} "
        f"(e.g., gemini-3-flash-preview-es, aws-translate-fr, es)"
    )


def is_cloud_translation(model: str) -> bool:
    """Returns True if the translation model is a cloud service."""
    return model in CLOUD_SERVICES


def translate_text(
    text: str,
    model: str,
    target_language: str,
    source_language: str = "en",
) -> Tuple[str, int, int]:
    """
    Translates text using the specified model or service.

    Returns (translated_text, input_tokens, output_tokens).
    For cloud services, tokens are reported as 0.

    Args:
        text: The text to translate.
        model: The translation model/service (e.g., "gemini-3-flash-preview",
               "aws-translate").
        target_language: Target language code (e.g., "es", "fr").
        source_language: Source language code. Defaults to "en".
    """
    if not text or not text.strip():
        return text, 0, 0

    if target_language == source_language:
        return text, 0, 0

    if is_cloud_translation(model):
        translated = _translate_aws(text, target_language, source_language)
        return translated, 0, 0
    else:
        return _translate_llm(text, model, target_language, source_language)


def _translate_llm(
    text: str,
    model_name: str,
    target_language: str,
    source_language: str = "en",
) -> Tuple[str, int, int]:
    """
    Translates text using an LLM with a translation prompt.

    Uses the same _query_llm infrastructure as other LLM calls.
    Returns (translated_text, input_tokens, output_tokens).
    """
    from youtube_to_docs.llms import _query_llm

    if model_name == "gemini":
        model_name = "gemini-3-flash-preview"

    prompt = (
        f"Translate the following text from {source_language} to "
        f"{target_language}. "
        "Preserve the original formatting (markdown, tables, etc). "
        "Return ONLY the translated text without any introductory or "
        "concluding commentary.\n\n"
        f"{text}"
    )
    return _query_llm(model_name, prompt)


def _translate_aws(
    text: str,
    target_language: str,
    source_language: str = "en",
) -> str:
    """
    Translates text using AWS Translate.

    Handles chunking for texts exceeding the 10,000 byte API limit.
    Uses the same boto3 dependency as other AWS integrations.
    """
    if boto3 is None:
        raise ImportError(
            "boto3 is required for AWS Translate. Install with: pip install boto3"
        )

    client = boto3.client("translate", region_name="us-east-1")

    # AWS Translate has a 10,000 byte limit per request
    max_bytes = 9500  # Leave some margin
    text_bytes = text.encode("utf-8")

    if len(text_bytes) <= max_bytes:
        response = client.translate_text(
            Text=text,
            SourceLanguageCode=source_language,
            TargetLanguageCode=target_language,
        )
        return response["TranslatedText"]

    # Chunk by paragraphs first, then by lines if needed
    chunks = _chunk_text_for_translation(text, max_bytes)
    translated_chunks = []

    for chunk in chunks:
        response = client.translate_text(
            Text=chunk,
            SourceLanguageCode=source_language,
            TargetLanguageCode=target_language,
        )
        translated_chunks.append(response["TranslatedText"])

    return "\n\n".join(translated_chunks)


def _chunk_text_for_translation(text: str, max_bytes: int) -> list[str]:
    """
    Splits text into chunks that fit within the byte limit.

    Tries to split on paragraph boundaries first, then line boundaries.
    """
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0

    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para_bytes = len(para.encode("utf-8"))

        if para_bytes > max_bytes:
            # Paragraph itself is too large, split by lines
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            lines = para.split("\n")
            line_chunk: list[str] = []
            line_size = 0

            for line in lines:
                line_bytes = len(line.encode("utf-8"))
                if line_size + line_bytes + 1 > max_bytes and line_chunk:
                    chunks.append("\n".join(line_chunk))
                    line_chunk = []
                    line_size = 0
                line_chunk.append(line)
                line_size += line_bytes + 1

            if line_chunk:
                chunks.append("\n".join(line_chunk))
        elif current_size + para_bytes + 2 > max_bytes:
            # Current chunk would exceed limit, start new chunk
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_bytes
        else:
            current_chunk.append(para)
            current_size += para_bytes + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
