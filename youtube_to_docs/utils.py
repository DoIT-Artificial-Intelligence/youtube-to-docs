import importlib.metadata
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import polars as pl
from rich import print as rprint

T = TypeVar("T")

# Local output written by untrusted callers (web app, MCP server) must stay
# inside this directory, relative to the working directory.
OUTPUT_ROOT = "youtube-to-docs-artifacts"

# Remote/special output_file values that are not local paths
REMOTE_OUTPUT_VALUES = {
    "workspace",
    "w",
    "sharepoint",
    "s",
    "hf",
    "memory",
    "m",
    "none",
    "n",
}


def is_remote_output(output_file: str) -> bool:
    """Return True if main() treats output_file as non-local storage."""
    if output_file.lower() in REMOTE_OUTPUT_VALUES:
        return True
    # Mirrors the Google Drive folder ID heuristic in main()
    return (
        len(output_file) > 20
        and "." not in output_file
        and "/" not in output_file
        and "\\" not in output_file
    )


def resolve_within(base: str, user_path: str) -> str:
    """Resolve user_path against base (following symlinks) and ensure the
    result is base itself or inside it. Raises ValueError otherwise.
    """
    real_base = os.path.realpath(base)
    resolved = os.path.realpath(os.path.join(real_base, user_path))
    if not (resolved == real_base or resolved.startswith(real_base + os.sep)):
        raise ValueError(f"Invalid path: must be within {base}")
    return resolved


def validate_output_file(output_file: str) -> str:
    """Validate an output_file supplied by an untrusted caller.

    Remote/special values pass through unchanged. Local paths are resolved
    relative to the working directory and must be a ``.csv`` file inside
    OUTPUT_ROOT, so generated output can never overwrite source, config, or
    credential files. Returns the path as ``OUTPUT_ROOT/<relative path>``.
    Raises ValueError for invalid paths.
    """
    if is_remote_output(output_file):
        return output_file

    cwd = os.path.realpath(os.getcwd())
    resolved = os.path.realpath(os.path.join(cwd, output_file))
    root = os.path.join(cwd, OUTPUT_ROOT)
    real_root = os.path.realpath(root)
    try:
        resolve_within(root, resolved)
    except ValueError:
        raise ValueError(
            f"Invalid output_file: must be inside '{OUTPUT_ROOT}/' "
            f"(e.g. '{OUTPUT_ROOT}/youtube-docs.csv')"
        ) from None
    if resolved == real_root or not resolved.lower().endswith(".csv"):
        raise ValueError("Invalid output_file: must be a .csv file")
    return os.path.join(OUTPUT_ROOT, os.path.relpath(resolved, real_root)).replace(
        "\\", "/"
    )


def optional_value_args(flag: str, value: str) -> list[str]:
    """CLI args for an option whose value is optional (e.g. `--corrected-captions`)
    from the MCP server or web app. 'default' passes the bare flag, so the CLI's
    default applies."""
    if value.strip().lower() in ("default", "true", "yes"):
        return [flag]
    return [flag, value.strip()]


def is_error_response(text: object) -> bool:
    """Return True if an LLM/API output string represents an error message."""
    return isinstance(text, str) and text.strip().startswith("Error:")


def build_cli_args(
    url: str,
    output_file: str = f"{OUTPUT_ROOT}/youtube-docs.csv",
    transcript_source: str | None = "youtube",
    model: str | None = None,
    tts_model: str | None = None,
    infographic_model: str | None = None,
    alt_text_model: str | None = None,
    no_youtube_summary: bool = False,
    from_video: str | None = None,
    combine_infographic_audio: bool = False,
    all_suite: str | None = None,
    translate: str | None = None,
    corrected_captions: str | None = None,
    post_process: str | None = None,
    hugging_face_dataset: str | None = None,
    verbose: bool = False,
) -> list[str]:
    """Build the CLI argument list passed to ``youtube_to_docs.main.main``."""
    args = [url, "--outfile", output_file]

    if transcript_source:
        args.extend(["--transcript", transcript_source])
    if hugging_face_dataset:
        args.extend(["--hugging-face-dataset", hugging_face_dataset])
    if translate:
        args.extend(["--translate", translate])
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
    if from_video:
        args.extend(["--from-video", from_video])
    if combine_infographic_audio:
        args.append("--combine-infographic-audio")
    if all_suite:
        args.extend(["--all", all_suite])
    if corrected_captions:
        args.extend(optional_value_args("--corrected-captions", corrected_captions))
    if post_process:
        args.extend(["--post-process", post_process])
    if verbose:
        args.append("--verbose")

    return args


def get_version() -> str:
    """Return the installed package version, or "unknown" if not installed."""
    try:
        return importlib.metadata.version("youtube-to-docs")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_gcp_client(
    callable_item: Callable[..., T], service_name: str, **kwargs
) -> T | None:
    """
    Safely initializes a GCP client or calls a GCP auth function.
    Handles DefaultCredentialsError and provides helpful instructions to the user.
    """
    try:
        return callable_item(**kwargs)
    except Exception as e:
        # Use isinstance for better type checking if google-auth is available
        creds_error = False
        try:
            from google.auth import exceptions as google_auth_exceptions

            if isinstance(e, google_auth_exceptions.DefaultCredentialsError):
                creds_error = True
        except ImportError:
            pass

        if not creds_error and "DefaultCredentialsError" in str(e):
            creds_error = True

        if creds_error:
            rprint(
                f"\n[bold red]Error: Google Cloud Application Default Credentials "
                f"not found for {service_name}.[/bold red]"
            )
            rprint(
                "[yellow]To fix this, please run the following command in your "
                "terminal:[/yellow]\n"
            )
            rprint(
                "    [bold white]gcloud auth application-default login[/bold white]\n"
            )
            rprint(
                "[yellow]Alternatively, set the "
                "[bold white]GOOGLE_APPLICATION_CREDENTIALS[/bold white] "
                "environment variable to the path of your service account JSON key "
                "file.[/yellow]\n"
            )
            return None
        else:
            rprint(
                f"[bold red]Error initializing {service_name} client: {e}[/bold red]"
            )
            return None


def format_clickable_path(path: str) -> str:
    """
    Formats a path or URL as a clickable link for Rich.
    If it's a local file path, it uses the file:// URI scheme.
    """
    if not path or not isinstance(path, str):
        return str(path)

    if path.startswith("http"):
        return f"[link={path}]{path}[/link]"

    # Assume it's a local path if it doesn't start with http
    try:
        abs_path = os.path.abspath(path)
        uri = Path(abs_path).as_uri()
        return f"[link={uri}]{path}[/link]"
    except (ValueError, OSError):
        return path


def reorder_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Reorder columns according to the specified logical structure."""
    cols = df.columns
    final_order = []

    # 1. Video metadata
    metadata = [
        "Title",
        "URL",
        "Description",
        "Data Published",
        "Channel",
        "Tags",
        "Duration",
    ]
    final_order.extend([c for c in metadata if c in cols])

    # 2. Audio extraction
    if "Audio File" in cols:
        final_order.append("Audio File")

    # 3. Transcript extraction
    # Youtube chars/files
    # English fallback / specific language columns might exist
    # Pattern: Transcript characters ..., Transcript File ...
    transcript_cols = [
        c
        for c in cols
        if c.startswith("Transcript characters from ")
        or c.startswith("Transcript File ")
    ]
    final_order.extend(sorted(transcript_cols))

    # 4. Summary (Summary Text, Summary File, One Sentence Summary)
    summary_cols = [
        c
        for c in cols
        if (
            c.startswith("Summary Text ")
            or c.startswith("Summary File ")
            or c.startswith("One Sentence Summary ")
        )
        and "Infographic" not in c
        and "Audio" not in c
    ]
    final_order.extend(sorted(summary_cols))

    # 5. Speaker extraction
    speaker_cols = [c for c in cols if c.startswith("Speakers ") and "cost" not in c]
    final_order.extend(sorted(speaker_cols))

    # 5.5 AI Tags
    ai_tags_cols = [c for c in cols if c.startswith("Tags ") and c.endswith(" model")]
    final_order.extend(sorted(ai_tags_cols))

    # 6. Questions and answers
    qa_cols = [
        c for c in cols if (c.startswith("QA Text ") or c.startswith("QA File "))
    ]
    final_order.extend(sorted(qa_cols))

    # 7. Infographic
    infographic_cols = [
        c
        for c in cols
        if c.startswith("Summary Infographic File ")
        or c.startswith("Summary Infographic Alt Text ")
    ]
    final_order.extend(sorted(infographic_cols))

    # 8. Audio creation
    audio_video_cols = [
        c for c in cols if c.startswith("Summary Audio File ") or c == "Video File"
    ]
    final_order.extend(sorted(audio_video_cols))

    # 9. Costs
    cost_cols = [c for c in cols if " cost " in c or c.endswith(" cost")]
    # Include STT cost which is usually "STT cost" not " STT cost "
    stt_costs = [c for c in cols if "STT cost" in c and c not in cost_cols]

    all_costs = sorted(cost_cols + stt_costs)
    final_order.extend(all_costs)

    # Add any remaining columns that weren't caught
    remaining = [c for c in cols if c not in final_order]
    final_order.extend(sorted(remaining))

    return df.select(final_order)


def normalize_model_name(model_name: str) -> str:
    """
    Normalizes a model name by stripping prefixes and suffixes.
    Suffixes handled: @20251001, -20251001-v1, -v1.
    Prefixes handled: vertex-, bedrock-, foundry-.
    """
    # Strip prefixes
    normalized = model_name
    prefixes = ["vertex-", "bedrock-", "foundry-"]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    # Strip suffixes using regex: (@\d{8}|-\d{8}-v\d+|-v\d+)$
    normalized = re.sub(r"(@\d{8}|-\d{8}-v\d+|-v\d+)$", "", normalized)

    return normalized


def add_question_numbers(markdown_table: str) -> str:
    """
    Adds a 'question number' column to the markdown table.
    """
    lines = markdown_table.strip().split("\n")
    if not lines:
        return markdown_table

    # Find the header row and separator row
    header_idx = -1
    for i in range(len(lines) - 1):
        line = lines[i].strip()
        next_line = lines[i + 1].strip()
        if "|" in line and ("---" in next_line or "-|-" in next_line):
            header_idx = i
            break

    if header_idx == -1:
        # Fallback: if we can't find a clear table header+separator, return as is
        return markdown_table

    new_lines = []
    # Add lines before the table as is
    for i in range(header_idx):
        new_lines.append(lines[i])

    # Header row
    header = lines[header_idx].strip()
    if not header.startswith("|"):
        header = "|" + header
    new_lines.append(f"| question number {header}")

    # Separator row
    separator = lines[header_idx + 1].strip()
    if not separator.startswith("|"):
        separator = "|" + separator
    new_lines.append(f"|---{separator}")

    # Data rows
    question_counter = 1
    for i in range(header_idx + 2, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if line.startswith("|"):
            new_lines.append(f"| {question_counter} {line}")
            question_counter += 1
        elif "|" in line:
            new_lines.append(f"| {question_counter} | {line}")
            question_counter += 1
        else:
            new_lines.append(lines[i])

    return "\n".join(new_lines)
