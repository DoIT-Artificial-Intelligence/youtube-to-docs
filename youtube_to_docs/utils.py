import os
import re
from pathlib import Path
from typing import Any, Type, TypeVar

import polars as pl
from rich import print as rprint

T = TypeVar("T")


def get_gcp_client(
    callable_item: Type[T] | Any, service_name: str, **kwargs
) -> T | Any | None:
    """
    Safely initializes a GCP client or calls a GCP auth function.
    Handles DefaultCredentialsError and provides helpful instructions to the user.
    """
    try:
        return callable_item(**kwargs)
    except Exception as e:
        # We check for 'DefaultCredentialsError' in the string representation
        # of the error to avoid direct dependency on google-auth if not present.
        if "DefaultCredentialsError" in str(e):
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
