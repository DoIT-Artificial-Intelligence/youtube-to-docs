import os
import re
from pathlib import Path

import polars as pl


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
    infographic_cols = [c for c in cols if c.startswith("Summary Infographic File ")]
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

    # Check if it's a valid table (has header and separator)
    if len(lines) < 2:
        return markdown_table

    new_lines = []
    question_counter = 1

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if i == 0:
            # Header row
            # Ensure it starts with | (some LLMs might miss it)
            if not stripped_line.startswith("|"):
                stripped_line = "|" + stripped_line
            new_lines.append(f"| question number {stripped_line}")
        elif i == 1 and ("---" in stripped_line or "-|-" in stripped_line):
            # Separator row
            if not stripped_line.startswith("|"):
                stripped_line = "|" + stripped_line
            new_lines.append(f"|---{stripped_line}")
        else:
            # Data row
            if stripped_line.startswith("|"):
                new_lines.append(f"| {question_counter} {stripped_line}")
                question_counter += 1
            else:
                # Handle potential malformed table rows or text outside the table
                if "|" in stripped_line:  # It has columns but maybe missing start pipe
                    new_lines.append(f"| {question_counter} | {stripped_line}")
                    question_counter += 1
                else:
                    new_lines.append(line)

    return "\n".join(new_lines)


def reauthenticate_google() -> bool:
    """
    Deletes the cached Google OAuth token and triggers a new authentication flow.
    Returns True if successful, False otherwise.
    """
    token_file = Path.home() / ".google_client_token.json"
    creds_file = Path.home() / ".google_client_secret.json"

    if token_file.exists():
        try:
            token_file.unlink()
            print(f"Deleted cached token: {token_file}")
        except Exception as e:
            print(f"Error deleting token: {e}")
            return False
    else:
        print("No cached token found. Starting fresh authentication.")

    if not creds_file.exists():
        print(f"Error: Client secrets not found at {creds_file}")
        print("Please ensure you have your Google OAuth client secrets JSON file.")
        return False

    try:
        # Lazy imports to avoid dependency issues for non-workspace users
        from google_auth_oauthlib.flow import InstalledAppFlow

        scopes = ["https://www.googleapis.com/auth/drive.file"]
        flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), scopes)
        creds = flow.run_local_server(port=0)

        token_file.write_text(creds.to_json())
        print(f"Successfully re-authenticated. Token saved to {token_file}")
        return True
    except ImportError:
        print(
            "Error: google-auth-oauthlib not installed. "
            "Run 'uv sync --extra workspace' to install it."
        )
        return False
    except Exception as e:
        print(f"Error during re-authentication: {e}")
        return False
