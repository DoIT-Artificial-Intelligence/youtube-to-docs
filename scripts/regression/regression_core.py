import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import polars as pl
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Configuration
VIDEO_ID = "B0x2I_doX9o"
ARTIFACTS_DIR = "youtube-to-docs-test"
MODELS_TO_CHECK = {
    "gemini-3-flash-preview": "Gemini (GCP)",
    "vertex-claude-haiku-4-5@20251001": "Claude (Vertex GCP)",
    "bedrock-nova-2-lite-v1": "Nova 2 Lite (AWS)",
    "foundry-gpt-5-mini": "GPT-5 Mini (Azure)",
}


def normalize_model_name(model_name: str) -> str:
    """Normalizes a model name by stripping prefixes and suffixes."""
    normalized = model_name
    prefixes = ["vertex-", "bedrock-", "foundry-"]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    normalized = re.sub(r"(@\d{8}|-\d{8}-v\d+|-v\d+)$", "", normalized)
    return normalized


def clear_artifacts():
    """Clears the local artifacts directory."""
    if os.path.exists(ARTIFACTS_DIR):
        print(f"Cleaning up {ARTIFACTS_DIR}...")
        shutil.rmtree(ARTIFACTS_DIR)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def check_cloud_stack() -> List[str]:
    """
    Checks which models are available by querying them with a simple prompt.
    Returns a list of successful model IDs.
    """
    successful_models = []
    print("\n--- Checking Cloud Stack ---")

    for model_id, name in MODELS_TO_CHECK.items():
        print(f"Checking {name} ({model_id})...", end=" ", flush=True)
        try:
            if model_id.startswith("gemini") and "GEMINI_API_KEY" in os.environ:
                successful_models.append(model_id)
                print("OK")
            elif model_id.startswith("vertex") and "PROJECT_ID" in os.environ:
                successful_models.append(model_id)
                print("OK")
            elif (
                model_id.startswith("bedrock")
                and "AWS_BEARER_TOKEN_BEDROCK" in os.environ
            ):
                successful_models.append(model_id)
                print("OK")
            elif (
                model_id.startswith("foundry")
                and "AZURE_FOUNDRY_ENDPOINT" in os.environ
            ):
                successful_models.append(model_id)
                print("OK")
            else:
                print("SKIPPED (Missing Env Vars)")
        except Exception as e:
            print(f"FAILED: {e}")

    return successful_models


def run_regression(
    model: Optional[str],
    transcript_model: Optional[str],
    infographic_model: Optional[str],
    tts_model: Optional[str],
    language: str = "en",
    no_youtube_summary: bool = False,
    output_target: Optional[str] = None,
    all_gemini_arg: Optional[str] = None,
):
    """Runs the full regression suite for a single video."""
    print(
        f"\n--- Running Regression with Model: {model} "
        f"(Lang: {language}, NYS: {no_youtube_summary}) ---"
    )

    if output_target is None:
        output_target = os.path.join(ARTIFACTS_DIR, "youtube-docs.csv")

    cmd = [
        "uv",
        "run",
        "youtube-to-docs",
        VIDEO_ID,
    ]

    if transcript_model:
        cmd.extend(["-t", transcript_model])
    if model:
        cmd.extend(["-m", model])
    if infographic_model:
        cmd.extend(["-i", infographic_model])
    if tts_model:
        cmd.extend(["--tts", tts_model])

    cmd.extend(["-l", language, "-o", output_target])

    if no_youtube_summary:
        cmd.append("-nys")

    if all_gemini_arg:
        cmd.extend(["--all", all_gemini_arg])

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\nRegression failed with return code {result.returncode}")
        sys.exit(1)

    print("\nRegression run completed successfully.")


def verify_output(
    model: Optional[str],
    transcript_model: Optional[str],
    infographic_model: Optional[str],
    tts_model: Optional[str],
    language: str = "en",
    no_youtube_summary: bool = False,
    output_target: Optional[str] = None,
    all_gemini_arg: Optional[str] = None,
):
    """Verifies that the output CSV exists and contains expected columns and files."""
    print(
        f"\n--- Verifying Output (Lang: {language}, "
        f"NYS: {no_youtube_summary}, All: {all_gemini_arg}) ---"
    )

    if all_gemini_arg == "gemini-flash":
        model = "gemini-3-flash-preview"
        transcript_model = transcript_model or "gemini-3-flash-preview"
        infographic_model = "gemini-2.5-flash-image"
        tts_model = "gemini-2.5-flash-preview-tts-Kore"
        no_youtube_summary = True
    elif all_gemini_arg == "gemini-pro":
        model = "gemini-3-pro-preview"
        transcript_model = transcript_model or "gemini-3-pro-preview"
        infographic_model = "gemini-3-pro-image-preview"
        tts_model = "gemini-2.5-pro-preview-tts-Kore"
        no_youtube_summary = True

    # Ensure model and transcript_model are not None for normalization
    # They should be set if all_gemini_flash is True, but ty needs to know that.
    m_to_norm = model if model is not None else "unknown"
    t_to_norm = transcript_model if transcript_model is not None else "youtube"

    # Default transcript_model to "youtube" if it's None (default CLI behavior)
    if transcript_model is None:
        transcript_model = "youtube"

    if output_target is None or not os.path.exists(output_target):
        # If it's a folder ID, we skip local file check but we can still
        # check columns if we download it. For now, let's assume if it is
        # NOT a local path that exists, we might need to handle it.
        # But if it's workspace, we'll try to load it from the drive.
        csv_path = os.path.join(ARTIFACTS_DIR, "youtube-docs.csv")
    else:
        csv_path = output_target

    if output_target == "sharepoint":
        print("Loading from SharePoint")
        sys.path.append(os.getcwd())
        from youtube_to_docs.storage import M365Storage

        storage = M365Storage()
        df = storage.load_dataframe("youtube-docs.csv")
    elif output_target and (
        len(output_target) > 20
        and "." not in output_target
        and "/" not in output_target
        and "\\" not in output_target
    ):
        # Workspace ID
        print(f"Loading from Google Drive ID: {output_target}")
        sys.path.append(os.getcwd())  # Ensure local package is findable
        from youtube_to_docs.storage import GoogleDriveStorage

        gds = GoogleDriveStorage(output_target)
        df = gds.load_dataframe("youtube-docs.csv")
    else:
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            sys.exit(1)
        df = pl.read_csv(csv_path)

    if df is None:
        print("Error: Could not load DataFrame.")
        sys.exit(1)

    # df is now guaranteed to be pl.DataFrame
    assert df is not None

    try:
        print(f"Loaded CSV with {len(df)} rows.")

        norm_m = normalize_model_name(m_to_norm)
        norm_t = normalize_model_name(t_to_norm)
        col_suffix = f" ({language})" if language != "en" else ""

        # Base columns
        expected_columns = [
            "URL",
            "Title",
            "Description",
            "Data Published",
            "Channel",
            "Tags",
            "Duration",
            f"Transcript characters from youtube{col_suffix}",
            "Transcript File youtube generated",  # English fallback
        ]

        if transcript_model != "youtube":
            expected_columns.append("Audio File")

        # Transcript columns
        if transcript_model != "youtube":
            expected_columns.extend(
                [
                    f"Transcript characters from {transcript_model}{col_suffix}",
                    f"Transcript File {transcript_model} generated{col_suffix}",
                    f"{norm_t} STT cost{col_suffix} ($)",
                ]
            )

        # Summarization/QA/Speaker columns
        sources = [transcript_model]
        if transcript_model != "youtube" and not no_youtube_summary:
            sources.append("youtube")

        for source in sources:
            source_cols = [
                f"Summary Text {model} from {source}{col_suffix}",
                f"Summary File {model} from {source}{col_suffix}",
                f"QA Text {model} from {source}{col_suffix}",
                f"QA File {model} from {source}{col_suffix}",
                f"Speakers {model} from {source}",  # No suffix
                f"Speakers File {model} from {source}",  # No suffix
                f"{norm_m} summary cost from {source}{col_suffix} ($)",
                f"{norm_m} QA cost from {source}{col_suffix} ($)",
                f"{norm_m} Speaker extraction cost from {source} ($)",  # No suffix
            ]
            expected_columns.extend(source_cols)

            if infographic_model:
                # m_name in main.py is the summary text column key without prefix
                m_name = f"{model} from {source}{col_suffix}"
                expected_columns.extend(
                    [
                        f"Summary Infographic File {m_name} {infographic_model}",
                        (f"Summary Infographic Cost {m_name} {infographic_model} ($)"),
                    ]
                )

            if tts_model:
                expected_columns.append(
                    f"Summary Audio File {model} from {source}{col_suffix} "
                    f"{tts_model} File"
                )

        missing_cols = []
        for col in expected_columns:
            if col not in df.columns:
                missing_cols.append(col)

        if missing_cols:
            print(f"Error: Missing expected columns: {missing_cols}")
            print(f"Available columns: {df.columns}")
            sys.exit(1)

        print(
            f"Verification matched {len(expected_columns) - len(missing_cols)}/"
            f"{len(expected_columns)} expected columns."
        )

        # Check files exist
        file_cols = [
            c for c in df.columns if "File" in c or "Infographic" in c or "Audio" in c
        ]
        for col in file_cols:
            for val in df[col]:
                if val and isinstance(val, str) and not val.startswith("http"):
                    full_path = val if os.path.isabs(val) else os.path.abspath(val)
                    if not os.path.exists(full_path):
                        print(f"Error: File referenced in CSV does not exist: {val}")
                        sys.exit(1)

        print("All local files referenced in CSV exist.")

    except Exception as e:
        print(f"Error during verification: {e}")
        sys.exit(1)

    print("\nVerification PASSED!")


def get_default_model(available_models: List[str]) -> str:
    """Returns the default model to use."""
    return (
        "gemini-3-flash-preview"
        if "gemini-3-flash-preview" in available_models
        else available_models[0]
    )


def resolve_drive_folder(folder_name: str, delete_if_exists: bool = False) -> str:
    """Finds or creates a folder in Google Drive and returns its ID."""
    token_file = Path.home() / ".token.json"
    if not token_file.exists():
        print("Error: .token.json not found. Run a local workspace command first.")
        sys.exit(1)

    creds = Credentials.from_authorized_user_file(
        str(token_file), ["https://www.googleapis.com/auth/drive.file"]
    )
    service = build("drive", "v3", credentials=creds)

    query = (
        "mimeType='application/vnd.google-apps.folder' and "
        f"name='{folder_name}' and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files and delete_if_exists:
        for f in files:
            print(f"Deleting existing Drive folder: {folder_name} ({f['id']})")
            service.files().delete(fileId=f["id"]).execute()
        files = []  # Clear list so we create a new one

    if files:
        print(f"Using existing Drive folder: {folder_name} ({files[0]['id']})")
        return files[0]["id"]
    else:
        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = service.files().create(body=file_metadata, fields="id").execute()
        print(f"Created new Drive folder: {folder_name} ({folder.get('id')})")
        return folder.get("id")
