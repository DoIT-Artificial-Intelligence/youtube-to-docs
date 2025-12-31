import io
import os
import time
from pathlib import Path

import pytest
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CREDS_FILE = Path.home() / ".google_client_secret.json"
TOKEN_FILE = Path.home() / ".token.json"


def get_creds():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDS_FILE.exists():
                raise FileNotFoundError(
                    f"Client secrets not found at {CREDS_FILE.absolute()}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return creds


@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="Skipping this test on CI environment"
)
def test_google_drive_upload():
    markdown_text = "This is markdown. \n\n # Header 1 \n\n ## Header 2 \n\n plain text"
    file_name = "test"
    try:
        service = build("drive", "v3", credentials=get_creds())
        fh = io.BytesIO(markdown_text.encode("utf-8"))
        media = MediaIoBaseUpload(fh, mimetype="text/markdown", resumable=True)
        print("Uploading...")
        file_meta = {
            "name": file_name,
            "mimeType": "application/vnd.google-apps.document",
        }
        doc = (
            service.files()
            .create(body=file_meta, media_body=media, fields="id, webViewLink")
            .execute()
        )
        file_id = doc.get("id")
        print(f"Success! ID: {file_id}\nLink: {doc.get('webViewLink')}")
        print("Deleting in 5 seconds...")
        time.sleep(5)
        service.files().delete(fileId=file_id).execute()
        print("Done.")

    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="Skipping this test on CI environment"
)
def test_google_sheet_upload():
    csv_content = "data\nrow 1\nrow 2"
    file_name = "test_sheet"
    try:
        service = build("drive", "v3", credentials=get_creds())
        fh = io.BytesIO(csv_content.encode("utf-8"))
        # Upload as text/csv, asking Drive to convert to a Google Sheet
        media = MediaIoBaseUpload(fh, mimetype="text/csv", resumable=True)
        print("Uploading CSV to Sheet...")
        file_meta = {
            "name": file_name,
            "mimeType": "application/vnd.google-apps.spreadsheet",
        }
        sheet = (
            service.files()
            .create(body=file_meta, media_body=media, fields="id, webViewLink")
            .execute()
        )
        file_id = sheet.get("id")
        print(f"Success! ID: {file_id}\nLink: {sheet.get('webViewLink')}")
        print("Deleting in 5 seconds...")
        time.sleep(5)
        service.files().delete(fileId=file_id).execute()
        print("Done.")

    except Exception as e:
        pytest.fail(f"An error occurred: {e}")
