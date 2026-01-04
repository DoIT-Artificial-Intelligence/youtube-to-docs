import atexit
import base64
import io
import json
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import polars as pl
import requests


class Storage(ABC):
    """Abstract base class for file storage operations."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Checks if a file exists at the given path."""
        pass

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Reads text content from a file."""
        pass

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Reads byte content from a file."""
        pass

    @abstractmethod
    def write_text(self, path: str, content: str) -> str:
        """Writes text content to a file. Returns the path or link to the file."""
        pass

    @abstractmethod
    def write_bytes(self, path: str, content: bytes) -> str:
        """Writes byte content to a file. Returns the path or link to the file."""
        pass

    @abstractmethod
    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        """Loads a DataFrame from a CSV file (or Sheet)."""
        pass

    @abstractmethod
    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        """Saves a DataFrame to a CSV file (or Sheet). Returns the path or link."""
        pass

    @abstractmethod
    def ensure_directory(self, path: str) -> None:
        """Ensures a directory exists."""
        pass

    @abstractmethod
    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        """Uploads a local file to the storage. Returns the path or link."""
        pass

    @abstractmethod
    def get_full_path(self, path: str) -> str:
        """Returns the full path or link to the file."""
        pass

    @abstractmethod
    def get_local_file(
        self, path: str, download_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        Ensures the file is available locally.
        If it's already local, returns the path.
        If it's remote, downloads it to download_dir (or a temp file) and returns
        the path.
        Returns None if retrieval fails.
        """
        pass


class LocalStorage(Storage):
    """Implementation of Storage for the local filesystem."""

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def read_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def write_text(self, path: str, content: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(path)

    def write_bytes(self, path: str, content: bytes) -> str:
        with open(path, "wb") as f:
            f.write(content)
        return os.path.abspath(path)

    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        try:
            return pl.read_csv(path)
        except Exception:
            return None

    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        df.write_csv(path)
        return os.path.abspath(path)

    def ensure_directory(self, path: str) -> None:
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        if os.path.abspath(local_path) != os.path.abspath(target_path):
            shutil.copy2(local_path, target_path)
        return os.path.abspath(target_path)

    def get_full_path(self, path: str) -> str:
        return os.path.abspath(path)

    def get_local_file(
        self, path: str, download_dir: Optional[str] = None
    ) -> Optional[str]:
        if self.exists(path):
            return os.path.abspath(path)
        return None


class GoogleDriveStorage(Storage):
    """Implementation of Storage for Google Drive."""

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self, output_arg: str):
        from googleapiclient.discovery import build

        self.creds = self._get_creds()
        self.service = build("drive", "v3", credentials=self.creds)
        self.sheets_service = build("sheets", "v4", credentials=self.creds)
        self.root_folder_id = self._resolve_root_folder_id(output_arg)
        # Cache for folder IDs to avoid constant lookups
        self.folder_cache: dict[str, str] = {}
        # Cache for file metadata (path -> dict)
        self.file_cache: dict[str, dict] = {}

    def _get_creds(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        creds_file = Path.home() / ".google_client_secret.json"
        token_file = Path.home() / ".token.json"

        if token_file.exists():
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not creds_file.exists():
                    raise FileNotFoundError(
                        f"Client secrets not found at {creds_file.absolute()}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_file), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            token_file.write_text(creds.to_json())
        return creds

    def _resolve_root_folder_id(self, output_arg: str) -> str:
        if output_arg in ("workspace", "w"):
            folder_name = "youtube-to-docs-artifacts"
            # specific logic to find or create "youtube-to-docs-artifacts" in root
            query = (
                "mimeType='application/vnd.google-apps.folder' and "
                f"name='{folder_name}' and trashed=false"
            )
            results = (
                self.service.files().list(q=query, fields="files(id, name)").execute()
            )
            files = results.get("files", [])
            if files:
                print(f"Using existing folder: {folder_name} ({files[0]['id']})")
                return files[0]["id"]
            else:
                file_metadata = {
                    "name": folder_name,
                    "mimeType": "application/vnd.google-apps.folder",
                }
                folder = (
                    self.service.files()
                    .create(body=file_metadata, fields="id")
                    .execute()
                )
                print(f"Created folder: {folder_name} ({folder.get('id')})")
                return folder.get("id")
        else:
            # Assume it is a Folder ID
            return output_arg

    def _get_parent_id(self, path: str) -> str:
        """
        Given a 'path' which mimics os.path.join(base_dir, subfolder, filename),
        figure out the parent folder ID.
        We assume 'path' starts with the storage's 'virtual' paths.
        However, main.py passes paths joined with os.path.sep.
        We need to handle normalization.
        """
        if path.startswith("http"):
            raise ValueError(f"Cannot resolve parent folder for URL: {path}")

        parts = Path(path).parts
        parent_id = self.root_folder_id

        # Iterate over parts except the last one (filename)
        for part in parts[:-1]:
            if part == "." or part == "youtube-to-docs-artifacts":  # Root or redundant
                continue

            cache_key = f"{parent_id}/{part}"
            if cache_key in self.folder_cache:
                parent_id = self.folder_cache[cache_key]
                continue

            # Search for folder in parent_id
            query = (
                "mimeType='application/vnd.google-apps.folder' and "
                f"name='{part}' and '{parent_id}' in parents and trashed=false"
            )
            results = self.service.files().list(q=query, fields="files(id)").execute()
            files = results.get("files", [])

            if files:
                current_id = files[0]["id"]
            else:
                # Create it
                file_metadata = {
                    "name": part,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [parent_id],
                }
                folder = (
                    self.service.files()
                    .create(body=file_metadata, fields="id")
                    .execute()
                )
                current_id = folder.get("id")

            self.folder_cache[cache_key] = current_id
            parent_id = current_id

        return parent_id

    def _get_file_metadata(self, path: str) -> Optional[dict]:
        if path in self.file_cache:
            return self.file_cache[path]

        # See if we can find the file given the path structure
        # path is like "summary-files/my-video.txt"
        parent_id = self._get_parent_id(path)
        filename = Path(path).name

        # Handle "Google Sheet" name special case for main csv
        # If path ends in .csv, we might look for a Sheet
        if filename == "youtube-docs.csv":
            filename = "youtube-docs"  # We store it as a Sheet

        query = f"name='{filename}' and '{parent_id}' in parents and trashed=false"
        results = (
            self.service.files()
            .list(q=query, fields="files(id, webViewLink, mimeType)")
            .execute()
        )
        files = results.get("files", [])
        if files:
            self.file_cache[path] = files[0]
            return files[0]
        return None

    def _get_file_id(self, path: str) -> Optional[str]:
        metadata = self._get_file_metadata(path)
        return metadata["id"] if metadata else None

    def _extract_id_from_url(self, url: str) -> Optional[str]:
        # Typical patterns:
        # https://docs.google.com/document/d/<ID>/edit
        # https://docs.google.com/spreadsheets/d/<ID>/edit
        # https://drive.google.com/file/d/<ID>/view
        match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
        if match:
            return match.group(1)
        return None

    def exists(self, path: str) -> bool:
        # Check if path starts with http - it's a link we already stored
        if path.startswith("http"):
            return True  # Assume links exist
        return self._get_file_id(path) is not None

    def read_text(self, path: str) -> str:
        # This implies downloading the Google Doc as text/markdown
        if path.startswith("http"):
            file_id = self._extract_id_from_url(path)
        else:
            file_id = self._get_file_id(path)

        if not file_id:
            raise FileNotFoundError(f"File not found: {path} (extracted id: {file_id})")

        # Export as plain text
        # If it is a google doc
        # mimeType = application/vnd.google-apps.document
        # export mimeType = text/plain
        try:
            content = (
                self.service.files()
                .export(fileId=file_id, mimeType="text/plain")
                .execute()
            )
            return content.decode("utf-8")
        except Exception as e:
            from googleapiclient.errors import HttpError
            from googleapiclient.http import MediaIoBaseDownload

            if not isinstance(e, HttpError):
                raise e

            # Maybe it is a binary file or text file? use get_media
            # But we are in read_text
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()

            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            return fh.getvalue().decode("utf-8")

    def read_bytes(self, path: str) -> bytes:
        if path.startswith("http"):
            file_id = self._extract_id_from_url(path)
        else:
            file_id = self._get_file_id(path)

        if not file_id:
            raise FileNotFoundError(f"File not found: {path}")

        from googleapiclient.http import MediaIoBaseDownload

        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()

        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return fh.getvalue()

    def write_text(self, path: str, content: str) -> str:
        parent_id = self._get_parent_id(path)
        filename = Path(path).name

        # Check if exists to overwrite?
        existing_id = self._get_file_id(path)

        file_metadata = {
            "name": filename,
            "mimeType": "application/vnd.google-apps.document",  # Convert to Doc
            "parents": [parent_id],
        }

        from googleapiclient.http import MediaIoBaseUpload

        fh = io.BytesIO(content.encode("utf-8"))
        media = MediaIoBaseUpload(fh, mimetype="text/markdown", resumable=True)

        if existing_id:
            # Update
            file_metadata.pop("parents", None)  # Can't reparent easily on update
            file = (
                self.service.files()
                .update(
                    fileId=existing_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )
        else:
            file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id, webViewLink")
                .execute()
            )

        # Update cache
        if file.get("id"):
            self.file_cache[path] = file

        return file.get("webViewLink")

    def write_bytes(self, path: str, content: bytes) -> str:
        parent_id = self._get_parent_id(path)
        filename = Path(path).name

        existing_id = self._get_file_id(path)

        file_metadata = {
            "name": filename,
            "parents": [parent_id],
        }

        from googleapiclient.http import MediaIoBaseUpload

        fh = io.BytesIO(content)
        mime_type = "application/octet-stream"
        if filename.endswith(".wav"):
            mime_type = "audio/wav"
        elif filename.endswith(".png"):
            mime_type = "image/png"
        media = MediaIoBaseUpload(fh, mimetype=mime_type, resumable=True)

        if existing_id:
            file_metadata.pop("parents", None)
            file = (
                self.service.files()
                .update(
                    fileId=existing_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )
        else:
            file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id, webViewLink")
                .execute()
            )

        # Update cache
        if file.get("id"):
            self.file_cache[path] = file

        return file.get("webViewLink")

    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        # path is expected to be "youtube-docs.csv" usually
        # We look for "youtube-docs" Sheet
        file_id = self._get_file_id(path)
        if not file_id:
            return None

        # Export sheet to CSV
        try:
            csv_content = (
                self.service.files()
                .export(fileId=file_id, mimeType="text/csv")
                .execute()
            )
            return pl.read_csv(io.BytesIO(csv_content))
        except Exception as e:
            print(f"Error loading dataframe from drive: {e}")
            return None

    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        parent_id = self._get_parent_id(path)
        filename = Path(path).stem  # youtube-docs

        # Convert DF to CSV string
        csv_buffer = io.BytesIO()
        df.write_csv(csv_buffer)
        csv_buffer.seek(0)

        existing_id = self._get_file_id(path)

        file_metadata = {
            "name": filename,
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "parents": [parent_id],
        }

        from googleapiclient.http import MediaIoBaseUpload

        media = MediaIoBaseUpload(csv_buffer, mimetype="text/csv", resumable=True)

        if existing_id:
            # Update
            file_metadata.pop("parents", None)
            file = (
                self.service.files()
                .update(
                    fileId=existing_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )
        else:
            file = (
                self.service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )

        # Update sheet properties (freeze header and ensure min rows)
        try:
            spreadsheet_id = file.get("id")
            # Fetch the spreadsheet to get the actual sheetId of the first sheet
            spreadsheet = (
                self.sheets_service.spreadsheets()
                .get(spreadsheetId=spreadsheet_id)
                .execute()
            )
            sheets = spreadsheet.get("sheets", [])
            if sheets:
                sheet = sheets[0]
                sheet_id = sheet.get("properties", {}).get("sheetId")
                grid_props = sheet.get("properties", {}).get("gridProperties", {})
                current_rows = grid_props.get("rowCount", 0)

                if sheet_id is not None:
                    requests = []
                    # Freeze rows/cols
                    requests.append(
                        {
                            "updateSheetProperties": {
                                "properties": {
                                    "sheetId": sheet_id,
                                    "gridProperties": {
                                        "frozenRowCount": 1,
                                        "frozenColumnCount": 1,
                                    },
                                },
                                "fields": (
                                    "gridProperties(frozenRowCount,frozenColumnCount)"
                                ),
                            }
                        }
                    )

                    # Ensure minimum rows (1000) for better UX
                    if current_rows < 1000:
                        requests.append(
                            {
                                "updateSheetProperties": {
                                    "properties": {
                                        "sheetId": sheet_id,
                                        "gridProperties": {
                                            "rowCount": 1000,
                                        },
                                    },
                                    "fields": "gridProperties(rowCount)",
                                }
                            }
                        )

                    self.sheets_service.spreadsheets().batchUpdate(
                        spreadsheetId=spreadsheet_id, body={"requests": requests}
                    ).execute()
        except Exception as e:
            print(f"Warning: Could not update sheet properties: {e}")

        # Update cache
        if file.get("id"):
            self.file_cache[path] = file

        return file.get("webViewLink")

    def ensure_directory(self, path: str) -> None:
        # _get_parent_id creates directories as side effect
        self._get_parent_id(os.path.join(path, "dummy"))

    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        parent_id = self._get_parent_id(target_path)
        filename = Path(target_path).name

        existing_id = self._get_file_id(target_path)

        file_metadata = {"name": filename, "parents": [parent_id]}

        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(local_path, mimetype=content_type, resumable=True)

        if existing_id:
            file_metadata.pop("parents", None)
            file = (
                self.service.files()
                .update(
                    fileId=existing_id,
                    body=file_metadata,
                    media_body=media,
                    fields="id, webViewLink",
                )
                .execute()
            )
        else:
            file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id, webViewLink")
                .execute()
            )

        # Update cache
        if file.get("id"):
            self.file_cache[target_path] = file

        return file.get("webViewLink")

    def get_full_path(self, path: str) -> str:
        """For Drive, returns the path (or link if possible/cached)."""
        if path.startswith("http"):
            return path

        metadata = self._get_file_metadata(path)
        if metadata and "webViewLink" in metadata:
            return metadata["webViewLink"]
        return path

    def get_local_file(
        self, path: str, download_dir: Optional[str] = None
    ) -> Optional[str]:
        if not self.exists(path):
            return None

        if not download_dir:
            import tempfile

            download_dir = tempfile.gettempdir()

        filename = Path(path).name
        local_path = os.path.join(download_dir, filename)

        try:
            data = self.read_bytes(path)
            with open(local_path, "wb") as f:
                f.write(data)
            return local_path
        except Exception as e:
            from googleapiclient.errors import HttpError

            if isinstance(e, (HttpError, OSError)):
                print(f"Error downloading file {path} to local: {e}")
            else:
                raise e
            return None


class M365Storage(Storage):
    """Implementation of Storage for Microsoft 365 (OneDrive/SharePoint)."""

    CLIENT_CONFIG_FILE = Path.home() / ".azure_client.json"
    TOKEN_CACHE_FILE = Path.home() / ".msal_token_cache.json"
    SCOPES = ["Files.ReadWrite"]
    ROOT_FOLDER_NAME = "youtube-to-docs-artifacts"

    def __init__(self):
        self.token = self._get_access_token()
        # Cache for folder paths to avoid constant lookups
        # Map path (relative to root) to webUrl or item metadata
        self.item_cache: dict[str, dict] = {}

    def _get_client_config(self) -> dict:
        if not self.CLIENT_CONFIG_FILE.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {self.CLIENT_CONFIG_FILE}. "
                "Please create it with {'client_id': '...', 'authority': '...'}"
            )
        return json.loads(self.CLIENT_CONFIG_FILE.read_text(encoding="utf-8"))

    def _build_msal_app(self) -> Any:
        import msal

        config = self._get_client_config()
        client_id = config.get("client_id")
        authority = config.get("authority")

        if not client_id or not authority:
            raise ValueError(
                "Config file must contain both 'client_id' and 'authority'"
            )

        cache = msal.SerializableTokenCache()
        if self.TOKEN_CACHE_FILE.exists():
            cache.deserialize(self.TOKEN_CACHE_FILE.read_text(encoding="utf-8"))

        def persist_cache():
            if cache.has_state_changed:
                self.TOKEN_CACHE_FILE.write_text(cache.serialize(), encoding="utf-8")

        atexit.register(persist_cache)

        return msal.PublicClientApplication(
            client_id=client_id,
            authority=authority,
            token_cache=cache,
        )

    def _get_access_token(self) -> str:
        app = self._build_msal_app()
        accounts = app.get_accounts()
        result = None
        if accounts:
            result = app.acquire_token_silent(self.SCOPES, account=accounts[0])

        if not result:
            # Fallback to interactive
            result = app.acquire_token_interactive(self.SCOPES)

        if "access_token" not in result:
            error_msg = (
                result.get("error_description")
                or result.get("error")
                or "Unknown error"
            )
            raise RuntimeError(f"Could not authenticate: {error_msg}")

        return result["access_token"]

    def _get_full_remote_path(self, path: str) -> str:
        """
        Converts a relative path like "summary-files/foo.md" to
        "youtube-to-docs-test-sharepoint/summary-files/foo.md".
        """
        if path.startswith("http"):
            return path
        # Normalize to POSIX style, handling . and ..
        # We assume path is relative to the root of our "virtual" storage

        # Use os.path.normpath to handle ./ and ../
        # If path is "." it returns "."
        norm_path = os.path.normpath(path)

        if norm_path == ".":
            return self.ROOT_FOLDER_NAME

        # Convert backslashes to slashes
        clean_path = norm_path.replace("\\", "/")

        # If it starts with root folder, don't prepend
        if clean_path.startswith(self.ROOT_FOLDER_NAME) and (
            len(clean_path) == len(self.ROOT_FOLDER_NAME)
            or clean_path[len(self.ROOT_FOLDER_NAME)] == "/"
        ):
            return clean_path

        return f"{self.ROOT_FOLDER_NAME}/{clean_path}"

    def _get_item_from_url(self, url: str) -> Optional[dict]:
        # Use shares API to resolve webUrl to DriveItem
        # 1. Base64 encode
        b64 = base64.b64encode(url.encode("utf-8")).decode("utf-8")
        # 2. Make URL safe
        encoded_url = "u!" + b64.rstrip("=").replace("/", "_").replace("+", "-")

        api_url = f"https://graph.microsoft.com/v1.0/shares/{encoded_url}/driveItem"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            resp = requests.get(api_url, headers=headers)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"Warning: Failed to resolve URL {url}: {e}")

        return None

    def _get_item(self, path: str) -> Optional[dict]:
        """Gets item metadata from Graph API."""
        if path.startswith("http"):
            return self._get_item_from_url(path)

        remote_path = self._get_full_remote_path(path)
        if remote_path in self.item_cache:
            return self.item_cache[remote_path]

        # Handle special case: csv file might be stored as xlsx
        filename = Path(path).name
        if filename == "youtube-docs.csv":
            # Check for .xlsx version
            xlsx_path = str(Path(path).with_suffix(".xlsx"))
            remote_xlsx_path = self._get_full_remote_path(xlsx_path)
            encoded_xlsx = quote(remote_xlsx_path)
            url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{encoded_xlsx}"
        else:
            encoded_remote = quote(remote_path)
            url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{encoded_remote}"

        headers = {"Authorization": f"Bearer {self.token}"}
        resp = requests.get(url, headers=headers)

        if resp.status_code == 200:
            data = resp.json()
            self.item_cache[remote_path] = data
            return data
        elif resp.status_code == 404:
            # If we are looking for .md, maybe check .docx?
            if path.endswith(".md"):
                docx_path = str(Path(path).with_suffix(".docx"))
                remote_docx_path = self._get_full_remote_path(docx_path)
                encoded_docx = quote(remote_docx_path)
                url_docx = (
                    f"https://graph.microsoft.com/v1.0/me/drive/root:/{encoded_docx}"
                )
                resp_docx = requests.get(url_docx, headers=headers)
                if resp_docx.status_code == 200:
                    data = resp_docx.json()
                    self.item_cache[remote_path] = data
                    return data
            return None
        else:
            return None

    def exists(self, path: str) -> bool:
        if path.startswith("http"):
            return True
        return self._get_item(path) is not None

    def read_text(self, path: str) -> str:
        if path.startswith("http"):
            pass

        item = self._get_item(path)
        if not item:
            raise FileNotFoundError(f"File not found: {path}")

        download_url = item.get("@microsoft.graph.downloadUrl")
        if not download_url:
            item_id = item["id"]
            download_url = (
                f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/content"
            )

        resp = requests.get(download_url)
        resp.raise_for_status()

        content_bytes = resp.content
        filename = item.get("name", "")

        if filename.endswith(".docx"):
            import pypandoc

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

            try:
                output = pypandoc.convert_file(tmp_path, "md")
                return output
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            return content_bytes.decode("utf-8")

    def read_bytes(self, path: str) -> bytes:
        item = self._get_item(path)
        if not item:
            raise FileNotFoundError(f"File not found: {path}")

        download_url = item.get("@microsoft.graph.downloadUrl")
        if not download_url:
            item_id = item["id"]
            download_url = (
                f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/content"
            )

        resp = requests.get(download_url)
        resp.raise_for_status()
        return resp.content

    def _upload(self, remote_path: str, content: bytes, content_type: str) -> dict:
        encoded_path = quote(remote_path)
        url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{encoded_path}:/content"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": content_type,
        }
        resp = requests.put(url, headers=headers, data=content)
        if not resp.ok:
            raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def write_text(self, path: str, content: str) -> str:
        filename = Path(path).name
        if filename.endswith(".md") or filename.endswith(".txt"):
            import pypandoc

            remote_path = self._get_full_remote_path(
                str(Path(path).with_suffix(".docx"))
            )

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp_docx = tmp.name

            try:
                pypandoc.convert_text(
                    content,
                    to="docx",
                    format="md",
                    outputfile=tmp_docx,
                    extra_args=["--wrap=preserve"],
                )
                with open(tmp_docx, "rb") as f:
                    docx_bytes = f.read()

                item = self._upload(
                    remote_path,
                    docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            finally:
                if os.path.exists(tmp_docx):
                    os.remove(tmp_docx)
        else:
            remote_path = self._get_full_remote_path(path)
            item = self._upload(remote_path, content.encode("utf-8"), "text/plain")

        if "id" in item:
            full_orig_path = self._get_full_remote_path(path)
            self.item_cache[full_orig_path] = item
            if "webUrl" in item:
                # Also cache by remote_path used in upload
                # (which might be diff from full_orig_path if extension changed)
                pass

        return item.get("webUrl", "")

    def write_bytes(self, path: str, content: bytes) -> str:
        remote_path = self._get_full_remote_path(path)

        mime_type = "application/octet-stream"
        if path.endswith(".wav") or path.endswith(".m4a"):
            mime_type = "audio/mp4"
        elif path.endswith(".png"):
            mime_type = "image/png"

        item = self._upload(remote_path, content, mime_type)

        self.item_cache[remote_path] = item
        return item.get("webUrl", "")

    def load_dataframe(self, path: str) -> Optional[pl.DataFrame]:
        filename = Path(path).name
        if filename == "youtube-docs.csv":
            xlsx_path = str(Path(path).with_suffix(".xlsx"))
            try:
                xlsx_bytes = self.read_bytes(xlsx_path)
                return pl.read_excel(io.BytesIO(xlsx_bytes))
            except FileNotFoundError:
                return None
            except Exception as e:
                print(f"Error loading dataframe from OneDrive: {e}")
                return None
        return None

    def save_dataframe(self, df: pl.DataFrame, path: str) -> str:
        filename = Path(path).name
        if filename == "youtube-docs.csv":
            xlsx_path = str(Path(path).with_suffix(".xlsx"))
            remote_path = self._get_full_remote_path(xlsx_path)

            with io.BytesIO() as output:
                df.write_excel(output, freeze_panes=(1, 1))
                xlsx_bytes = output.getvalue()

            item = self._upload(
                remote_path,
                xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            self.item_cache[self._get_full_remote_path(path)] = item
            self.item_cache[remote_path] = item

            return item.get("webUrl", "")

        return ""

    def ensure_directory(self, path: str) -> None:
        remote_path = self._get_full_remote_path(path)
        parts = remote_path.split("/")

        current_path = ""
        for part in parts:
            if not part:
                continue

            if not current_path:
                current_path = part
            else:
                current_path = f"{current_path}/{part}"

            if current_path in self.item_cache:
                continue

            encoded_current = quote(current_path)
            url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{encoded_current}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self.token}"})

            if resp.status_code == 200:
                self.item_cache[current_path] = resp.json()
            else:
                parent_path = current_path.rsplit("/", 1)[0]
                if parent_path == current_path:
                    post_url = "https://graph.microsoft.com/v1.0/me/drive/root/children"
                else:
                    encoded_parent = quote(parent_path)
                    post_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{encoded_parent}:/children"

                body = {
                    "name": part,
                    "folder": {},
                    "@microsoft.graph.conflictBehavior": "rename",
                }
                resp_create = requests.post(
                    post_url,
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=body,
                )
                if resp_create.status_code in (200, 201):
                    self.item_cache[current_path] = resp_create.json()
                elif resp_create.status_code == 409:
                    pass
                else:
                    print(
                        f"Warning: Could not create folder {current_path}: "
                        f"{resp_create.text}"
                    )

    def upload_file(
        self, local_path: str, target_path: str, content_type: Optional[str] = None
    ) -> str:
        with open(local_path, "rb") as f:
            content = f.read()

        parent = str(Path(target_path).parent)
        if parent != ".":
            self.ensure_directory(parent)

        if not content_type:
            content_type = "application/octet-stream"

        return self.write_bytes(target_path, content)

    def get_full_path(self, path: str) -> str:
        if path.startswith("http"):
            return path

        item = self._get_item(path)
        if item:
            return item.get("webUrl", path)
        return path

    def get_local_file(
        self, path: str, download_dir: Optional[str] = None
    ) -> Optional[str]:
        if not self.exists(path):
            return None

        if not download_dir:
            import tempfile

            download_dir = tempfile.gettempdir()

        filename = Path(path).name
        local_path = os.path.join(download_dir, filename)

        try:
            if path.endswith(".md") or path.endswith(".txt"):
                content = self.read_text(path)
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                content_bytes = self.read_bytes(path)
                with open(local_path, "wb") as f:
                    f.write(content_bytes)
            return local_path
        except Exception as e:
            print(f"Error downloading {path}: {e}")
            return None
