import io
import os
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import polars as pl
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


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
    def save_dataframe(self, df: pl.DataFrame, path: str) -> None:
        """Saves a DataFrame to a CSV file (or Sheet)."""
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


class LocalStorage(Storage):
    """Implementation of Storage for the local filesystem."""

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def read_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
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

    def save_dataframe(self, df: pl.DataFrame, path: str) -> None:
        df.write_csv(path)

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


class GoogleDriveStorage(Storage):
    """Implementation of Storage for Google Drive."""

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self, output_arg: str):
        self.creds = self._get_creds()
        self.service = build("drive", "v3", credentials=self.creds)
        self.root_folder_id = self._resolve_root_folder_id(output_arg)
        # Cache for folder IDs to avoid constant lookups
        self.folder_cache: dict[str, str] = {}
        # Simple cache for file existence/IDs (path -> id)
        self.file_cache: dict[str, str] = {}

    def _get_creds(self):
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
        if output_arg == "workspace":
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
            # Verify?
            return output_arg

    def _get_parent_id(self, path: str) -> str:
        """
        Given a 'path' which mimics os.path.join(base_dir, subfolder, filename),
        figure out the parent folder ID.
        We assume 'path' starts with the storage's 'virtual' paths.
        However, main.py passes paths joined with os.path.sep.
        We need to handle normalization.
        """
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

    def _get_file_id(self, path: str) -> Optional[str]:
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
            return files[0]["id"]
        return None

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
        except Exception:
            # Maybe it is a binary file or text file? use get_media
            # But we are in read_text
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseUpload(fh, mimetype="text/plain")
            # Wait, MediaIoBaseUpload is for upload.
            from googleapiclient.http import MediaIoBaseDownload

            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            return fh.getvalue().decode("utf-8")

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

        fh = io.BytesIO(content.encode("utf-8"))
        media = MediaIoBaseUpload(fh, mimetype="text/plain", resumable=True)

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

        return file.get("webViewLink")

    def write_bytes(self, path: str, content: bytes) -> str:
        parent_id = self._get_parent_id(path)
        filename = Path(path).name

        existing_id = self._get_file_id(path)

        file_metadata = {
            "name": filename,
            "parents": [parent_id],
        }

        fh = io.BytesIO(content)
        media = MediaIoBaseUpload(
            fh, mimetype="application/octet-stream", resumable=True
        )

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

    def save_dataframe(self, df: pl.DataFrame, path: str) -> None:
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

        media = MediaIoBaseUpload(csv_buffer, mimetype="text/csv", resumable=True)

        if existing_id:
            # Update
            file_metadata.pop("parents", None)
            self.service.files().update(
                fileId=existing_id, body=file_metadata, media_body=media
            ).execute()
        else:
            self.service.files().create(body=file_metadata, media_body=media).execute()

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

        media = MediaIoBaseUpload(local_path, mimetype=content_type, resumable=True)

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

        return file.get("webViewLink")

    def get_full_path(self, path: str) -> str:
        """For Drive, returns the path (or link if possible/cached)."""
        if path.startswith("http"):
            return path
        return path
