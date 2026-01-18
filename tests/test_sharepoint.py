import atexit
import io
import json
import os
import random
import tempfile
import time
from pathlib import Path

import msal
import polars as pl
import pypandoc
import pytest
import requests

CLIENT_CONFIG_FILE = Path.home() / ".azure_client.json"
TOKEN_CACHE_FILE = Path.home() / ".msal_token_cache.json"
SCOPES = ["Files.ReadWrite"]


def get_client_config() -> dict:
    if not CLIENT_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {CLIENT_CONFIG_FILE}. "
            "Please create it with {'client_id': '...', 'authority': '...'}"
        )
    return json.loads(CLIENT_CONFIG_FILE.read_text(encoding="utf-8"))


def build_msal_app() -> msal.PublicClientApplication:
    config = get_client_config()
    client_id = config.get("client_id")
    authority = config.get("authority")

    if not client_id or not authority:
        raise ValueError("Config file must contain both 'client_id' and 'authority'")

    cache = msal.SerializableTokenCache()
    if TOKEN_CACHE_FILE.exists():
        cache.deserialize(TOKEN_CACHE_FILE.read_text(encoding="utf-8"))

    def persist_cache():
        if cache.has_state_changed:
            TOKEN_CACHE_FILE.write_text(cache.serialize(), encoding="utf-8")

    atexit.register(persist_cache)

    return msal.PublicClientApplication(
        client_id=client_id,
        authority=authority,
        token_cache=cache,
    )


def get_access_token() -> str:
    app = build_msal_app()
    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])

    if not result:
        # In a real test environment (CI), we can't interactively login.
        # But this test is skipped in CI, so for local dev it's fine.
        result = app.acquire_token_interactive(SCOPES)

    if "access_token" not in result:
        error_msg = (
            result.get("error_description") or result.get("error") or "Unknown error"
        )
        raise RuntimeError(f"Could not authenticate: {error_msg}")

    return result["access_token"]


def convert_md_to_docx_bytes(md_text: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
        tmp_docx_path = tmp_docx.name

    try:
        pypandoc.convert_text(
            md_text,
            to="docx",
            format="md",
            outputfile=tmp_docx_path,
            extra_args=["--wrap=preserve"],
        )
        return Path(tmp_docx_path).read_bytes()
    finally:
        try:
            os.remove(tmp_docx_path)
        except FileNotFoundError:
            pass


def convert_csv_to_xlsx_bytes(csv_text: str) -> bytes:
    df = pl.read_csv(io.StringIO(csv_text))
    with io.BytesIO() as output:
        df.write_excel(output)
        return output.getvalue()


def upload_to_onedrive(
    token: str,
    file_bytes: bytes,
    file_name: str,
    content_type: str,
    folder: str = "",
) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": content_type,
    }

    folder = folder.strip().strip("/")
    if folder:
        remote_path = f"{folder}/{file_name}"
    else:
        remote_path = file_name

    upload_url = (
        f"https://graph.microsoft.com/v1.0/me/drive/root:/{remote_path}:/content"
    )
    resp = requests.put(upload_url, headers=headers, data=file_bytes)

    if not resp.ok:
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")

    return resp.json()


def delete_with_retry(token: str, file_id: str, max_tries: int = 8) -> None:
    url = f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "If-Match": "*",
    }

    for attempt in range(1, max_tries + 1):
        resp = requests.delete(url, headers=headers)

        if resp.status_code in (204, 200):
            return

        if resp.status_code in (423, 409):
            sleep_s = min(30.0, 2.0**attempt) + random.random()
            time.sleep(sleep_s)
            continue

        raise RuntimeError(f"Delete failed ({resp.status_code}): {resp.text}")

    raise RuntimeError(f"Could not delete after {max_tries} tries; still locked.")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="Skipping this test on CI environment"
)
def test_markdown_to_word_upload():
    try:
        # Check if config exists before proceeding, otherwise skip or fail gracefully
        if not CLIENT_CONFIG_FILE.exists():
            pytest.skip("Azure client config file not found.")

        token = get_access_token()
        markdown_text = "# Test Header\n\nThis is a test document."
        docx_bytes = convert_md_to_docx_bytes(markdown_text)

        file_name = "test_render.docx"
        print(f"Uploading '{file_name}' to OneDrive...")
        item = upload_to_onedrive(
            token,
            docx_bytes,
            file_name=file_name,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        file_id = item.get("id")
        web_url = item.get("webUrl")
        print(f"Success! ID: {file_id}\nLink: {web_url}")

        if not isinstance(file_id, str):
            raise RuntimeError("Upload response did not contain a valid ID")

        # Cleanup
        print("Deleting uploaded file...")
        delete_with_retry(token, file_id)

    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="Skipping this test on CI environment"
)
def test_csv_to_excel_upload():
    try:
        if not CLIENT_CONFIG_FILE.exists():
            pytest.skip("Azure client config file not found.")

        token = get_access_token()
        csv_text = "ID,Name,Role\n1,Alice,Engineer\n2,Bob,Manager"
        xlsx_bytes = convert_csv_to_xlsx_bytes(csv_text)

        file_name = "test_sheet.xlsx"
        print(f"Uploading '{file_name}' to OneDrive...")
        item = upload_to_onedrive(
            token,
            xlsx_bytes,
            file_name=file_name,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        file_id = item.get("id")
        web_url = item.get("webUrl")
        print(f"Success! ID: {file_id}\nLink: {web_url}")

        if not isinstance(file_id, str):
            raise RuntimeError("Upload response did not contain a valid ID")

        # Cleanup
        print("Deleting uploaded file...")
        delete_with_retry(token, file_id)

    except Exception as e:
        pytest.fail(f"An error occurred: {e}")
