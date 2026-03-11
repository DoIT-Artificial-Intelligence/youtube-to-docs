import asyncio
import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from youtube_to_docs.app import (
    Job,
    _extract_video_id,
    _run_job,
    _scan_artifacts,
    _validate_output_file,
    app,
    jobs,
)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Clear the jobs dict before each test."""
    jobs.clear()
    yield
    jobs.clear()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "YouTube to Docs" in resp.text


# ---------------------------------------------------------------------------
# GET /api/model-suites
# ---------------------------------------------------------------------------


def test_model_suites_returns_dict(client):
    resp = client.get("/api/model-suites")
    assert resp.status_code == 200
    data = resp.json()
    assert "gemini-flash" in data
    assert "model" in data["gemini-flash"]


# ---------------------------------------------------------------------------
# POST /api/process
# ---------------------------------------------------------------------------


def test_process_defaults(client):
    with patch("youtube_to_docs.app._run_job", return_value=asyncio.sleep(0)):
        resp = client.post(
            "/api/process",
            json={"url": "https://www.youtube.com/watch?v=abc12345678"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["job_id"] in jobs


def test_process_creates_job_with_correct_video_id(client):
    with patch("youtube_to_docs.app._run_job", return_value=asyncio.sleep(0)):
        resp = client.post("/api/process", json={"url": "abc12345678"})
    job_id = resp.json()["job_id"]
    assert jobs[job_id].video_id == "abc12345678"


def test_process_all_args(client):
    with patch("youtube_to_docs.app._run_job", return_value=asyncio.sleep(0)):
        resp = client.post(
            "/api/process",
            json={
                "url": "abc12345678",
                "output_file": "custom.csv",
                "transcript_source": "gemini-pro",
                "model": "gemini-3-flash-preview",
                "tts_model": "gemini-2.5-flash-preview-tts-Kore",
                "infographic_model": "gemini-3.1-flash-image-preview",
                "alt_text_model": "gemini-3-flash-preview",
                "no_youtube_summary": True,
                "translate": "gemini-3-flash-preview-es",
                "combine_infographic_audio": True,
                "all_suite": "gemini-flash",
                "suggest_corrected_captions": "gemini-3-flash-preview",
                "post_process": '{"word count": "apple"}',
                "verbose": True,
            },
        )
    assert resp.status_code == 200


def test_process_missing_url(client):
    resp = client.post("/api/process", json={})
    assert resp.status_code == 422


def test_process_path_traversal_rejected(client):
    resp = client.post(
        "/api/process",
        json={"url": "abc12345678", "output_file": "../../etc/evil.csv"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}
# ---------------------------------------------------------------------------


def test_get_job_not_found(client):
    resp = client.get("/api/jobs/nonexistent")
    assert resp.status_code == 404


def test_get_job_running(client):
    job = Job(id="test123", video_id="abc12345678")
    jobs["test123"] = job

    resp = client.get("/api/jobs/test123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "test123"
    assert data["status"] == "running"
    assert "artifacts" not in data


def test_get_job_completed(client):
    job = Job(id="test456", video_id="abc12345678", status="completed")
    job.output = ["line 1", "line 2"]
    jobs["test456"] = job

    resp = client.get("/api/jobs/test456")
    data = resp.json()
    assert data["status"] == "completed"
    assert data["output"] == ["line 1", "line 2"]
    assert "artifacts" in data


def test_get_job_error(client):
    job = Job(id="testerr", video_id="abc12345678", status="error")
    job.error = "Something went wrong"
    jobs["testerr"] = job

    resp = client.get("/api/jobs/testerr")
    data = resp.json()
    assert data["status"] == "error"
    assert data["error"] == "Something went wrong"


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}/stream
# ---------------------------------------------------------------------------


def test_stream_not_found(client):
    resp = client.get("/api/jobs/nonexistent/stream")
    assert resp.status_code == 404


def test_stream_completed_job(client):
    job = Job(id="stream1", video_id="abc12345678", status="completed")
    job.output = ["hello", "world"]
    jobs["stream1"] = job

    resp = client.get("/api/jobs/stream1/stream")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data: hello" in resp.text
    assert "data: world" in resp.text
    assert "event: done" in resp.text


def test_stream_error_job(client):
    job = Job(id="stream2", video_id="abc12345678", status="error")
    job.error = "fail"
    job.output = ["started"]
    jobs["stream2"] = job

    resp = client.get("/api/jobs/stream2/stream")
    assert "event: error" in resp.text
    assert "data: fail" in resp.text


# ---------------------------------------------------------------------------
# GET /api/artifacts/{path}
# ---------------------------------------------------------------------------


def test_artifact_path_traversal(client):
    # URL-level traversal gets normalized by the ASGI server (404),
    # while encoded traversal reaches the handler and is rejected (400).
    resp = client.get("/api/artifacts/../../etc/passwd")
    assert resp.status_code in (400, 404)

    resp = client.get("/api/artifacts/%2e%2e/%2e%2e/etc/passwd")
    assert resp.status_code in (400, 404)


def test_artifact_not_found(client):
    resp = client.get("/api/artifacts/youtube-to-docs-artifacts/nonexistent.csv")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# _extract_video_id
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    def test_full_url(self):
        assert (
            _extract_video_id("https://www.youtube.com/watch?v=abc12345678")
            == "abc12345678"
        )

    def test_short_url(self):
        assert _extract_video_id("https://youtu.be/abc12345678") == "abc12345678"

    def test_embed_url(self):
        assert (
            _extract_video_id("https://www.youtube.com/embed/abc12345678")
            == "abc12345678"
        )

    def test_bare_id(self):
        assert _extract_video_id("abc12345678") == "abc12345678"

    def test_comma_separated(self):
        assert _extract_video_id("abc12345678,def12345678") == "abc12345678"

    def test_invalid(self):
        assert _extract_video_id("not-a-valid-video-id") is None

    def test_empty(self):
        assert _extract_video_id("") is None


# ---------------------------------------------------------------------------
# _scan_artifacts
# ---------------------------------------------------------------------------


def test_scan_artifacts_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _scan_artifacts(
        "abc12345678", "youtube-to-docs-artifacts/youtube-docs.csv"
    )
    assert result == []


def test_scan_artifacts_finds_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_dir = tmp_path / "summary-files"
    summary_dir.mkdir()
    (summary_dir / "gemini - abc12345678 - summary.md").write_text("test")

    result = _scan_artifacts(
        "abc12345678", "youtube-to-docs-artifacts/youtube-docs.csv"
    )
    assert len(result) == 1
    assert result[0]["name"] == "gemini - abc12345678 - summary.md"
    assert result[0]["directory"] == "summary-files"


def test_scan_artifacts_ignores_unrelated_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_dir = tmp_path / "summary-files"
    summary_dir.mkdir()
    (summary_dir / "gemini - other_id - summary.md").write_text("test")

    result = _scan_artifacts(
        "abc12345678", "youtube-to-docs-artifacts/youtube-docs.csv"
    )
    assert result == []


def test_scan_artifacts_custom_output_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom_csv = tmp_path / "custom.csv"
    custom_csv.write_text("header")

    result = _scan_artifacts("abc12345678", "custom.csv")
    assert len(result) == 1
    assert result[0]["name"] == "custom.csv"


def test_scan_artifacts_remote_output_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _scan_artifacts("abc12345678", "workspace")
    assert result == []


# ---------------------------------------------------------------------------
# _validate_output_file
# ---------------------------------------------------------------------------


class TestValidateOutputFile:
    def test_valid_relative_path(self):
        result = _validate_output_file("custom.csv")
        assert result == "custom.csv"

    def test_valid_nested_path(self):
        result = _validate_output_file("youtube-to-docs-artifacts/youtube-docs.csv")
        assert result == "youtube-to-docs-artifacts/youtube-docs.csv"

    def test_remote_workspace(self):
        assert _validate_output_file("workspace") == "workspace"
        assert _validate_output_file("w") == "w"

    def test_remote_sharepoint(self):
        assert _validate_output_file("sharepoint") == "sharepoint"
        assert _validate_output_file("s") == "s"

    def test_remote_none(self):
        assert _validate_output_file("none") == "none"
        assert _validate_output_file("n") == "n"

    def test_path_traversal_rejected(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _validate_output_file("../../etc/passwd")
        assert exc_info.value.status_code == 400

    def test_absolute_path_rejected(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _validate_output_file("/etc/passwd")
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Missing environment variables
# ---------------------------------------------------------------------------

_VIDEO_DETAILS = (
    "Test Title",
    "Test Description",
    "2024-01-01",
    "Test Channel",
    "tag1,tag2",
    "0:05:00",
    "https://www.youtube.com/watch?v=abc12345678",
    300.0,
)


def _run_job_sync(job: Job, args: list[str]) -> None:
    """Helper to run _run_job synchronously in tests."""
    asyncio.run(_run_job(job, args))


@patch("youtube_to_docs.main.get_youtube_service")
@patch("youtube_to_docs.main.resolve_video_ids", return_value=["abc12345678"])
@patch("youtube_to_docs.main.get_video_details", return_value=_VIDEO_DETAILS)
@patch(
    "youtube_to_docs.main.fetch_transcript",
    return_value=("Test transcript text", False, ""),
)
def test_missing_gemini_api_key_job_completes(
    _mock_transcript, _mock_details, _mock_resolve, _mock_svc
):
    """When GEMINI_API_KEY is absent the job should complete (not crash) and
    the output should contain a recognisable error message."""
    with tempfile.TemporaryDirectory() as tmp:
        outfile = os.path.join(tmp, "out.csv")
        job = Job(id="nokey1", video_id="abc12345678", output_file=outfile)
        args = [
            "abc12345678",
            "--outfile",
            outfile,
            "--transcript",
            "youtube",
            "--model",
            "gemini-3-flash-preview",
        ]
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            _run_job_sync(job, args)

    assert job.status == "completed"
    combined = "\n".join(job.output)
    assert "GEMINI_API_KEY" in combined


@patch("youtube_to_docs.main.get_youtube_service")
@patch("youtube_to_docs.main.resolve_video_ids", return_value=["abc12345678"])
@patch("youtube_to_docs.main.get_video_details", return_value=_VIDEO_DETAILS)
@patch(
    "youtube_to_docs.main.fetch_transcript",
    return_value=("Test transcript text", False, ""),
)
def test_missing_aws_bearer_token_job_completes(
    _mock_transcript, _mock_details, _mock_resolve, _mock_svc
):
    """When AWS_BEARER_TOKEN_BEDROCK is absent the job should complete and
    include a recognisable error message in the output."""
    with tempfile.TemporaryDirectory() as tmp:
        outfile = os.path.join(tmp, "out.csv")
        job = Job(id="nokey2", video_id="abc12345678", output_file=outfile)
        args = [
            "abc12345678",
            "--outfile",
            outfile,
            "--transcript",
            "youtube",
            "--model",
            "bedrock-nova-2-lite-v1",
        ]
        env = {k: v for k, v in os.environ.items() if k != "AWS_BEARER_TOKEN_BEDROCK"}
        with patch.dict(os.environ, env, clear=True):
            _run_job_sync(job, args)

    assert job.status == "completed"
    combined = "\n".join(job.output)
    assert "AWS_BEARER_TOKEN_BEDROCK" in combined


@patch("youtube_to_docs.main.get_youtube_service")
@patch("youtube_to_docs.main.resolve_video_ids", return_value=["abc12345678"])
@patch("youtube_to_docs.main.get_video_details", return_value=_VIDEO_DETAILS)
@patch(
    "youtube_to_docs.main.fetch_transcript",
    return_value=("Test transcript text", False, ""),
)
def test_missing_azure_credentials_job_completes(
    _mock_transcript, _mock_details, _mock_resolve, _mock_svc
):
    """When Azure Foundry credentials are absent the job should complete and
    include a recognisable error message."""
    with tempfile.TemporaryDirectory() as tmp:
        outfile = os.path.join(tmp, "out.csv")
        job = Job(id="nokey3", video_id="abc12345678", output_file=outfile)
        args = [
            "abc12345678",
            "--outfile",
            outfile,
            "--transcript",
            "youtube",
            "--model",
            "foundry-gpt-5-mini",
        ]
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_API_KEY")
        }
        with patch.dict(os.environ, env, clear=True):
            _run_job_sync(job, args)

    assert job.status == "completed"
    combined = "\n".join(job.output)
    assert "AZURE_FOUNDRY" in combined


@patch("youtube_to_docs.main.get_youtube_service", return_value=None)
@patch(
    "youtube_to_docs.main.fetch_transcript",
    return_value=("Test transcript text", False, ""),
)
def test_missing_youtube_data_api_key_direct_video_id_completes(
    _mock_transcript, _mock_svc
):
    """When YOUTUBE_DATA_API_KEY is absent but input is a direct video ID,
    the job should complete — no YouTube Data API is needed for single videos."""
    with tempfile.TemporaryDirectory() as tmp:
        outfile = os.path.join(tmp, "out.csv")
        job = Job(id="ytkey1", video_id="abc12345678", output_file=outfile)
        args = ["abc12345678", "--outfile", outfile, "--transcript", "youtube"]
        env = {k: v for k, v in os.environ.items() if k != "YOUTUBE_DATA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            _run_job_sync(job, args)

    assert job.status == "completed"


def test_missing_youtube_data_api_key_playlist_errors():
    """When YOUTUBE_DATA_API_KEY is absent and input is a playlist ID,
    the job should end in error state with a clear message."""
    with tempfile.TemporaryDirectory() as tmp:
        outfile = os.path.join(tmp, "out.csv")
        job = Job(
            id="ytkey2",
            video_id="PLabc123",
            output_file=outfile,
        )
        args = ["PLabc123", "--outfile", outfile, "--transcript", "youtube"]
        env = {k: v for k, v in os.environ.items() if k != "YOUTUBE_DATA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            _run_job_sync(job, args)

    assert job.status == "error"
    assert "YOUTUBE_DATA_API_KEY" in "\n".join(job.output)


def test_missing_youtube_data_api_key_channel_handle_errors():
    """When YOUTUBE_DATA_API_KEY is absent and input is a channel handle,
    the job should end in error state with a clear message."""
    with tempfile.TemporaryDirectory() as tmp:
        outfile = os.path.join(tmp, "out.csv")
        job = Job(id="ytkey3", video_id="@testchannel", output_file=outfile)
        args = ["@testchannel", "--outfile", outfile, "--transcript", "youtube"]
        env = {k: v for k, v in os.environ.items() if k != "YOUTUBE_DATA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            _run_job_sync(job, args)

    assert job.status == "error"
    assert "YOUTUBE_DATA_API_KEY" in "\n".join(job.output)


@patch("youtube_to_docs.main.get_youtube_service")
@patch("youtube_to_docs.main.resolve_video_ids", return_value=["abc12345678"])
@patch("youtube_to_docs.main.get_video_details", return_value=_VIDEO_DETAILS)
@patch(
    "youtube_to_docs.main.fetch_transcript",
    return_value=("Test transcript text", False, ""),
)
def test_no_model_transcript_only_job_completes(
    _mock_transcript, _mock_details, _mock_resolve, _mock_svc
):
    """When no model is specified the job should complete without requiring
    any AI API credentials."""
    with tempfile.TemporaryDirectory() as tmp:
        outfile = os.path.join(tmp, "out.csv")
        job = Job(id="nomodel", video_id="abc12345678", output_file=outfile)
        args = [
            "abc12345678",
            "--outfile",
            outfile,
            "--transcript",
            "youtube",
        ]
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "GEMINI_API_KEY",
                "AWS_BEARER_TOKEN_BEDROCK",
                "AZURE_FOUNDRY_ENDPOINT",
                "AZURE_FOUNDRY_API_KEY",
            )
        }
        with patch.dict(os.environ, env, clear=True):
            _run_job_sync(job, args)

    assert job.status == "completed"
