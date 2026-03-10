import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from youtube_to_docs.app import (
    Job,
    _extract_video_id,
    _scan_artifacts,
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
    resp = client.get("/api/artifacts/../../etc/passwd")
    assert resp.status_code in (403, 404)


def test_artifact_not_found(client):
    resp = client.get("/api/artifacts/youtube-to-docs-artifacts/nonexistent.csv")
    assert resp.status_code in (403, 404)


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
    result = _scan_artifacts("abc12345678")
    assert result == []


def test_scan_artifacts_finds_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_dir = tmp_path / "summary-files"
    summary_dir.mkdir()
    (summary_dir / "gemini - abc12345678 - summary.md").write_text("test")

    result = _scan_artifacts("abc12345678")
    assert len(result) == 1
    assert result[0]["name"] == "gemini - abc12345678 - summary.md"
    assert result[0]["directory"] == "summary-files"


def test_scan_artifacts_ignores_unrelated_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_dir = tmp_path / "summary-files"
    summary_dir.mkdir()
    (summary_dir / "gemini - other_id - summary.md").write_text("test")

    result = _scan_artifacts("abc12345678")
    assert result == []
