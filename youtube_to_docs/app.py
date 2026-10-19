"""FastAPI web application for youtube-to-docs.

Usage::

    # Install with app extras
    uv sync --all-extras

    # Start the server (default: http://localhost:8000)
    uv run youtube-to-docs-app

    # Custom host/port with auto-reload for development
    uv run youtube-to-docs-app --port 9000 --reload
"""

import asyncio
import contextlib
import contextvars
import io
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from youtube_to_docs.main import main as app_main
from youtube_to_docs.models import MODEL_SUITES
from youtube_to_docs.storage import MemoryStorage
from youtube_to_docs.utils import (
    OUTPUT_ROOT,
    build_cli_args,
    is_remote_output,
    resolve_within,
    validate_output_file,
)

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
        Response,
        StreamingResponse,
    )
    from pydantic import BaseModel
    from starlette.staticfiles import StaticFiles
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed. "
        "Install with: pip install 'youtube-to-docs[app]'"
    ) from e


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

_MAX_JOBS = 200
_JOB_TTL_SECONDS = 3600


@dataclass
class Job:
    id: str
    status: str = "running"  # running | completed | error
    output: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    video_id: str = ""
    output_file: str = "youtube-to-docs-artifacts/youtube-docs.csv"
    storage: MemoryStorage | None = None


jobs: dict[str, Job] = {}


def _evict_old_jobs(
    max_jobs: int = _MAX_JOBS, ttl_seconds: float = _JOB_TTL_SECONDS
) -> None:
    """Evict expired or excess completed/errored jobs and release MemoryStorage."""
    now = time.time()
    finished = [
        (jid, j)
        for jid, j in list(jobs.items())
        if j.status in ("completed", "error") and j.finished_at is not None
    ]
    # Remove any finished jobs older than ttl_seconds
    for jid, j in finished:
        if j.finished_at is not None and (now - j.finished_at) > ttl_seconds:
            j.storage = None
            jobs.pop(jid, None)

    # If still over max_jobs, evict oldest finished jobs first
    if len(jobs) >= max_jobs:
        remaining_finished = sorted(
            ((jid, j) for jid, j in jobs.items() if j.status in ("completed", "error")),
            key=lambda item: item[1].finished_at or item[1].started_at,
        )
        while len(jobs) >= max_jobs and remaining_finished:
            jid, j = remaining_finished.pop(0)
            j.storage = None
            jobs.pop(jid, None)


# Context-isolated stdout/stderr routing for concurrent background jobs
_active_capture_var: contextvars.ContextVar[io.TextIOBase | None] = (
    contextvars.ContextVar("_active_capture_var", default=None)
)
_stream_lock = threading.Lock()
_active_captures: list[io.TextIOBase] = []
_orig_stdout: object = None
_orig_stderr: object = None


class _ContextStreamRouter(io.TextIOBase):
    """Routes writes to the active job's StreamCapture in the current context."""

    def __init__(self, fallback_getter):
        self._fallback_getter = fallback_getter

    def _target(self):
        target = _active_capture_var.get()
        if target is not None:
            return target
        with _stream_lock:
            if _active_captures:
                return _active_captures[-1]
        return self._fallback_getter()

    def write(self, s: str) -> int:
        target = self._target()
        if target is not None:
            return target.write(s)
        return len(s)

    def flush(self) -> None:
        target = self._target()
        if target is not None:
            target.flush()


@contextlib.contextmanager
def _capture_job_streams(capture: io.TextIOBase):
    """Redirect stdout/stderr for current context without clobbering jobs."""
    global _orig_stdout, _orig_stderr
    token = _active_capture_var.set(capture)
    with _stream_lock:
        if not _active_captures:
            _orig_stdout = sys.stdout
            _orig_stderr = sys.stderr
            sys.stdout = _ContextStreamRouter(lambda: _orig_stdout)  # type: ignore[assignment]
            sys.stderr = _ContextStreamRouter(lambda: _orig_stderr)  # type: ignore[assignment]
        _active_captures.append(capture)
    try:
        yield
    finally:
        with _stream_lock:
            if capture in _active_captures:
                _active_captures.remove(capture)
            if not _active_captures:
                if _orig_stdout is not None:
                    sys.stdout = _orig_stdout  # type: ignore[assignment]
                if _orig_stderr is not None:
                    sys.stderr = _orig_stderr  # type: ignore[assignment]
                _orig_stdout = None
                _orig_stderr = None
        _active_capture_var.reset(token)


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    url: str
    output_file: str = "youtube-to-docs-artifacts/youtube-docs.csv"
    hugging_face_dataset: str | None = None
    transcript_source: str | None = None
    model: str | None = None
    tts_model: str | None = None
    infographic_model: str | None = None
    alt_text_model: str | None = None
    no_youtube_summary: bool = False
    from_video: str | None = None
    translate: str | None = None
    combine_infographic_audio: bool = False
    all_suite: str | None = None
    corrected_captions: str | None = None
    post_process: str | None = None
    verbose: bool = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

logger = logging.getLogger("youtube_to_docs.app")

app = FastAPI(title="YouTube to Docs")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exception(exc)
    logger.error("Unhandled exception: %s\n%s", exc, "".join(tb))
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    svg = (STATIC_DIR / "favicon.svg").read_bytes()
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(html_file.read_text())


@app.get("/api/model-suites")
async def model_suites():
    return MODEL_SUITES


def _validate_output_file(output_file: str) -> str:
    """Validate output_file, confining local CSVs to OUTPUT_ROOT."""
    try:
        return validate_output_file(output_file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@app.post("/api/process")
async def process(req: ProcessRequest):
    try:
        _evict_old_jobs()
        job_id = uuid.uuid4().hex[:12]
        validated_output = _validate_output_file(req.output_file)
        job = Job(id=job_id, video_id=req.url, output_file=validated_output)
        jobs[job_id] = job

        args = build_cli_args(
            url=req.url,
            output_file=validated_output,
            hugging_face_dataset=req.hugging_face_dataset,
            transcript_source=req.transcript_source,
            model=req.model,
            tts_model=req.tts_model,
            infographic_model=req.infographic_model,
            alt_text_model=req.alt_text_model,
            no_youtube_summary=req.no_youtube_summary,
            from_video=req.from_video,
            translate=req.translate,
            combine_infographic_audio=req.combine_infographic_audio,
            all_suite=req.all_suite,
            corrected_captions=req.corrected_captions,
            post_process=req.post_process,
            verbose=req.verbose,
        )

        asyncio.create_task(_run_job(job, args))
        return {"job_id": job_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _run_job(job: Job, args: list[str]):
    """Run app_main in a background thread, capturing output line by line."""

    class StreamCapture(io.StringIO):
        def __init__(self, job: Job):
            super().__init__()
            self._job = job
            self._buffer = ""

        def write(self, s: str) -> int:
            result = super().write(s)
            self._buffer += s
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._job.output.append(line)
            return result

        def flush(self):
            super().flush()
            if self._buffer:
                self._job.output.append(self._buffer)
                self._buffer = ""

    capture = StreamCapture(job)

    def _run():
        with _capture_job_streams(capture):
            return app_main(args)

    try:
        result = await asyncio.to_thread(_run)
        capture.flush()
        job.status = "completed"
        if isinstance(result, MemoryStorage):
            job.storage = result
    except (Exception, SystemExit) as e:
        capture.flush()
        job.error = str(e)
        job.status = "error"
    finally:
        job.finished_at = time.time()


def _scan_artifacts(video_id: str, output_file: str) -> list[dict]:
    """Scan OUTPUT_ROOT for files matching the video ID. Only paths that
    get_artifact() will serve are returned."""
    artifacts = []
    root = Path(OUTPUT_ROOT)
    files = sorted(root.rglob("*")) if root.is_dir() else []
    if not is_remote_output(output_file):
        files.append(Path(output_file))
    seen = set()
    for file_path in files:
        is_csv = file_path == Path(output_file)
        if not file_path.is_file() or not (is_csv or video_id in file_path.name):
            continue
        try:
            resolve_within(OUTPUT_ROOT, os.path.abspath(file_path))
        except ValueError:
            continue
        if file_path.as_posix() in seen:
            continue
        seen.add(file_path.as_posix())
        artifacts.append(
            {
                "path": file_path.as_posix(),
                "name": file_path.name,
                "directory": file_path.parent.as_posix(),
                "size": file_path.stat().st_size,
            }
        )
    return artifacts


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result: dict[str, object] = {
        "id": job.id,
        "status": job.status,
        "output": job.output,
        "error": job.error,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }

    if job.status in ("completed", "error"):
        if job.storage is not None:
            result["artifacts"] = job.storage.get_artifacts()
            result["memory_mode"] = True
        else:
            vid = _extract_video_id(job.video_id)
            if vid:
                result["artifacts"] = _scan_artifacts(vid, job.output_file)

    return result


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        sent = 0
        while True:
            # Send any new lines
            while sent < len(job.output):
                line = job.output[sent]
                # SSE format: data lines, double newline to end event
                yield f"data: {line}\n\n"
                sent += 1

            if job.status != "running":
                # Send final status event
                if job.error:
                    yield f"event: error\ndata: {job.error}\n\n"
                else:
                    yield "event: done\ndata: completed\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/artifacts/{path:path}")
async def get_artifact(path: str):
    # Only serve files under OUTPUT_ROOT. realpath follows symlinks, so a link
    # inside OUTPUT_ROOT that points elsewhere is rejected too. The check is
    # inline (realpath + startswith) so CodeQL recognises it as a sanitizer.
    cwd = os.path.realpath(os.getcwd())
    root = os.path.realpath(os.path.join(cwd, OUTPUT_ROOT))
    safe_path = os.path.realpath(os.path.join(cwd, path))
    if not safe_path.startswith(root + os.sep):
        raise HTTPException(status_code=400, detail="Access denied")

    if not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(safe_path)


@app.get("/api/jobs/{job_id}/artifacts/{path:path}")
async def get_job_artifact(job_id: str, path: str):
    """Serve an artifact from a memory-mode job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.storage:
        raise HTTPException(
            status_code=404, detail="No in-memory artifacts for this job"
        )
    try:
        content, media_type = job.storage.serve_artifact(path)
        return Response(content=content, media_type=media_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifact not found")


def _extract_video_id(url: str) -> str | None:
    """Extract a YouTube video ID from a URL or return the string as-is if short."""
    import re

    # Handle youtube.com URLs
    m = re.search(r"(?:v=|youtu\.be/|/embed/|/v/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    # If it looks like a bare video ID
    if re.match(r"^[A-Za-z0-9_-]{11}$", url):
        return url
    # Comma-separated — take first
    if "," in url:
        first = url.split(",")[0].strip()
        return _extract_video_id(first)
    return None


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is already in use on the given host."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def start_server():
    """Entry point for the youtube-to-docs-app command."""
    import argparse

    from youtube_to_docs.utils import get_version

    parser = argparse.ArgumentParser(description="YouTube to Docs Web App")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to. Use 0.0.0.0 to expose on the network "
        "(the app has no authentication).",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    version = get_version()

    port = args.port
    # If the user didn't specify a port (it's the default 8000), or even if they did,
    # try to find the next available one if it's in use.
    original_port = port
    while _is_port_in_use(port, args.host):
        port += 1
        if port > original_port + 100:  # Safety limit
            print(
                f"Error: Could not find an available port after 100 attempts "
                f"starting from {original_port}."
            )
            return

    if port != args.port:
        print(f"Port {args.port} is in use, using port {port} instead.")

    if args.host in ("127.0.0.1", "localhost", "0.0.0.0"):
        print(f"YouTube to Docs v{version}")
        print(f"App available at: http://localhost:{port}")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"Warning: binding to {args.host} exposes the app to the network. "
            "It has no authentication; anyone who can reach it can run jobs "
            "and read generated artifacts."
        )

    uvicorn.run(
        "youtube_to_docs.app:app",
        host=args.host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    start_server()
