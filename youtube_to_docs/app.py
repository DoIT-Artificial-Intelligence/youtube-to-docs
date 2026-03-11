import asyncio
import contextlib
import io
import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from youtube_to_docs.main import main as app_main
from youtube_to_docs.models import MODEL_SUITES

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
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


# Remote/special output_file values that are not local paths
REMOTE_OUTPUT_VALUES = {"workspace", "w", "sharepoint", "s", "none", "n"}


jobs: dict[str, Job] = {}

# Known artifact directories to scan
ARTIFACT_DIRS = [
    "youtube-to-docs-artifacts",
    "summary-files",
    "transcript-files",
    "audio-files",
    "infographic-files",
    "speaker-extraction-files",
    "qa-files",
    "video-files",
    "srt-files",
    "tag-files",
    "one-sentence-summary-files",
    "infographic-alt-text",
    "suggested-corrected-caption-files",
]


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    url: str
    output_file: str = "youtube-to-docs-artifacts/youtube-docs.csv"
    transcript_source: str = "youtube"
    model: str | None = None
    tts_model: str | None = None
    infographic_model: str | None = None
    alt_text_model: str | None = None
    no_youtube_summary: bool = False
    translate: str | None = None
    combine_infographic_audio: bool = False
    all_suite: str | None = None
    suggest_corrected_captions: str | None = None
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


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(html_file.read_text())


@app.get("/api/model-suites")
async def model_suites():
    return MODEL_SUITES


def _safe_resolve_within_cwd(user_path: str) -> str:
    """Resolve a user-provided path and ensure it stays within cwd.

    Returns the resolved absolute path string. Raises HTTPException if the
    path escapes the working directory.

    Uses os.path.realpath + startswith, which is the pattern recognised by
    CodeQL as a path-traversal sanitizer.
    """
    cwd = os.path.realpath(os.getcwd())
    resolved = os.path.realpath(os.path.join(cwd, user_path))
    # Ensure the resolved path is within cwd (cwd itself or a child)
    if not (resolved == cwd or resolved.startswith(cwd + os.sep)):
        raise HTTPException(
            status_code=400,
            detail="Invalid path: must be within the working directory",
        )
    return resolved


def _validate_output_file(output_file: str) -> str:
    """Validate and sanitize the output_file parameter to prevent path traversal."""
    # Allow known remote/special values as-is
    if output_file.lower() in REMOTE_OUTPUT_VALUES:
        return output_file

    resolved = _safe_resolve_within_cwd(output_file)
    return os.path.relpath(resolved, os.getcwd())


@app.post("/api/process")
async def process(req: ProcessRequest):
    try:
        job_id = uuid.uuid4().hex[:12]
        validated_output = _validate_output_file(req.output_file)
        job = Job(id=job_id, video_id=req.url, output_file=validated_output)
        jobs[job_id] = job

        # Build args exactly like mcp_server.py
        args = [
            req.url,
            "--outfile",
            validated_output,
            "--transcript",
            req.transcript_source,
        ]

        if req.translate:
            args.extend(["--translate", req.translate])
        if req.model:
            args.extend(["--model", req.model])
        if req.tts_model:
            args.extend(["--tts", req.tts_model])
        if req.infographic_model:
            args.extend(["--infographic", req.infographic_model])
        if req.alt_text_model:
            args.extend(["--alt-text-model", req.alt_text_model])
        if req.no_youtube_summary:
            args.append("--no-youtube-summary")
        if req.combine_infographic_audio:
            args.append("--combine-infographic-audio")
        if req.all_suite:
            args.extend(["--all", req.all_suite])
        if req.suggest_corrected_captions:
            args.extend(
                ["--suggest-corrected-captions", req.suggest_corrected_captions]
            )
        if req.post_process:
            args.extend(["--post-process", req.post_process])
        if req.verbose:
            args.append("--verbose")

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
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            app_main(args)

    try:
        await asyncio.to_thread(_run)
        capture.flush()
        job.status = "completed"
    except Exception as e:
        capture.flush()
        job.error = str(e)
        job.status = "error"
    finally:
        job.finished_at = time.time()


def _scan_artifacts(video_id: str, output_file: str) -> list[dict]:
    """Scan known artifact directories for files matching the video ID."""
    artifacts = []
    for dir_name in ARTIFACT_DIRS:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            continue
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file() and video_id in file_path.name:
                artifacts.append(
                    {
                        "path": str(file_path),
                        "name": file_path.name,
                        "directory": dir_name,
                        "size": file_path.stat().st_size,
                    }
                )
    # Also check the output CSV (skip remote storage values)
    if output_file.lower() not in REMOTE_OUTPUT_VALUES:
        safe_csv = _safe_resolve_within_cwd(output_file)
        if os.path.isfile(safe_csv):
            csv_path = Path(safe_csv)
            artifacts.append(
                {
                    "path": os.path.relpath(safe_csv, os.getcwd()),
                    "name": csv_path.name,
                    "directory": str(
                        Path(os.path.relpath(safe_csv, os.getcwd())).parent
                    ),
                    "size": csv_path.stat().st_size,
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
        # Extract video ID from URL for artifact scanning
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
    # Sanitize: normalise the joined path and confirm it stays within cwd.
    # Uses os.path.normpath + startswith, the pattern CodeQL recognises as a
    # path-traversal sanitizer for py/path-injection.
    base_dir = os.path.normpath(os.getcwd())
    safe_path = os.path.normpath(os.path.join(base_dir, path))
    if not safe_path.startswith(base_dir + os.sep):
        raise HTTPException(status_code=400, detail="Access denied")

    if not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(safe_path)


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


def start_server():
    """Entry point for the youtube-to-docs-app command."""
    import argparse

    parser = argparse.ArgumentParser(description="YouTube to Docs Web App")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    if args.host == "0.0.0.0":
        print(f"App available at: http://localhost:{args.port}")

    uvicorn.run(
        "youtube_to_docs.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    start_server()
