"""
Audio description (`-ad`): describes the visual information that people who are
blind or have low vision would miss (slides, on-screen text, chart figures,
votes, video clips), for WCAG 2.1 Level AA (1.2.5 Audio Description) and
Section 508, following the DCMP Description Key.

1. Moments: scene changes in the downloaded video (ffmpeg), plus on-screen
   events noted in the video context.
2. Descriptions: a Gemini model looks at the frame at each moment, with what is
   said around it, and describes only what isn't said aloud.
3. Placement: standard audio description is spoken only in pauses, so each
   description goes in a pause while its visual is still on screen, most
   important first, shortened if needed. Nothing is narrated over speech.
4. Outputs: WebVTT descriptions tracks for websites (standard, in pauses, and
   extended, for players that pause the video while reading), a descriptive
   transcript with every description, and narration (TTS) of the standard
   descriptions for the captioned, audio-described video.
"""

import concurrent.futures
import html
import math
import os
import re
import subprocess
import tempfile
import time
import wave
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rich import print as rprint

from youtube_to_docs.captions import (
    CaptionFiles,
    Cue,
    _clock,
    _ends_sentence,
    format_timestamp,
    parse_srt,
    parse_video_context,
    prepare_cues,
)
from youtube_to_docs.llms import get_model_pricing
from youtube_to_docs.utils import format_clickable_path, normalize_model_name

DEFAULT_DESCRIPTION_MODEL = "gemini-4-argon"
DEFAULT_NARRATION_TTS = "gemini-3.8-flash-tts-Kore"

# A scene-change score above this (0-1, on a small copy of each frame) is a
# new slide or shot.
SCENE_THRESHOLD = 0.3
# Changes closer together than this (slide builds, transitions) count as one.
MIN_SCENE_SECONDS = 4.0
MAX_MOMENTS = 400
FRAME_WIDTH = 1024
FRAMES_PER_REQUEST = 8
LLM_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 15
DEFAULT_WORKERS = 6

# Speech this close to a pause's edges is left alone.
SPEECH_MARGIN_SECONDS = 0.25
# The second transcript's lines only count as speech this far inside a pause.
REFERENCE_EDGE_SECONDS = 0.75
MIN_PAUSE_SECONDS = 1.0
# How long after a visual appears its description may still be spoken, by
# priority (1 = needed to follow the video, 2 = helpful detail). It is never
# spoken after the next visual change. Dense dialogue leaves few pauses, so
# essential descriptions may wait for one while their visual is on screen.
MAX_DELAY_SECONDS = {1: 120.0, 2: 10.0}
NARRATION_WORDS_PER_SECOND = 2.6
NARRATION_GAP_SECONDS = 0.15


@dataclass
class Moment:
    """A visual change worth describing: its time, when the next change comes
    (after which its description is stale) and any on-screen event noted."""

    time: float
    until: float
    hint: str = ""


@dataclass
class Description:
    time: float
    until: float
    priority: int
    text: str
    spoken: str = ""
    start: float = -1.0
    duration: float = 0.0
    audio: bytes = b""
    rate: int = 0

    @property
    def placed(self) -> bool:
        return self.start >= 0


# ---------------------------------------------------------------------------
# Visual moments and frames
# ---------------------------------------------------------------------------


def _ffmpeg() -> str:
    try:
        from static_ffmpeg import run
    except ImportError as e:
        raise ImportError(
            "Missing dependencies for audio/video processing. "
            'Please run with: uvx "youtube-to-docs[all]"'
        ) from e
    ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
    return ffmpeg_path


def detect_scene_changes(video_path: str) -> List[float]:
    """Times (seconds) where the picture changes, e.g. a new slide."""
    command = [
        _ffmpeg(),
        "-hide_banner",
        "-nostats",
        "-i",
        video_path,
        "-an",
        "-vf",
        f"scale=320:-2,select='gt(scene,{SCENE_THRESHOLD})',showinfo",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error detecting scene changes: {result.stderr.strip()[-300:]}")
        return []
    return [float(t) for t in re.findall(r"pts_time:([\d.]+)", result.stderr)]


def find_moments(
    scene_times: Sequence[float],
    events: Sequence[Tuple[float, str]],
    duration: float,
) -> List[Moment]:
    """
    Visual moments to describe: the start of the video, scene changes (merging
    bursts such as slide builds) and on-screen events from the video context,
    which also mark changes inside one shot, such as a vote.
    """
    scenes: List[float] = [0.0]
    for t in sorted(scene_times):
        if t - scenes[-1] >= MIN_SCENE_SECONDS:
            scenes.append(t)
    standalone = sorted(
        t for t, _ in events if min(abs(s - t) for s in scenes) > MIN_SCENE_SECONDS
    )
    room = max(1, MAX_MOMENTS - len(standalone))
    if len(scenes) > room:
        step = len(scenes) / room
        scenes = [scenes[int(i * step)] for i in range(room)]
    times = sorted(scenes + standalone)
    hints: Dict[int, List[str]] = {}
    for t, text in events:
        k = min(range(len(times)), key=lambda i: abs(times[i] - t))
        if abs(times[k] - t) <= MIN_SCENE_SECONDS:
            hints.setdefault(k, []).append(text)
    end = max(duration, times[-1] + 1)
    return [
        Moment(
            t,
            times[i + 1] if i + 1 < len(times) else end,
            "; ".join(hints.get(i, [])),
        )
        for i, t in enumerate(times)
    ]


def extract_frames(
    video_path: str, moments: Sequence[Moment], out_dir: str
) -> List[str]:
    """JPEG frames for each moment, a second after the change so any
    transition has finished (but before the next change)."""
    ffmpeg = _ffmpeg()
    paths = []
    for index, moment in enumerate(moments):
        t = min(moment.time + 1.0, max(moment.time, moment.until - 0.2))
        path = os.path.join(out_dir, f"frame_{index:04d}.jpg")
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{t:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-vf",
                f"scale={FRAME_WIDTH}:-2",
                "-q:v",
                "4",
                path,
            ],
            check=False,
            # Seeking into some H.264 streams makes the decoder log harmless
            # errors (e.g. "mmco: unref short failure"); a missing frame is
            # handled below.
            capture_output=True,
        )
        paths.append(path if os.path.exists(path) else "")
    return paths


# ---------------------------------------------------------------------------
# Descriptions (Gemini, frames + dialogue)
# ---------------------------------------------------------------------------

_DESCRIBE_RULES = """\
## Task
For each frame, describe the visual information a listener would miss: new \
slide titles and key on-screen text, the main figures or trend in a chart or \
table, a vote shown by raised hands, a video clip starting, a document or \
website shown, or the name caption of someone new. Leave out what the speakers \
say aloud, anything already described for an earlier frame, decoration and \
camera framing.

Write like a professional describer (DCMP Description Key):
- Present tense and objective, with no interpretation or opinion ("A bar chart \
shows...", not "We see...").
- Most important information first.
- Quote short titles exactly, in normal capitalization rather than all \
capitals; summarize long text and give only the key figures of charts.
- Priority 1 if it's needed to follow the video (a new slide or data, a vote, a \
clip); priority 2 for helpful detail.
- If nothing important is new, output NONE.

## Output
One line per frame: `frame number|priority|short|full`, or `frame number|NONE`.
- short: what a narrator says in a brief pause in the dialogue: at most 12 \
words, just the key point (e.g. the slide title).
- full: the complete description, one or two sentences, at most 30 words.
Output nothing else."""

_DESCRIPTION_RE = re.compile(
    r"^\s*#?(\d+)\s*\|\s*(?:(NONE)|([12])\s*\|\s*([^|]+?)\s*(?:\|\s*(.+?))?)\s*$"
)


def _dialogue_near(
    captions: Sequence[Cue], t: float, before: float, after: float
) -> str:
    starts = [cue.start for cue in captions]
    lo = bisect_left(starts, t - before)
    hi = bisect_left(starts, t + after)
    return " ".join(cue.text for cue in captions[lo:hi])


def _ask_with_images(
    model_name: str, parts: List[Any], valid: Callable[[str], bool]
) -> Tuple[str, int, int]:
    """Sends text and image parts to a Gemini model, retrying errors and
    unusable answers. Returns (text, input_tokens, output_tokens)."""
    from google import genai
    from google.genai import types

    if not os.environ.get("GEMINI_API_KEY"):
        return "Error: GEMINI_API_KEY not found", 0, 0
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    input_tokens = output_tokens = 0
    text = ""
    for attempt in range(LLM_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[types.Content(role="user", parts=parts)],
            )
            text = response.text or ""
            usage = response.usage_metadata
            if usage:
                input_tokens += usage.prompt_token_count or 0
                output_tokens += usage.candidates_token_count or 0
                thoughts = usage.thoughts_token_count
                if isinstance(thoughts, int):
                    output_tokens += thoughts
        except Exception as e:
            text = f"Error: {e}"
        if not text.startswith("Error") and valid(text):
            break
        if attempt < LLM_ATTEMPTS - 1:
            time.sleep(RETRY_WAIT_SECONDS * (attempt + 1))
    return text, input_tokens, output_tokens


def speakable(text: str) -> str:
    """Text as a narrator should read it: words in all capitals (slide titles)
    become title case, since TTS reads them slowly or letter by letter.
    Short acronyms such as AIB or MCAP stay as they are."""
    return re.sub(r"\b[A-Z][A-Z'’-]{4,}\b", lambda m: m.group(0).capitalize(), text)


def _workers() -> int:
    return max(1, int(os.environ.get("YTD_CC_WORKERS", DEFAULT_WORKERS)))


def describe_moments(
    model_name: str,
    moments: Sequence[Moment],
    frames: Sequence[str],
    captions: Sequence[Cue],
    video_title: str,
    people: str,
) -> Tuple[List[Description], int, int]:
    """
    Describes the frames in parallel batches. Each batch also gets the frame
    before it, so a slide carried over from the previous batch isn't
    described twice. Returns (descriptions, input_tokens, output_tokens).
    """
    from google.genai import types

    intro = (
        "You are writing audio description for people who are blind or have low "
        "vision (WCAG 2.1 Level AA 1.2.5, Section 508). Below are frames taken "
        "where the picture changes in this video, in order, each with what is "
        "said around that time."
        + (f"\n\nVideo: {video_title}" if video_title else "")
        + (f"\n\nPeople in the video:\n{people}" if people else "")
    )
    batches = [
        list(range(start, min(start + FRAMES_PER_REQUEST, len(moments))))
        for start in range(0, len(moments), FRAMES_PER_REQUEST)
    ]

    def frame_parts(index: int, label: str) -> List[Any]:
        moment = moments[index]
        said = _dialogue_near(captions, moment.time, 10, 20) or "(no speech)"
        note = f"\nNoted on screen around here: {moment.hint}" if moment.hint else ""
        parts = [
            types.Part.from_text(
                text=(
                    f"{label} — {_clock(moment.time)}{note}\n"
                    f"Said around this time: {said}"
                )
            )
        ]
        if frames[index]:
            with open(frames[index], "rb") as f:
                parts.append(
                    types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
                )
        return parts

    def run(batch: List[int]) -> Tuple[List[Description], int, int]:
        parts: List[Any] = [types.Part.from_text(text=intro)]
        if batch[0] > 0:
            parts += frame_parts(
                batch[0] - 1, "Previous frame (already described; don't output)"
            )
        for index in batch:
            parts += frame_parts(index, f"Frame {index + 1}")
        parts.append(types.Part.from_text(text=_DESCRIBE_RULES))
        wanted = {index + 1 for index in batch}

        def answers(text: str) -> Dict[int, Optional[Tuple[int, str, str]]]:
            found: Dict[int, Optional[Tuple[int, str, str]]] = {}
            for line in text.splitlines():
                match = _DESCRIPTION_RE.match(line)
                if match and int(match.group(1)) in wanted:
                    number = int(match.group(1))
                    short = match.group(4) or ""
                    found[number] = (
                        None
                        if match.group(2)
                        else (int(match.group(3)), short, match.group(5) or short)
                    )
            return found

        text, in_tok, out_tok = _ask_with_images(
            model_name, parts, lambda t: len(answers(t)) >= 0.6 * len(batch)
        )
        described = []
        for number, answer in sorted(answers(text).items()):
            if answer:
                moment = moments[number - 1]
                priority, short, full = answer
                described.append(
                    Description(
                        moment.time,
                        moment.until,
                        priority,
                        full.strip(),
                        spoken=speakable(short.strip()),
                    )
                )
        return described, in_tok, out_tok

    descriptions: List[Description] = []
    input_tokens = output_tokens = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=_workers()) as pool:
        for found, in_tok, out_tok in pool.map(run, batches):
            descriptions.extend(found)
            input_tokens += in_tok
            output_tokens += out_tok
    descriptions.sort(key=lambda d: d.time)
    return descriptions, input_tokens, output_tokens


def shorten_descriptions(
    model_name: str, items: Sequence[Tuple[Description, int]]
) -> Tuple[int, int]:
    """Shortens descriptions to fit their pauses (`spoken`), keeping the most
    important information. Returns (input_tokens, output_tokens)."""
    from google.genai import types

    if not items:
        return 0, 0
    lines = "\n".join(f"{k + 1}|{words}|{d.text}" for k, (d, words) in enumerate(items))
    prompt = (
        "Shorten each audio description to at most the given number of words, "
        "keeping the most important information (titles and key figures first) "
        "and the present-tense, objective style. Input lines are "
        "`id|max words|description`; output `id|shortened description`, one per "
        "line, nothing else.\n\n" + lines
    )
    found: Dict[int, str] = {}

    def valid(text: str) -> bool:
        found.clear()
        for line in text.splitlines():
            match = re.match(r"^\s*(\d+)\s*\|\s*(.+?)\s*$", line)
            if match and 1 <= int(match.group(1)) <= len(items):
                found[int(match.group(1))] = match.group(2)
        return len(found) >= 0.6 * len(items)

    _, in_tok, out_tok = _ask_with_images(
        model_name, [types.Part.from_text(text=prompt)], valid
    )
    for k, (description, words) in enumerate(items):
        short = found.get(k + 1, "")
        if short and len(short.split()) <= words + 2:
            description.spoken = short
    return in_tok, out_tok


# ---------------------------------------------------------------------------
# Pauses and placement
# ---------------------------------------------------------------------------


def speech_pauses(
    source: Sequence[Cue], reference: Sequence[Cue], duration: float
) -> List[Tuple[float, float]]:
    """
    Pauses in the speech, from the timed transcript's cues. A line starting in
    the second transcript well inside a pause is speech the first one missed;
    near a pause's edges it is just timing jitter between the two.
    """
    merged: List[List[float]] = []
    for start, end in sorted((cue.start, cue.end) for cue in source):
        start, end = start - SPEECH_MARGIN_SECONDS, end + SPEECH_MARGIN_SECONDS
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = []
    previous_end = 0.0
    for start, end in merged:
        gaps.append((previous_end, start))
        previous_end = max(previous_end, end)
    gaps.append((previous_end, duration))

    onsets = sorted(cue.start for cue in reference)
    pauses = []
    for start, end in gaps:
        cursor = start
        lo = bisect_left(onsets, start + REFERENCE_EDGE_SECONDS)
        hi = bisect_left(onsets, end - REFERENCE_EDGE_SECONDS)
        for onset in onsets[lo:hi]:
            if onset - SPEECH_MARGIN_SECONDS - cursor >= MIN_PAUSE_SECONDS:
                pauses.append((cursor, onset - SPEECH_MARGIN_SECONDS))
            cursor = max(cursor, onset + 1.0 + SPEECH_MARGIN_SECONDS)
        if end - cursor >= MIN_PAUSE_SECONDS:
            pauses.append((cursor, end))
    return pauses


def _room(description: Description, pauses, cursors) -> List[Tuple[int, float, float]]:
    """(pause index, earliest start, free seconds) for each pause where the
    description could still be spoken."""
    latest = min(
        description.until, description.time + MAX_DELAY_SECONDS[description.priority]
    )
    ends = [end for _, end in pauses]
    options = []
    for k in range(bisect_left(ends, description.time), len(pauses)):
        start, end = pauses[k]
        if start > latest:
            break
        begin = max(cursors[k], start, description.time - 0.5)
        if end - begin > 0:
            options.append((k, begin, end - begin))
    return options


def place_descriptions(
    descriptions: Sequence[Description], pauses: Sequence[Tuple[float, float]]
) -> None:
    """
    Puts each description (with its narration `duration`) in the earliest
    pause that starts after its visual appears and before it's replaced, with
    room for it, most important first. Unplaced ones keep `start` < 0.
    """
    cursors = [start for start, _ in pauses]
    for description in sorted(descriptions, key=lambda d: (d.priority, d.time)):
        if description.placed:
            continue
        for k, begin, free in _room(description, pauses, cursors):
            if description.duration <= free:
                description.start = begin
                cursors[k] = begin + description.duration + NARRATION_GAP_SECONDS
                break


def _free_cursors(descriptions: Sequence[Description], pauses) -> List[float]:
    cursors = [start for start, _ in pauses]
    for d in descriptions:
        if d.placed:
            for k, (start, end) in enumerate(pauses):
                if start <= d.start < end:
                    cursors[k] = max(
                        cursors[k], d.start + d.duration + NARRATION_GAP_SECONDS
                    )
    return cursors


# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------


def _estimated_seconds(text: str) -> float:
    return len(text.split()) / NARRATION_WORDS_PER_SECOND + 0.3


def synthesize(descriptions: Sequence[Description], tts_arg: str) -> None:
    """
    Narrates each description's `spoken` text, setting `audio`, `rate` and
    `duration`. Where speech fails, `duration` is an estimate, so the text
    track can still be timed.
    """
    for description in descriptions:
        description.audio, description.rate = b"", 0
        description.duration = _estimated_seconds(description.spoken)
    from youtube_to_docs.providers import TTSProvider, get_provider
    from youtube_to_docs.tts import parse_tts_arg

    model_name, voice = parse_tts_arg(tts_arg)
    try:
        provider = get_provider(model_name)
    except ValueError as e:
        print(f"Error: {e}")
        return
    if not isinstance(provider, TTSProvider):
        print(f"Error: {model_name} does not support text-to-speech.")
        return

    def run(description: Description) -> None:
        for attempt in range(LLM_ATTEMPTS):
            pcm, rate = provider.generate_speech(description.spoken, voice)
            if pcm and rate:
                description.audio, description.rate = pcm, rate
                description.duration = len(pcm) / 2 / rate
                return
            if attempt < LLM_ATTEMPTS - 1:
                time.sleep(RETRY_WAIT_SECONDS * (attempt + 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=_workers()) as pool:
        list(pool.map(run, descriptions))


def write_narration_track(
    descriptions: Sequence[Description], duration: float, path: str
) -> bool:
    """Writes a mono WAV the length of the video with each placed narration
    at its start time and silence elsewhere."""
    placed = sorted(
        (d for d in descriptions if d.placed and d.audio), key=lambda d: d.start
    )
    if not placed:
        return False
    rate = placed[0].rate
    total = int(math.ceil(max(duration, placed[-1].start + placed[-1].duration) * rate))
    chunk = b"\x00\x00" * rate  # one second of silence
    with wave.open(path, "wb") as track:
        track.setnchannels(1)
        track.setsampwidth(2)
        track.setframerate(rate)
        written = 0
        for d in placed:
            if d.rate != rate:
                continue  # a different TTS rate can't share the track
            offset = max(written, int(d.start * rate))
            while written < offset:
                n = min(rate, offset - written)
                track.writeframes(chunk[: n * 2])
                written += n
            track.writeframes(d.audio)
            written += len(d.audio) // 2
        while written < total:
            n = min(rate, total - written)
            track.writeframes(chunk[: n * 2])
            written += n
    return True


# ---------------------------------------------------------------------------
# Text outputs
# ---------------------------------------------------------------------------


def descriptions_webvtt(descriptions: Sequence[Description]) -> str:
    """WebVTT descriptions track (`<track kind="descriptions">`) with each
    placed description at the time it's narrated."""
    cues = []
    for d in sorted((d for d in descriptions if d.placed), key=lambda d: d.start):
        start = format_timestamp(d.start).replace(",", ".")
        end = format_timestamp(d.start + d.duration).replace(",", ".")
        text = d.spoken.replace("&", "&amp;").replace("<", "&lt;")
        cues.append(f"{start} --> {end}\n{text}")
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


def extended_descriptions_webvtt(descriptions: Sequence[Description]) -> str:
    """
    WebVTT track with every full description at the moment its visual
    appears, for players that pause the video while a description is read
    (extended audio description, WCAG 1.2.7), such as Able Player.
    """
    ordered = sorted(descriptions, key=lambda d: d.time)
    cues = []
    for k, d in enumerate(ordered):
        end = d.time + max(1.0, len(d.text.split()) / NARRATION_WORDS_PER_SECOND)
        if k + 1 < len(ordered):
            end = min(end, max(ordered[k + 1].time - 0.05, d.time + 0.5))
        text = d.text.replace("&", "&amp;").replace("<", "&lt;")
        cues.append(
            f"{format_timestamp(d.time).replace(',', '.')} --> "
            f"{format_timestamp(end).replace(',', '.')}\n{text}"
        )
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


_LABEL_RE = re.compile(r"^\[([A-Z][^\]]*)\]\s*")


def descriptive_transcript(
    captions_srt: str, descriptions: Sequence[Description], video_title: str
) -> str:
    """
    Markdown transcript of the dialogue (by speaker, from the corrected
    captions) with every description inserted where its visual appears.
    """
    turns: List[Tuple[float, str, str]] = []  # (time, speaker, text) per line
    speaker = ""
    text = captions_srt.replace("\r\n", "\n").strip()
    for block in re.split(r"\n[ \t]*\n", text):
        cues = parse_srt(block)
        if not cues:
            continue
        lines = block.split("\n")
        timing = next(i for i, line in enumerate(lines) if "-->" in line)
        for line in lines[timing + 1 :]:
            match = _LABEL_RE.match(line)
            if match:
                speaker, line = match.group(1), line[match.end() :]
            if line.strip():
                turns.append((cues[0].start, speaker, line.strip()))

    out = [
        f"# {video_title or 'Video'}: descriptive transcript",
        "",
        "The spoken dialogue, by speaker, with descriptions of visual information "
        "(slides, charts, on-screen text) where it appears.",
    ]
    pending = sorted(descriptions, key=lambda d: d.time)
    k = 0
    paragraph: List[str] = []
    current: Optional[str] = None
    started = 0.0

    def flush() -> None:
        if paragraph:
            name = current or "Speaker"
            out.extend(["", f"**[{_clock(started)}] {name}:** " + " ".join(paragraph)])
            paragraph.clear()

    for t, name, text in turns:
        # Descriptions go in where their visual appears, but wait (up to 10
        # seconds) for the sentence in progress to end.
        mid_sentence = bool(paragraph) and not _ends_sentence(
            paragraph[-1].split()[-1], text.split()[0] if text.split() else None
        )
        if (
            k < len(pending)
            and pending[k].time <= t
            and (not mid_sentence or name != current or t - pending[k].time > 10)
        ):
            flush()
            while k < len(pending) and pending[k].time <= t:
                out.extend(
                    [
                        "",
                        f"*[{_clock(pending[k].time)}] Description: {pending[k].text}*",
                    ]
                )
                k += 1
            current = None
        if name != current:
            flush()
            current, started = name, t
        paragraph.append(text)
    flush()
    for d in pending[k:]:
        out.extend(["", f"*[{_clock(d.time)}] Description: {d.text}*"])
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI step
# ---------------------------------------------------------------------------


def process_audio_description(
    *,
    row: Dict[str, Any],
    storage: Any,
    model_name: str,
    captions: CaptionFiles,
    get_video: Callable[[], Optional[str]],
    tts_arg: Optional[str],
    video_id: str,
    video_title: str,
    safe_title: str,
    dirs: Dict[str, str],
    duration_seconds: Optional[float],
    verbose: bool,
    vprint: Callable[..., None] = print,
) -> Optional[List[Description]]:
    """
    Writes the audio description for a video (part of `-cc`): WebVTT
    descriptions tracks (standard, fitted into pauses, and extended) and a
    descriptive transcript. Returns the standard descriptions with their
    narration for the audio-described video, or None if they weren't
    generated in this run (e.g. the outputs already exist).
    """
    if not model_name.startswith("gemini-"):
        print(f"Audio description needs a Gemini model; skipping {model_name}.")
        return None
    stem = f"{model_name} - {video_id} - {safe_title}"
    outputs = {
        "vtt": (
            f"Descriptions VTT File {model_name}",
            os.path.join(dirs["vtt"], f"{stem} - descriptions.vtt").replace("\\", "/"),
        ),
        "extended": (
            f"Extended Descriptions VTT File {model_name}",
            os.path.join(dirs["vtt"], f"{stem} - extended descriptions.vtt").replace(
                "\\", "/"
            ),
        ),
        "transcript": (
            f"Descriptive Transcript File {model_name}",
            os.path.join(
                dirs["descriptive_transcripts"], f"{stem} - descriptive transcript.md"
            ).replace("\\", "/"),
        ),
    }
    missing = {}
    for key, (col, path) in outputs.items():
        if storage.exists(path):
            row[col] = row.get(col) or path
        else:
            missing[key] = (col, path)
    if not missing:
        vprint(f"Audio description already exists for {video_id}.")
        return None

    video = get_video()
    if not video:
        return None
    context = parse_video_context(
        next(
            (
                value
                for key, value in row.items()
                if key.startswith("Video Context ")
                and not key.startswith("Video Context File ")
                and isinstance(value, str)
                and value.strip()
            ),
            "",
        )
    )
    captions_srt = storage.read_text(captions.path)
    caption_cues = parse_srt(captions_srt)
    duration = max(
        duration_seconds or 0.0, caption_cues[-1].end if caption_cues else 0.0
    )
    rprint("Finding visual changes for audio description...")
    scenes = detect_scene_changes(video)
    duration = max(duration, scenes[-1] if scenes else 0.0)
    moments = find_moments(scenes, context.events, duration)
    people = "\n".join(
        f"- {p.name}" + (f" ({p.title})" if p.title else "") for p in context.people
    )

    with tempfile.TemporaryDirectory() as work:
        rprint(f"Describing {len(moments)} visual moments using {model_name}...")
        frames = extract_frames(video, moments, work)
        descriptions, input_tokens, output_tokens = describe_moments(
            model_name, moments, frames, caption_cues, video_title, people
        )
    if not descriptions:
        print("No visual information needing description was found.")
        return None
    for d in descriptions:
        d.spoken = d.spoken or speakable(d.text)

    # Fit the short versions into pauses, shortening what doesn't fit once.
    source = prepare_cues(parse_srt(storage.read_text(captions.source_srt)))
    reference = (
        prepare_cues(parse_srt(storage.read_text(captions.reference_srt)))
        if captions.reference_srt
        else []
    )
    pauses = speech_pauses(source, reference, duration)
    narration = tts_arg or DEFAULT_NARRATION_TTS
    rprint(f"Narrating {len(descriptions)} descriptions using {narration}...")
    synthesize(descriptions, narration)
    place_descriptions(descriptions, pauses)
    cursors = _free_cursors(descriptions, pauses)
    to_shorten = []
    for d in descriptions:
        if d.placed:
            continue
        free = max((room for _, _, room in _room(d, pauses, cursors)), default=0.0)
        words = int((free - 0.3) * NARRATION_WORDS_PER_SECOND)
        if words >= 3:
            to_shorten.append((d, words))
    before = {id(d): d.spoken for d, _ in to_shorten}
    in_tok, out_tok = shorten_descriptions(model_name, to_shorten)
    input_tokens += in_tok
    output_tokens += out_tok
    shortened = [d for d, _ in to_shorten if d.spoken != before[id(d)]]
    synthesize(shortened, narration)
    place_descriptions(shortened, pauses_with_room(pauses, descriptions))
    placed = [d for d in descriptions if d.placed]
    rprint(
        f"Placed {len(placed)} of {len(descriptions)} descriptions in pauses "
        "(standard track and narration); all are in the extended track and "
        "the descriptive transcript."
    )

    texts = {
        "vtt": (descriptions_webvtt(descriptions), "descriptions WebVTT"),
        "extended": (
            extended_descriptions_webvtt(descriptions),
            "extended descriptions WebVTT",
        ),
        "transcript": (
            descriptive_transcript(captions_srt, descriptions, video_title),
            "descriptive transcript",
        ),
    }
    for key, (col, path) in missing.items():
        _save_text(row, storage, col, path, *texts[key])

    if verbose:
        input_price, output_price = get_model_pricing(model_name)
        if input_price is not None and output_price is not None:
            cost = (input_tokens / 1_000_000) * input_price + (
                output_tokens / 1_000_000
            ) * output_price
            row[f"{normalize_model_name(model_name)} audio description cost ($)"] = (
                round(cost, 2)
            )
            vprint(f"Audio description cost (excluding narration): ${cost:.2f}")
    return placed


def descriptions_from_webvtt(text: str) -> List[Description]:
    """Standard descriptions read back from a saved descriptions track, so the
    audio-described video can be made without describing the video again."""
    found = []
    for cue in parse_srt(text):
        if cue.text:
            line = html.unescape(cue.text)
            d = Description(cue.start, cue.end, 1, line, spoken=line)
            d.start, d.duration = cue.start, cue.end - cue.start
            found.append(d)
    return found


def pauses_with_room(
    pauses: Sequence[Tuple[float, float]], descriptions: Sequence[Description]
) -> List[Tuple[float, float]]:
    """The parts of each pause not yet taken by placed narration."""
    taken = sorted(
        (d.start, d.start + d.duration + NARRATION_GAP_SECONDS)
        for d in descriptions
        if d.placed
    )
    free = []
    for start, end in pauses:
        cursor = start
        for a, b in taken:
            if b <= cursor or a >= end:
                continue
            if a - cursor >= MIN_PAUSE_SECONDS:
                free.append((cursor, a))
            cursor = max(cursor, b)
        if end - cursor >= MIN_PAUSE_SECONDS:
            free.append((cursor, end))
    return free


def _save_text(row, storage, col: str, path: str, text: str, kind: str) -> None:
    try:
        saved = storage.write_text(path, text)
    except Exception as e:
        print(f"Error writing {kind}: {e}")
        return
    row[col] = saved
    rprint(f"Saved {kind}: {format_clickable_path(saved)}")
