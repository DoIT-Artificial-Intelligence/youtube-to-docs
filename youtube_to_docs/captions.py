"""
Corrected captions (`-cc`): turns a speech-to-text SRT into a complete caption
file that meets WCAG 2.1 Level AA and Section 508
(https://www.section508.gov/create/captions-transcripts/).

1. Timing: the source SRT (by default `gemini-3.5-transcribe`, which has
   word-level timestamps) is aligned word by word with a second transcript of
   the same audio (usually YouTube's captions), and cues with broken timestamps
   are re-timed from it.
2. Speakers: the speaker of every cue is named in ~20-minute windows from
   diarization labels, the second transcript's speaker changes (`>>`), the
   times the video shows each person speaking (written onto the lines they
   apply to), on-screen names, introductions and hand-offs.
3. Text: sections are corrected in parallel against the second transcript and
   on-screen spellings. The model edits text by cue id and never writes
   timestamps, so long videos can't be truncated or drift out of sync.
4. Layout: the corrected words are re-segmented into readable captions (at most
   two lines of 42 characters, 1-7 seconds, broken at sentence and clause
   boundaries) with `[Name]` labels on speaker changes.
"""

import concurrent.futures
import difflib
import math
import os
import re
import statistics
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rich import print as rprint

from youtube_to_docs.llms import _query_llm, get_model_pricing
from youtube_to_docs.utils import format_clickable_path, normalize_model_name
from youtube_to_docs.video_understanding import ensure_video_context

# --- Caption layout (DCMP Captioning Key and common web practice) ---
MAX_LINE_CHARS = 42
MAX_CAPTION_CHARS = 2 * MAX_LINE_CHARS
MAX_CAPTION_SECONDS = 7.0
MIN_CAPTION_SECONDS = 1.0
# Captions stay up (into following silence) long enough to read at this rate.
READING_CHARS_PER_SECOND = 17.0
# Typical speaking rate, used to estimate how long untimed words take.
SPEECH_CHARS_PER_SECOND = 15.0
# Gaps between captions shorter than this are closed to avoid flicker.
CLOSE_GAP_SECONDS = 0.5
# A pause at least this long always ends a caption.
PAUSE_BREAK_SECONDS = 2.0
# Crosstalk is moved up to this far to a sentence or clause boundary of the
# speech it interrupts.
CROSSTALK_SHIFT_SECONDS = 3.0
# Cost of each extra caption when segmenting, against the cost of breaking a
# phrase (see _pair_break_cost): lower gives shorter captions with cleaner
# breaks, higher gives fuller captions. Line breaks inside a caption count
# for LINE_BREAK_WEIGHT of a caption break.
CAPTION_COST = 0.6
LINE_BREAK_WEIGHT = 0.6
# Cost of putting a speaker label on its own line above a one-line caption.
LABEL_LINE_COST = 0.3

# --- LLM passes ---
SECTION_CUES = 150
SECTION_CONTEXT_BEFORE = 8
SECTION_CONTEXT_AFTER = 4
# Parallel section requests; lower with YTD_CC_WORKERS if rate limited.
DEFAULT_WORKERS = 6
LLM_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 15
# The speaker pass runs in windows of about this many cues (~20 minutes of
# speech-to-text), with a few cues either side for continuity.
SPEAKER_WINDOW_CUES = 300
SPEAKER_CONTEXT_BEFORE = 30
SPEAKER_CONTEXT_AFTER = 10
# Words are aligned in blocks of this many, so long videos stay fast.
ALIGN_BLOCK_WORDS = 2000

UNKNOWN_SPEAKER = "Unidentified Speaker"


@dataclass
class Cue:
    """
    One SRT cue. `label` is the source's diarization label (e.g. "Speaker 3"),
    `turn_start` marks a speaker change at this cue, and `hints` holds the word
    positions where the second transcript marks a speaker change.
    """

    start: float
    end: float
    text: str
    label: str = ""
    turn_start: bool = False
    hints: List[int] = field(default_factory=list)


@dataclass
class CorrectedCaptions:
    srt: str
    input_tokens: int
    output_tokens: int
    captions: int
    repaired_cues: int
    fallback_cues: int
    added_segments: int


@dataclass
class _Word:
    norm: str
    time: float
    cue: int
    pos: int
    turn_before: bool = False


@dataclass
class _Token:
    text: str
    start: float
    end: float
    # Joined to the previous token without a space (after an em dash).
    glued: bool = False


@dataclass
class _Segment:
    """Corrected text for all or part of a cue, with its timing."""

    start: float
    end: float
    speaker: str
    tokens: List[_Token]


@dataclass
class _Caption:
    start: float
    end: float
    speech_end: float
    lines: List[str]
    labelled: bool


@dataclass
class _Person:
    name: str
    title: str
    times: List[float]
    note: str = ""


@dataclass
class _VideoContext:
    people: List[_Person] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)
    events: List[Tuple[float, str]] = field(default_factory=list)
    raw: str = ""


# ---------------------------------------------------------------------------
# SRT parsing and formatting
# ---------------------------------------------------------------------------

_TIMING_RE = re.compile(
    r"(\d+:)?(\d{1,2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d+:)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
)
_TOKEN_RE = re.compile(r"\[[^\]]*\]|\S+")
_DIARIZATION_RE = re.compile(r"^\[(Speaker \d+)\]\s*")


def _to_seconds(hours: Optional[str], minutes: str, seconds: str, ms: str) -> float:
    return (
        (int(hours[:-1]) if hours else 0) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(ms.ljust(3, "0")) / 1000
    )


def parse_srt(content: str) -> List[Cue]:
    """Parses SRT text into cues, joining multi-line cue text with spaces."""
    cues: List[Cue] = []
    text = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [line.strip() for line in block.split("\n")]
        for i, line in enumerate(lines):
            match = _TIMING_RE.search(line)
            if match:
                groups = match.groups()
                cue_text = " ".join(part for part in lines[i + 1 :] if part)
                cues.append(
                    Cue(
                        _to_seconds(*groups[:4]),
                        _to_seconds(*groups[4:]),
                        cue_text,
                    )
                )
                break
    return cues


def format_timestamp(seconds: float) -> str:
    """Formats seconds as an SRT timestamp (HH:MM:SS,mmm)."""
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _clock(seconds: float) -> str:
    """HH:MM:SS, as used in prompts."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _tokens(text: str) -> List[str]:
    """Words, keeping bracketed sound descriptions like [clears throat] whole."""
    return _TOKEN_RE.findall(text)


def _is_rolling(cues: List[Cue]) -> bool:
    """
    YouTube automatic captions "roll": each line stays up until the line after
    next appears, so most cues overlap the following one.
    """
    if len(cues) < 4:
        return False
    overlaps = sum(1 for a, b in zip(cues, cues[1:]) if a.end > b.start + 0.1)
    return overlaps > 0.5 * (len(cues) - 1)


def prepare_cues(cues: List[Cue]) -> List[Cue]:
    """
    Normalizes source cues: rolling cues end when the next line appears,
    `[Speaker N]` diarization labels (Gemini STT) move into `label` and
    `turn_start`, and cues are split at `>>` speaker-change marks (YouTube) so
    every speaker change falls on a cue boundary.
    """
    if _is_rolling(cues):
        cues = [
            Cue(cue.start, min(cue.end, nxt.start), cue.text)
            if nxt.start > cue.start
            else cue
            for cue, nxt in zip(cues, cues[1:])
        ] + cues[-1:]

    prepared: List[Cue] = []
    label = ""
    pending_turn = False
    for cue in cues:
        text = cue.text.strip()
        turn = pending_turn
        match = _DIARIZATION_RE.match(text)
        if match:
            label, text, turn = match.group(1), text[match.end() :], True
        parts = [" ".join(part.split()) for part in text.split(">>")]
        pieces = [(k > 0, part) for k, part in enumerate(parts) if part]
        pending_turn = turn and not pieces
        total = sum(len(piece) for _, piece in pieces)
        start = cue.start
        for k, (marked, piece) in enumerate(pieces):
            end = (
                cue.end
                if k == len(pieces) - 1
                else start + (cue.end - cue.start) * len(piece) / total
            )
            prepared.append(Cue(start, end, piece, label, turn or marked))
            turn = False
            start = end
    return prepared


# ---------------------------------------------------------------------------
# Word alignment with the second transcript, and timing repair
# ---------------------------------------------------------------------------


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower()).strip("'")


def _words(cues: List[Cue]) -> List[_Word]:
    """Words with start times interpolated across each cue by character offset."""
    words: List[_Word] = []
    for index, cue in enumerate(cues):
        tokens = _tokens(cue.text)
        weights = [len(token) + 1 for token in tokens]
        total = sum(weights) or 1
        offset = 0
        first = True
        for pos, token in enumerate(tokens):
            norm = "" if token.startswith("[") else _norm(token)
            if norm:
                words.append(
                    _Word(
                        norm,
                        cue.start + (cue.end - cue.start) * offset / total,
                        index,
                        pos,
                        cue.turn_start and first,
                    )
                )
                first = False
            offset += weights[pos]
    return words


def _align(a: List[str], b: List[str]) -> List[Tuple[int, int]]:
    """
    Index pairs of matching words in two transcripts of the same audio. Runs in
    overlapping blocks: a single global diff grows quadratically with length,
    and aligning by text rather than time copes with broken timestamps.
    """
    pairs: List[Tuple[int, int]] = []
    i0 = j0 = 0
    while i0 < len(a) and j0 < len(b):
        i1 = min(len(a), i0 + ALIGN_BLOCK_WORDS)
        j1 = min(len(b), j0 + 2 * (i1 - i0) + 200)
        last = i1 == len(a)
        keep = i1 if last else i0 + (i1 - i0) * 3 // 4
        matcher = difflib.SequenceMatcher(None, a[i0:i1], b[j0:j1], autojunk=False)
        next_j0 = None
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                if i0 + block.a + k >= keep:
                    break
                pairs.append((i0 + block.a + k, j0 + block.b + k))
                next_j0 = j0 + block.b + k + 1
        if last:
            break
        j0 = next_j0 if next_j0 is not None else j0 + (keep - i0)
        i0 = keep
    return pairs


def _is_broken(cues: List[Cue], i: int) -> bool:
    """
    True if a cue's timestamps can't be right: it ends before it starts, lasts
    far longer than its words could take, or sits a minute or more away from
    its neighbors (long Gemini STT runs occasionally return a few words with
    offsets that are minutes off).
    """
    cue = cues[i]
    duration = cue.end - cue.start
    if duration < 0 or duration > max(20.0, 0.75 * len(_tokens(cue.text))):
        return True
    neighbors = [c.start for c in cues[max(0, i - 5) : i] + cues[i + 1 : i + 6]]
    return bool(neighbors) and abs(cue.start - statistics.median(neighbors)) > 60


def repair_timing(
    cues: List[Cue],
    cue_words: List[_Word],
    ref_words: List[_Word],
    pairs: List[Tuple[int, int]],
) -> int:
    """
    Re-times cues with broken timestamps, from the second transcript's timing
    for the same words when they align, otherwise by spreading them between
    their nearest good neighbors. Returns the number of cues repaired.
    """
    broken = [i for i in range(len(cues)) if _is_broken(cues, i)]
    if not broken:
        return 0

    matched: Dict[int, List[Tuple[int, float]]] = {}
    for i, j in pairs:
        word = cue_words[i]
        matched.setdefault(word.cue, []).append((word.pos, ref_words[j].time))

    unresolved = set()
    for i in broken:
        cue = cues[i]
        hits = matched.get(i)
        if not hits:
            unresolved.add(i)
            continue
        tokens = _tokens(cue.text)
        first_pos, first_time = hits[0]
        last_pos, last_time = hits[-1]
        before = sum(len(t) + 1 for t in tokens[:first_pos])
        after = sum(len(t) + 1 for t in tokens[last_pos:])
        cue.start = max(0.0, first_time - before / SPEECH_CHARS_PER_SECOND)
        cue.end = max(last_time + after / SPEECH_CHARS_PER_SECOND, cue.start + 0.3)
    unresolved.update(i for i in broken if i not in unresolved and _is_broken(cues, i))

    # Spread anything still broken between the closest good cues.
    index = 0
    order = sorted(unresolved)
    while index < len(order):
        run_end = index
        while run_end + 1 < len(order) and order[run_end + 1] == order[run_end] + 1:
            run_end += 1
        first, last = order[index], order[run_end]
        chars = [len(cues[k].text) + 1 for k in range(first, last + 1)]
        needed = sum(chars) / SPEECH_CHARS_PER_SECOND
        lo = cues[first - 1].end if first > 0 else None
        hi = cues[last + 1].start if last + 1 < len(cues) else None
        if lo is None:
            lo = max(0.0, (hi or 0.0) - needed)
        if hi is None or hi - lo < needed / 2:
            hi = lo + needed
        t = lo
        for k, width in zip(range(first, last + 1), chars):
            span = (hi - lo) * width / sum(chars)
            cues[k].start, cues[k].end = t, t + span
            t += span
        index = run_end + 1
    return len(broken)


def add_reference_hints(
    cues: List[Cue],
    cue_words: List[_Word],
    ref_words: List[_Word],
    pairs: List[Tuple[int, int]],
) -> None:
    """
    Marks, as word positions in the source cues, where the second transcript
    detected a change of speaker (YouTube `>>` marks or diarization turns).
    """
    ref_to_cue_word = {j: i for i, j in pairs}
    for j, word in enumerate(ref_words):
        if not word.turn_before:
            continue
        for k in range(j, min(j + 3, len(ref_words))):
            if k in ref_to_cue_word:
                target = cue_words[ref_to_cue_word[k]]
                hints = cues[target.cue].hints
                if target.pos not in hints:
                    hints.append(target.pos)
                break


def _hinted_text(cue: Cue) -> str:
    return " ".join(
        f">> {token}" if pos in cue.hints else token
        for pos, token in enumerate(_tokens(cue.text))
    )


# ---------------------------------------------------------------------------
# On-screen context from the video (`--from-video` / automatic for -cc)
# ---------------------------------------------------------------------------

_DASH_RE = re.compile(r"\s+(?:[—–]|--?)\s+")
_EVENT_RE = re.compile(r"^[-*]\s*(\d+:\d{2}(?::\d{2})?)\s+(?:[—–]|--?)\s+(.+)$")


def _parse_clock(value: str) -> Optional[float]:
    """Seconds from MM:SS (minutes may exceed 59) or HH:MM:SS."""
    parts = value.strip().split(":")
    if not 2 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def parse_video_context(text: str) -> _VideoContext:
    """Parses the People / Names and terms / On-screen events sections."""
    context = _VideoContext(raw=text.strip())
    section = ""
    for line in text.splitlines():
        line = line.strip()
        heading = re.match(r"^#+\s*(.+?)\s*$", line)
        if heading:
            section = heading.group(1).lower()
            continue
        if not line.startswith(("-", "*")):
            continue
        if section.startswith("people"):
            # "- Name (Title) — speaking at 03:07, 12:40" or "— <other note>"
            head, *rest = _DASH_RE.split(line.lstrip("-* ").strip(), maxsplit=1)
            named = re.match(r"^(.*?)\s*\((.*)\)\s*$", head)
            name, title = (named.group(1), named.group(2)) if named else (head, "")
            note = rest[0] if rest else ""
            times = []
            if note.lower().startswith("speaking at"):
                for value in note[len("speaking at") :].split(","):
                    seconds = _parse_clock(value)
                    if seconds is not None:
                        times.append(seconds)
                note = ""
            context.people.append(_Person(name.strip(), title.strip(), times, note))
        elif section.startswith("names"):
            context.terms.append(line.lstrip("-* ").strip())
        elif section.startswith("on-screen") or section.startswith("events"):
            match = _EVENT_RE.match(line)
            if match:
                seconds = _parse_clock(match.group(1))
                if seconds is not None:
                    context.events.append((seconds, match.group(2).strip()))
    return context


def _people_text(context: _VideoContext, with_times: bool = True) -> str:
    lines = []
    for person in context.people:
        line = f"- {person.name}"
        if person.title:
            line += f" ({person.title})"
        if with_times and person.times:
            line += " — speaking at " + ", ".join(_clock(t) for t in person.times)
        if person.note:
            line += f" — {person.note}"
        lines.append(line)
    return "\n".join(lines)


def _speaker_lists_text(speaker_lists: Sequence[Tuple[str, str]]) -> str:
    blocks = []
    for source, text in speaker_lists:
        text = (text or "").strip()
        if text and text not in ("nan", 'float("nan")'):
            blocks.append(f"From {source}:\n{text}")
    return "\n\n".join(blocks)


def _context_window_text(
    context: _VideoContext, start: float, end: float, margin: float = 60.0
) -> Tuple[str, str]:
    """Speaking times and on-screen events falling in [start, end] (+ margin)."""
    times = []
    for person in context.people:
        hits = [t for t in person.times if start - margin <= t <= end + margin]
        if hits:
            times.append(f"- {person.name}: " + ", ".join(_clock(t) for t in hits))
    events = [
        f"- {_clock(t)} — {text}"
        for t, text in context.events
        if start - margin <= t <= end + margin
    ]
    return "\n".join(times), "\n".join(events)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _ask(
    model_name: str, prompt: str, valid: Callable[[str], bool]
) -> Tuple[str, int, int]:
    """
    Queries the model, retrying when it errors (e.g. rate limits) or returns
    something unusable. Returns the last response and the tokens of all tries.
    """
    input_tokens = output_tokens = 0
    response = ""
    for attempt in range(LLM_ATTEMPTS):
        response, in_tok, out_tok = _query_llm(model_name, prompt)
        input_tokens += in_tok
        output_tokens += out_tok
        if not response.startswith("Error") and valid(response):
            break
        if attempt < LLM_ATTEMPTS - 1:
            time.sleep(RETRY_WAIT_SECONDS * (attempt + 1))
    return response, input_tokens, output_tokens


def _clean_name(name: str) -> str:
    name = name.strip().strip("*`\"'")
    if name.startswith("[") and name.endswith("]"):
        name = name[1:-1]
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip(" :,")
    name = " ".join(name.split())[:60]
    if not name or re.fullmatch(r"speaker\s*\d+|unknown|n/?a|none", name, re.I):
        return UNKNOWN_SPEAKER
    return name


def _clean_text(text: str, speaker: str) -> str:
    text = re.sub(r"\((?:Speaker \d+)\)|\[Speaker \d+\]|>>", " ", text)
    text = " ".join(text.split())
    label = re.match(r"^\[([^\]]+)\]\s*", text)
    if label and _clean_name(label.group(1)).lower() == speaker.lower():
        text = text[label.end() :]
    return text.strip()


# ---------------------------------------------------------------------------
# Pass 1: speaker identification, in windows with the evidence on each line
# ---------------------------------------------------------------------------

_SPEAKER_LINE_RE = re.compile(r"^\s*[#\[]?(\d+)\]?\s*[|:\t]\s*(.+?)\s*$")


def video_speaker_hints(
    cues: List[Cue], context: _VideoContext
) -> Dict[int, List[str]]:
    """
    Cue indices where the video shows someone speaking (the "speaking at" times
    in the on-screen context), so that evidence can sit on the transcript line
    it applies to instead of in a long list the model has to cross-reference.
    """
    starts = [cue.start for cue in cues]
    hints: Dict[int, List[str]] = {}
    for person in context.people:
        for t in person.times:
            # The times are coarse (whole seconds), usually within a second or
            # two of the turn's start and slightly early, so prefer a
            # voice-turn start just after the time, then the next cue, then the
            # cue in progress.
            lo, hi = bisect_left(starts, t - 2), bisect_right(starts, t + 3)
            turns = [
                j for j in range(lo, hi) if cues[j].turn_start or 0 in cues[j].hints
            ]
            following = bisect_left(starts, t - 0.5)
            current = bisect_right(starts, t) - 1
            if turns:
                k = min(turns, key=lambda j: abs(starts[j] - t - 0.5))
            elif following < len(cues) and starts[following] - t <= 3:
                k = following
            elif current >= 0 and cues[current].end >= t - 0.5:
                k = current
            else:
                continue
            names = hints.setdefault(k, [])
            if person.name not in names:
                names.append(person.name)
    return hints


def _speaker_windows(count: int) -> List[Tuple[int, int]]:
    windows = max(1, math.ceil(count / SPEAKER_WINDOW_CUES))
    size = math.ceil(count / windows)
    return [(start, min(count, start + size)) for start in range(0, count, size)]


def build_speaker_prompt(
    cues: List[Cue],
    start: int,
    end: int,
    part: int,
    parts: int,
    context: _VideoContext,
    speaker_lists: Sequence[Tuple[str, str]],
    video_title: str,
    hints: Dict[int, List[str]],
) -> str:
    def line(k: int) -> str:
        cue = cues[k]
        label = f"({cue.label}) " if cue.turn_start and cue.label else ""
        seen = "".join("{video: " + name + "} " for name in hints.get(k, []))
        return f"{k + 1} [{_clock(cue.start)}] {label}{seen}{_hinted_text(cue)}"

    t0, t1 = cues[start].start, cues[end - 1].end
    sections = [
        "You are identifying who speaks each line of a video's transcript, so "
        "its captions can name the speakers (WCAG 2.1 Level AA / Section 508 "
        "speaker identification).",
        (f"Video: {video_title}\n" if video_title else "")
        + f"Part {part} of {parts}: {_clock(t0)}–{_clock(t1)}",
    ]
    if context.people:
        sections.append(
            "## People in the video (names and titles shown on screen; "
            "authoritative spellings)\n" + _people_text(context, with_times=False)
        )
    elif context.raw:
        sections.append("## On-screen context from the video\n" + context.raw)
    lists = _speaker_lists_text(speaker_lists)
    if lists:
        sections.append("## Speakers found by earlier analysis\n" + lists)
    _, events = _context_window_text(context, t0, t1)
    if events:
        sections.append("## On-screen events in this part\n" + events)

    before = max(0, start - SPEAKER_CONTEXT_BEFORE)
    after = min(len(cues), end + SPEAKER_CONTEXT_AFTER)
    transcript = (
        "## Transcript\n"
        "Each line is `cue id [start time] text`, with evidence about the "
        "speaker:\n"
        "- `{video: Name}`: the video shows Name speaking at about this time. "
        "Strong evidence of who is speaking around here, but the time is "
        "approximate, so their turn can start a line or two later (or "
        "earlier).\n"
        "- `(Speaker N)`: automatic voice label from speech-to-text, at the start "
        "of each voice turn. The same number means the same voice only within "
        "about 20 minutes of the recording, and one label sometimes covers two "
        "similar voices.\n"
        "- `>>`: a second, independent transcript detected a change of speaker "
        "here."
    )
    if before < start:
        transcript += "\n\nContext before this part (don't output):\n" + "\n".join(
            line(k) for k in range(before, start)
        )
    transcript += "\n\nLines to label:\n" + "\n".join(
        line(k) for k in range(start, end)
    )
    if end < after:
        transcript += "\n\nContext after this part (don't output):\n" + "\n".join(
            line(k) for k in range(end, after)
        )
    sections.append(transcript)
    sections.append(
        "## Task\n"
        "Name the speaker of every line to label. A speaker usually continues "
        "until a voice label, `>>` or the content shows a change: replies, "
        'questions, hand-offs (e.g. "Liz, go ahead" or "I\'ll turn it over to '
        'Megan"), people addressed by name (they often speak next), '
        "self-introductions and roles (e.g. the chair opens the meeting, calls "
        "votes and moves the agenda; a presenter walks through their slides). "
        "Weigh `{video: Name}` together with the voice labels, `>>` and the "
        "content to place each change of speaker on the right line, and use what "
        "it tells you about a voice label for that label's other turns nearby.\n\n"
        "Output exactly one line per line to label, in order, formatted `cue "
        f"id|speaker name`, from cue {start + 1} to cue {end}. Write each name "
        "as in the people list (e.g. `Dr. Jane Smith`), without the job title, "
        "and spell it the same way every time. If you can't tell who is "
        "speaking, use a short description such as `Board Member`, `Presenter`, "
        f"`Audience Member` or `{UNKNOWN_SPEAKER}`. Output only these lines."
    )
    return "\n\n".join(section for section in sections if section)


def parse_speaker_lines(response: str, ids: range) -> Dict[int, str]:
    found: Dict[int, str] = {}
    for line in response.splitlines():
        match = _SPEAKER_LINE_RE.match(line)
        if match and int(match.group(1)) in ids:
            found.setdefault(int(match.group(1)), _clean_name(match.group(2)))
    return found


def identify_speakers(
    model_name: str,
    cues: List[Cue],
    context: _VideoContext,
    speaker_lists: Sequence[Tuple[str, str]],
    video_title: str,
) -> Tuple[List[str], int, int]:
    """
    Names the speaker of every cue, in parallel windows of the transcript.
    Returns (speakers, input_tokens, output_tokens).
    """
    hints = video_speaker_hints(cues, context)
    windows = _speaker_windows(len(cues))
    found: Dict[int, str] = {}
    input_tokens = output_tokens = 0

    def run(index: int) -> Tuple[Dict[int, str], int, int]:
        start, end = windows[index]
        ids = range(start + 1, end + 1)
        prompt = build_speaker_prompt(
            cues,
            start,
            end,
            index + 1,
            len(windows),
            context,
            speaker_lists,
            video_title,
            hints,
        )

        def valid(response: str) -> bool:
            return len(parse_speaker_lines(response, ids)) >= 0.8 * len(ids)

        response, in_tok, out_tok = _ask(model_name, prompt, valid)
        return parse_speaker_lines(response, ids), in_tok, out_tok

    workers = max(1, int(os.environ.get("YTD_CC_WORKERS", DEFAULT_WORKERS)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for names, in_tok, out_tok in pool.map(run, range(len(windows))):
            found.update(names)
            input_tokens += in_tok
            output_tokens += out_tok

    if not found:
        rprint("[yellow]Speaker identification failed; speakers are unnamed.[/yellow]")
    speakers: List[str] = []
    current = found[min(found)] if found else UNKNOWN_SPEAKER
    for cue_id in range(1, len(cues) + 1):
        current = found.get(cue_id, current)
        speakers.append(current)
    return speakers, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Pass 2: text correction in parallel sections
# ---------------------------------------------------------------------------

_EDIT_RE = re.compile(r"^\s*#?(\d+)(\+?)\s*\|([^|]*)\|(.*)$")

_EDIT_RULES = """\
## Rules
1. Words: correct misrecognized words, choosing what was actually said by \
comparing both transcripts and using the context, the names and terms above and \
general knowledge (people, places, programs, legislation). Use the exact \
spellings listed above for names, titles, organizations, programs, acronyms and \
bill numbers. Never paraphrase, summarize, reorder or polish what was said.
2. Clean verbatim: drop filler sounds (um, uh, er, ah) and stutters or \
accidental repeats ("the the", "I, I") unless they matter to the meaning. Keep \
false starts and self-corrections, marked with an em dash (—).
3. Punctuation: use standard capitalization and punctuation, including question \
marks, apostrophes, hyphens and quotation marks for quoted speech. Sentences \
often run across cues: punctuate each cue so the sentences read correctly when \
the cues are joined, and only capitalize a cue's first word if it starts a \
sentence or is a proper noun.
4. Numbers: use numerals for statistics, percentages (47%), money ($3 million), \
times, dates, grades and years; spell out other numbers from one to nine.
5. Sounds: add meaningful non-speech information in lowercase square brackets, \
such as [laughter], [applause], [inaudible], [crosstalk] or [music], where it \
is heard (the second transcript often marks these) or clearly implied. Use \
[inaudible] for speech that can't be made out. Don't add on-screen text that \
isn't spoken.
6. Speakers: each cue's speaker was identified earlier from the voices and the \
whole video; copy it unchanged. You only choose speakers for new lines (rules 7 \
and 8), writing names exactly as in the people list.
7. Two speakers in one cue: if a cue clearly contains speech from two people \
(often where `>>` appears inside it), output it as two lines with the same id, \
one per speaker, in the order spoken.
8. Missing speech: if the second transcript has speech that is missing from the \
cues (often short replies such as "Yes." or "Thank you."), add it as a new line \
`id+|speaker|text` straight after the cue it follows.
9. Keep each cue's words in that cue, since its timing depends on them: don't \
move words or sentences to other cues.

## Output
One line per cue to edit, in order, formatted `id|speaker|text`, plus any \
`id+` lines. Always fill in the speaker. Output nothing else: no context cues, \
headings, notes or code fences."""


def _section_ranges(cues: List[Cue], speakers: List[str]) -> List[Tuple[int, int]]:
    """
    Splits cue indices into sections of about SECTION_CUES, ending each at a
    speaker change, sentence end or pause near the target length.
    """
    ranges = []
    start = 0
    while start < len(cues):
        target = start + SECTION_CUES
        if target >= len(cues) - SECTION_CUES // 3:
            ranges.append((start, len(cues)))
            break
        best, best_score = target, -1.0
        for k in range(target - 20, target + 21):
            score = 0.0
            if speakers[k] != speakers[k - 1]:
                score = 3
            elif cues[k - 1].text.rstrip().endswith((".", "?", "!")):
                score = 2
            score += min(cues[k].start - cues[k - 1].end, 2.0) / 2
            if score > best_score or (
                score == best_score and abs(k - target) < abs(best - target)
            ):
                best, best_score = k, score
        ranges.append((start, best))
        start = best
    return ranges


def build_section_prompt(
    cues: List[Cue],
    speakers: List[str],
    start: int,
    end: int,
    section: int,
    sections: int,
    context: _VideoContext,
    speaker_lists: Sequence[Tuple[str, str]],
    reference: List[Cue],
    video_title: str,
    source_name: str,
    reference_name: str,
) -> str:
    def cue_lines(lo: int, hi: int) -> str:
        return "\n".join(
            f"{k + 1}|{speakers[k]}|{_hinted_text(cues[k])}" for k in range(lo, hi)
        )

    t0, t1 = cues[start].start, cues[end - 1].end
    parts = [
        "You are a professional captioner. Edit the cues below into accurate, "
        "readable captions for people who are deaf or hard of hearing, meeting "
        "WCAG 2.1 Level AA (1.2.2 Captions) and Section 508 "
        "(https://www.section508.gov/create/captions-transcripts/) and following "
        "the DCMP Captioning Key: captions match what is said, identify speakers "
        "and include meaningful non-speech sounds.",
        (f"Video: {video_title}\n" if video_title else "")
        + f"Section {section} of {sections}: {_clock(t0)}–{_clock(t1)}",
    ]
    if context.people:
        parts.append(
            "## People (use these exact spellings)\n"
            + _people_text(context, with_times=False)
        )
    lists = _speaker_lists_text(speaker_lists)
    if lists and not context.people:
        parts.append("## Speakers found by earlier analysis\n" + lists)
    if context.terms:
        parts.append(
            "## Names and terms shown on screen (authoritative spellings)\n"
            + "\n".join(f"- {term}" for term in context.terms)
        )
    elif context.raw and not context.people:
        parts.append("## On-screen context from the video\n" + context.raw)
    times, events = _context_window_text(context, t0, t1)
    if times:
        parts.append("## Speaking times in this section (from the video)\n" + times)
    if events:
        parts.append("## On-screen events in this section\n" + events)

    if reference:
        starts = [cue.start for cue in reference]
        lo = bisect_left(starts, t0 - 5)
        hi = bisect_right(starts, t1 + 5)
        ref_lines = "\n".join(
            f"[{_clock(cue.start)}] {'>> ' if cue.turn_start else ''}{cue.text}"
            for cue in reference[lo:hi]
        )
        if ref_lines:
            parts.append(
                f"## Second transcript ({reference_name}) of the same audio\n"
                "Produced independently: where it disagrees with the cues, one "
                "of them is usually right. `>>` marks a change of speaker.\n"
                + ref_lines
            )

    cue_intro = (
        f"## Cues ({source_name})\n"
        "Formatted `id|speaker|text`. `>>` inside the text marks where the second "
        "transcript detected a change of speaker."
    )
    before = max(0, start - SECTION_CONTEXT_BEFORE)
    after = min(len(cues), end + SECTION_CONTEXT_AFTER)
    if before < start:
        cue_intro += "\n\nContext before this section (don't output):\n" + cue_lines(
            before, start
        )
    cue_intro += "\n\nCues to edit:\n" + cue_lines(start, end)
    if end < after:
        cue_intro += "\n\nContext after this section (don't output):\n" + cue_lines(
            end, after
        )
    parts.append(cue_intro)
    parts.append(_EDIT_RULES)
    return "\n\n".join(parts)


def parse_section_edits(
    response: str, ids: range, speakers: List[str]
) -> Tuple[Dict[int, List[Tuple[str, str]]], Dict[int, List[Tuple[str, str]]]]:
    """
    Parses `id|speaker|text` lines into per-cue edits and `id+` additions. A
    missing speaker falls back to the speaker pass's choice for that cue.
    """
    edits: Dict[int, List[Tuple[str, str]]] = {}
    additions: Dict[int, List[Tuple[str, str]]] = {}
    for line in response.splitlines():
        match = _EDIT_RE.match(line)
        if not match or int(match.group(1)) not in ids:
            continue
        cue_id = int(match.group(1))
        speaker = (
            _clean_name(match.group(3))
            if match.group(3).strip()
            else speakers[cue_id - 1]
        )
        target = additions if match.group(2) else edits
        target.setdefault(cue_id, []).append(
            (speaker, _clean_text(match.group(4), speaker))
        )
    return edits, additions


def correct_sections(
    model_name: str,
    cues: List[Cue],
    speakers: List[str],
    context: _VideoContext,
    speaker_lists: Sequence[Tuple[str, str]],
    reference: List[Cue],
    video_title: str,
    source_name: str,
    reference_name: str,
) -> Tuple[
    Dict[int, List[Tuple[str, str]]], Dict[int, List[Tuple[str, str]]], int, int
]:
    ranges = _section_ranges(cues, speakers)
    edits: Dict[int, List[Tuple[str, str]]] = {}
    additions: Dict[int, List[Tuple[str, str]]] = {}
    input_tokens = output_tokens = 0

    def run(index: int) -> Tuple[dict, dict, int, int]:
        start, end = ranges[index]
        ids = range(start + 1, end + 1)
        prompt = build_section_prompt(
            cues,
            speakers,
            start,
            end,
            index + 1,
            len(ranges),
            context,
            speaker_lists,
            reference,
            video_title,
            source_name,
            reference_name,
        )

        def valid(response: str) -> bool:
            found, _ = parse_section_edits(response, ids, speakers)
            return len(found) >= 0.6 * len(ids)

        response, in_tok, out_tok = _ask(model_name, prompt, valid)
        found, added = parse_section_edits(response, ids, speakers)
        return found, added, in_tok, out_tok

    workers = max(1, int(os.environ.get("YTD_CC_WORKERS", DEFAULT_WORKERS)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, k) for k in range(len(ranges))]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            found, added, in_tok, out_tok = future.result()
            edits.update(found)
            additions.update(added)
            input_tokens += in_tok
            output_tokens += out_tok
            rprint(f"  Corrected caption section {done}/{len(ranges)}")
    return edits, additions, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Layout: timed tokens -> readable, non-overlapping captions
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"[.?!…][\"')\]]*$")
_CLAUSE_END_RE = re.compile(r"(?:[,;:—–]|--)[\"')\]]*$")
_ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "mx.",
    "st.",
    "sen.",
    "rep.",
    "del.",
    "gov.",
    "lt.",
    "prof.",
    "rev.",
    "hon.",
    "gen.",
    "sgt.",
    "capt.",
    "jr.",
    "sr.",
    "vs.",
    "no.",
}
# Natural places to start a new caption or line: before a conjunction or
# relative pronoun, and (slightly less so) before a preposition.
_BREAK_BEFORE = {
    "and",
    "but",
    "or",
    "so",
    "because",
    "which",
    "who",
    "that",
    "when",
    "where",
    "while",
    "if",
    "although",
    "though",
    "then",
    "than",
    "since",
    "unless",
    "until",
}
_PREPOSITIONS = {
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "from",
    "by",
    "about",
    "into",
    "over",
    "across",
    "through",
    "after",
    "before",
    "during",
    "between",
    "under",
    "without",
}
# Words that shouldn't end a caption or line (they belong with the next word).
_NO_BREAK_AFTER = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "from",
    "by",
    "about",
    "into",
    "my",
    "your",
    "our",
    "their",
    "his",
    "her",
    "its",
    "this",
    "these",
    "those",
} | {abbreviation for abbreviation in _ABBREVIATIONS if abbreviation != "no."}


def _bare(word: str) -> str:
    return word.lower().strip("\"'()[]")


def _ends_sentence(word: str, next_word: Optional[str]) -> bool:
    if not _SENTENCE_END_RE.search(word):
        return False
    bare = _bare(word)
    if bare in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", bare):
        return False
    return next_word is None or not next_word[:1].islower()


def _pair_break_cost(word: str, next_word: Optional[str], gap: float = 0.0) -> float:
    """
    Cost of a caption or line break between two words (lower is better):
    sentence ends are free, then sound tags, clause ends, conjunctions and
    prepositions. Breaking inside a phrase is expensive, and more so after an
    article or title, inside a capitalized name ("Vice Chair Brit Kirwan") or
    between a number and its unit ("47 percent").
    """
    following = _bare(next_word) if next_word is not None else ""
    if _ends_sentence(word, next_word):
        return max(-min(gap, 1.0) * 0.8, 0.0)
    if word.startswith("[") or (next_word or "").startswith("["):
        cost = 0.2
    elif _CLAUSE_END_RE.search(word):
        cost = 0.4
    elif following in _BREAK_BEFORE:
        cost = 0.8
    elif following in _PREPOSITIONS:
        cost = 1.1
    else:
        cost = 2.4
    if _bare(word) in _NO_BREAK_AFTER:
        cost += 1.5
    elif next_word and re.fullmatch(r"[$£€]?\d[\d,.]*", word):
        cost += 1.2
    elif (
        next_word
        and word[:1].isupper()
        and next_word[:1].isupper()
        and word[-1].isalnum()
    ):
        cost += 1.0
    return max(cost - min(gap, 1.0) * 0.8, 0.0)


def _break_cost(words: Sequence[str], j: int, gap: float = 0.0) -> float:
    """Cost of breaking before words[j]; the ends of `words` are free."""
    if j <= 0 or j >= len(words):
        return 0.0
    return _pair_break_cost(words[j - 1], words[j], gap)


def _join(words: Sequence[str], spaces: Sequence[int]) -> str:
    """Joins words, with spaces[k] (0 or 1) spaces before words[k]."""
    return "".join(
        (" " * spaces[k] if k else "") + word for k, word in enumerate(words)
    )


def _line_split(
    words: Sequence[str],
    spaces: Sequence[int],
    pair: Sequence[float],
    i: int,
    j: int,
    extra: int,
) -> Tuple[Optional[int], float]:
    """
    Best place to split words[i:j] into two lines of at most MAX_LINE_CHARS
    (the first also holding `extra` characters of speaker label), weighing the
    break (`pair[k]` is the cost of breaking before words[k]) against uneven
    line lengths, slightly preferring a longer bottom line. Index i puts the
    label on a line of its own. Returns (index, cost), or (None, inf) if the
    words can't fit on two lines.
    """
    total = sum(len(word) + spaces[k] for k, word in enumerate(words) if i <= k < j)
    total -= spaces[i]
    best: Optional[int] = None
    best_cost = math.inf
    if extra and total <= MAX_LINE_CHARS:
        label = extra - 1
        best = i
        best_cost = (
            LABEL_LINE_COST
            + abs(label - total) / MAX_LINE_CHARS
            + (0.15 if label > total else 0.0)
        )
    first = extra - spaces[i]
    for k in range(i + 1, j):
        first += len(words[k - 1]) + spaces[k - 1]
        if first > MAX_LINE_CHARS:
            break
        second = total - (first - extra) - spaces[k]
        if second > MAX_LINE_CHARS:
            continue
        cost = (
            pair[k]
            + abs(first - second) / MAX_LINE_CHARS
            + (0.15 if first > second else 0.0)
        )
        if cost < best_cost:
            best, best_cost = k, cost
    return best, best_cost


def _pair_costs(words: Sequence[str]) -> List[float]:
    """pair[k]: cost of a line break before words[k]."""
    return [0.0] + [
        _pair_break_cost(words[k - 1], words[k]) for k in range(1, len(words))
    ]


def _layout_lines(words: Sequence[str], spaces: Sequence[int], label: str) -> List[str]:
    """Breaks a caption into one or two lines at the most natural point."""
    prefix = f"{label} " if label else ""
    text = prefix + _join(words, spaces)
    if len(text) <= MAX_LINE_CHARS or (len(words) < 2 and not label):
        return [text]
    k, _ = _line_split(words, spaces, _pair_costs(words), 0, len(words), len(prefix))
    if k is None:
        return [text]
    if k == 0:
        return [label, _join(words, spaces)]
    return [prefix + _join(words[:k], spaces[:k]), _join(words[k:], spaces[k:])]


def _caption_spans(tokens: List[_Token], label: str) -> List[Tuple[int, int]]:
    """
    Splits one speaker's run of tokens into captions with dynamic programming,
    minimizing the number of captions plus the cost of each caption and line
    break, within the line, length and duration limits.
    """
    n = len(tokens)
    words = [token.text for token in tokens]
    spaces = [0 if token.glued else 1 for token in tokens]
    prefix = [0]
    for word, space in zip(words, spaces):
        prefix.append(prefix[-1] + len(word) + space)
    pair = _pair_costs(words)
    label_len = len(label) + 1 if label else 0
    ends = [
        _ends_sentence(words[k], words[k + 1] if k + 1 < n else None) for k in range(n)
    ]
    sentence_ends = [0]
    for end in ends:
        sentence_ends.append(sentence_ends[-1] + end)

    best = [0.0] + [math.inf] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        gap = tokens[j].start - tokens[j - 1].end if j < n else 0.0
        end_cost = _break_cost(words, j, gap)
        for i in range(j - 1, -1, -1):
            single = i == j - 1
            extra = label_len if i == 0 else 0
            length = prefix[j] - prefix[i] - spaces[i] + extra
            duration = tokens[j - 1].end - tokens[i].start
            if not single and (
                length > MAX_CAPTION_CHARS or duration > MAX_CAPTION_SECONDS
            ):
                break
            cost = best[i] + CAPTION_COST + end_cost
            if not single and length > MAX_LINE_CHARS:
                split, line_cost = _line_split(words, spaces, pair, i, j, extra)
                if split is None:
                    continue
                cost += LINE_BREAK_WEIGHT * line_cost
            if j < n and not ends[j - 1]:
                if length < 16:
                    cost += 0.8
                if duration < MIN_CAPTION_SECONDS:
                    cost += 0.4
                # Ends one sentence, then starts another it doesn't finish.
                if sentence_ends[j - 1] > sentence_ends[i]:
                    cost += 0.6
            if cost < best[j]:
                best[j], back[j] = cost, i

    spans = []
    j = n
    while j > 0:
        spans.append((back[j], j))
        j = back[j]
    return spans[::-1]


def _segment_tokens(start: float, end: float, text: str) -> List[_Token]:
    """Timed tokens for a segment, split after em dashes so lines can break there."""
    pieces: List[Tuple[str, bool]] = []
    for word in _tokens(text):
        parts = re.split(r"(?<=—)(?=[^\s—])", word)
        pieces.extend((part, k > 0) for k, part in enumerate(parts) if part)
    weights = [len(word) + 1 for word, _ in pieces]
    total = sum(weights) or 1
    tokens = []
    t = start
    for (word, glued), weight in zip(pieces, weights):
        span = (end - start) * weight / total
        tokens.append(_Token(word, t, t + span, glued))
        t += span
    return tokens


def _move(segment: _Segment, start: float) -> None:
    delta = start - segment.start
    segment.start += delta
    segment.end += delta
    for token in segment.tokens:
        token.start += delta
        token.end += delta


def _crosstalk_split(previous: _Segment, at: float, prior: Optional[str]) -> int:
    """
    Where to break `previous` for speech that starts at `at` inside it: the
    sentence or clause end closest to that time (within a few seconds), so an
    interjection like "Yes." follows the question it answers instead of
    cutting a sentence in half. `prior` is the word spoken just before
    `previous`; 0 means before its first word.
    """
    tokens = previous.tokens
    words = [token.text for token in tokens]

    def boundary(k: int) -> float:
        return tokens[k - 1].end if k else previous.start

    nearest = min(range(len(words) + 1), key=lambda k: abs(boundary(k) - at))
    best, best_cost = nearest, math.inf
    for k in range(len(words) + 1):
        distance = abs(boundary(k) - at)
        if distance > CROSSTALK_SHIFT_SECONDS and k != nearest:
            continue
        if k == 0:
            cost = _pair_break_cost(prior, words[0]) if prior else 0.6
        else:
            cost = _pair_break_cost(words[k - 1], words[k] if k < len(words) else None)
        cost += distance * 0.5
        if cost < best_cost:
            best, best_cost = k, cost
    return best


def _linearize(segments: List[_Segment]) -> List[_Segment]:
    """
    Orders segments in time so captions never overlap. Speech from another
    speaker that starts inside a segment (crosstalk) is moved to a nearby
    sentence or clause boundary of the interrupted speech, which resumes after
    it.
    """
    ordered: List[_Segment] = []
    for segment in sorted(segments, key=lambda s: s.start):
        if not ordered or segment.start >= ordered[-1].end - 0.05:
            ordered.append(segment)
            continue
        previous = ordered.pop()
        if segment.speaker == previous.speaker:
            k = len(previous.tokens)
        else:
            prior = ordered[-1].tokens[-1].text if ordered else None
            k = _crosstalk_split(previous, segment.start, prior)
        before, after = previous.tokens[:k], previous.tokens[k:]
        if before:
            ordered.append(
                _Segment(previous.start, before[-1].end, previous.speaker, before)
            )
        _move(segment, before[-1].end if before else previous.start)
        ordered.append(segment)
        if after:
            rest = _Segment(after[0].start, previous.end, previous.speaker, after)
            _move(rest, max(rest.start, segment.end))
            ordered.append(rest)
    return ordered


def _captions_from_segments(segments: List[_Segment]) -> List[_Caption]:
    runs: List[Tuple[str, bool, List[_Token]]] = []
    previous_speaker = None
    for segment in _linearize([s for s in segments if s.tokens]):
        if (
            runs
            and runs[-1][0] == segment.speaker
            and segment.tokens[0].start - runs[-1][2][-1].end < PAUSE_BREAK_SECONDS
        ):
            runs[-1][2].extend(segment.tokens)
        else:
            runs.append(
                (segment.speaker, segment.speaker != previous_speaker, segment.tokens)
            )
        previous_speaker = segment.speaker

    captions: List[_Caption] = []
    for speaker, labelled, tokens in runs:
        label = f"[{speaker}]" if labelled else ""
        for i, j in _caption_spans(tokens, label):
            words = [token.text for token in tokens[i:j]]
            spaces = [0 if token.glued else 1 for token in tokens[i:j]]
            captions.append(
                _Caption(
                    tokens[i].start,
                    tokens[j - 1].end,
                    tokens[j - 1].end,
                    _layout_lines(words, spaces, label if i == 0 else ""),
                    bool(label) and i == 0,
                )
            )
    return captions


def _merge_quick_exchanges(captions: List[_Caption]) -> List[_Caption]:
    """
    Joins a caption too short to read with the next caption from another
    speaker, one line each, as broadcast captions do for quick exchanges.
    """
    merged: List[_Caption] = []
    k = 0
    while k < len(captions):
        caption = captions[k]
        nxt = captions[k + 1] if k + 1 < len(captions) else None
        if (
            nxt
            and nxt.labelled
            and len(caption.lines) == 1
            and len(nxt.lines) == 1
            and nxt.start - caption.start < MIN_CAPTION_SECONDS
            and nxt.speech_end - caption.start <= MAX_CAPTION_SECONDS
        ):
            merged.append(
                _Caption(
                    caption.start,
                    nxt.end,
                    nxt.speech_end,
                    caption.lines + nxt.lines,
                    caption.labelled,
                )
            )
            k += 2
            continue
        merged.append(caption)
        k += 1
    return merged


def _finalize_timing(captions: List[_Caption]) -> None:
    """
    Keeps captions in order without overlaps, and holds each one up long enough
    to read (into following silence, never past the next caption). A caption
    with less than MIN_CAPTION_SECONDS before the next one (e.g. a quick "Yes."
    in crosstalk) delays the next caption a little, while it is still spoken.
    """
    for k in range(1, len(captions)):
        earliest = min(
            captions[k - 1].start + MIN_CAPTION_SECONDS, captions[k].speech_end - 0.5
        )
        captions[k].start = max(
            captions[k].start, earliest, captions[k - 1].start + 0.25
        )
    for k, caption in enumerate(captions):
        chars = sum(len(line) for line in caption.lines)
        needed = max(MIN_CAPTION_SECONDS, chars / READING_CHARS_PER_SECOND)
        end = max(caption.speech_end, caption.start + needed)
        if k + 1 < len(captions):
            next_start = captions[k + 1].start
            if next_start - end < CLOSE_GAP_SECONDS:
                end = next_start
            end = min(end, next_start)
        caption.end = max(end, caption.start + 0.25)


def format_captions(captions: List[_Caption]) -> str:
    blocks = [
        f"{number}\n{format_timestamp(caption.start)} --> "
        f"{format_timestamp(caption.end)}\n" + "\n".join(caption.lines)
        for number, caption in enumerate(captions, 1)
    ]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _place_addition(
    text: str,
    after: Cue,
    following: Optional[Cue],
    ref_words: List[_Word],
    ref_times: List[float],
) -> Tuple[float, float]:
    """
    Timing for speech the model added from the second transcript: the time of
    the same words there if they can be found nearby, else the gap after the
    cue it follows.
    """
    duration = max(MIN_CAPTION_SECONDS, len(text) / SPEECH_CHARS_PER_SECOND)
    norms = [_norm(t) for t in _tokens(text) if not t.startswith("[")]
    norms = [n for n in norms if n]
    lo = bisect_left(ref_times, after.start - 5)
    hi = bisect_right(ref_times, (following.end if following else after.end) + 5)
    window = ref_words[lo:hi]
    if norms and window:
        matcher = difflib.SequenceMatcher(
            None, norms, [w.norm for w in window], autojunk=False
        )
        blocks = [b for b in matcher.get_matching_blocks() if b.size]
        if blocks and sum(b.size for b in blocks) >= max(1, len(norms) // 2):
            first = window[blocks[0].b]
            last = window[blocks[-1].b + blocks[-1].size - 1]
            end = last.time + max(0.3, len(last.norm) / SPEECH_CHARS_PER_SECOND)
            return first.time, max(end, first.time + 0.3)
    start = after.end
    if following and following.start - start >= 0.3:
        return start, min(following.start, start + duration)
    return start, start + duration


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def correct_captions(
    model_name: str,
    srt_content: str,
    reference_srt: str = "",
    speaker_lists: Sequence[Tuple[str, str]] = (),
    video_context: str = "",
    video_title: str = "",
    source_name: str = "speech-to-text",
    reference_name: str = "YouTube captions",
) -> CorrectedCaptions:
    """
    Produces a complete corrected SRT from `srt_content`, using `reference_srt`
    (a second transcript of the same audio) to repair timing and recover words,
    `speaker_lists` ((source, text) pairs from speaker extraction) and
    `video_context` (on-screen names, terms and events) for names.
    """
    cues = prepare_cues(parse_srt(srt_content))
    if not cues:
        raise ValueError("the source SRT has no cues")
    reference = prepare_cues(parse_srt(reference_srt)) if reference_srt else []

    cue_words = _words(cues)
    ref_words = _words(reference)
    pairs = (
        _align([w.norm for w in cue_words], [w.norm for w in ref_words])
        if cue_words and ref_words
        else []
    )
    repaired = repair_timing(cues, cue_words, ref_words, pairs)
    add_reference_hints(cues, cue_words, ref_words, pairs)

    # Speech-to-text lists crosstalk after the turn it interrupts; captions run
    # in time order, so diarization turns are re-derived after sorting.
    cues.sort(key=lambda cue: cue.start)
    for k, cue in enumerate(cues):
        if cue.label:
            cue.turn_start = k == 0 or cue.label != cues[k - 1].label
    ref_words.sort(key=lambda word: word.time)
    ref_times = [word.time for word in ref_words]

    context = parse_video_context(video_context)
    rprint(f"Identifying speakers across {len(cues)} cues...")
    speakers, input_tokens, output_tokens = identify_speakers(
        model_name, cues, context, speaker_lists, video_title
    )

    rprint("Correcting caption text...")
    edits, additions, in_tok, out_tok = correct_sections(
        model_name,
        cues,
        speakers,
        context,
        speaker_lists,
        reference,
        video_title,
        source_name,
        reference_name,
    )
    input_tokens += in_tok
    output_tokens += out_tok

    segments: List[_Segment] = []
    fallback = added = 0
    for k, cue in enumerate(cues):
        lines = edits.get(k + 1)
        if lines is None:
            fallback += 1
            lines = [(speakers[k], _clean_text(cue.text, speakers[k]))]
        elif len(lines) == 1:
            # The speaker pass is authoritative unless the cue was split.
            lines = [(speakers[k], lines[0][1])]
        spoken = [(speaker, text) for speaker, text in lines if text]
        total = sum(len(text) for _, text in spoken)
        start = cue.start
        for n, (speaker, text) in enumerate(spoken):
            end = (
                cue.end
                if n == len(spoken) - 1
                else start + (cue.end - cue.start) * len(text) / total
            )
            segments.append(
                _Segment(start, end, speaker, _segment_tokens(start, end, text))
            )
            start = end
        following = cues[k + 1] if k + 1 < len(cues) else None
        for speaker, text in additions.get(k + 1, []):
            if text:
                start, end = _place_addition(text, cue, following, ref_words, ref_times)
                segments.append(
                    _Segment(start, end, speaker, _segment_tokens(start, end, text))
                )
                added += 1

    captions = _merge_quick_exchanges(_captions_from_segments(segments))
    _finalize_timing(captions)
    return CorrectedCaptions(
        srt=format_captions(captions),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        captions=len(captions),
        repaired_cues=repaired,
        fallback_cues=fallback,
        added_segments=added,
    )


# ---------------------------------------------------------------------------
# CLI step: pick the SRTs and context for a video, correct, save
# ---------------------------------------------------------------------------


def _find_srts(
    row: Dict[str, Any],
    storage: Any,
    srt_dir: str,
    video_id: str,
    safe_title: str,
    sources: Sequence[str],
) -> Dict[str, str]:
    """English SRT paths for the video by source ("youtube" or an STT model)."""
    found: Dict[str, str] = {}
    for key, value in row.items():
        if (
            key.startswith("SRT File ")
            and "(" not in key
            and isinstance(value, str)
            and value
            and storage.exists(value)
        ):
            found[key[len("SRT File ") :]] = value
    for source in sources:
        if source in found:
            continue
        prefixes = (
            ["human generated", "youtube generated"]
            if source == "youtube"
            else [f"{source} generated"]
        )
        for prefix in prefixes:
            path = os.path.join(srt_dir, f"{prefix} - {video_id} - {safe_title}.srt")
            if storage.exists(path):
                found[source] = path
                break
    return found


def _describe(source: str) -> str:
    return "YouTube captions" if source == "youtube" else f"{source} speech-to-text"


def process_corrected_captions(
    *,
    row: Dict[str, Any],
    storage: Any,
    model_name: str,
    source: Optional[str],
    transcript_arg: str,
    video_id: str,
    video_title: str,
    safe_title: str,
    url: str,
    dirs: Dict[str, str],
    verbose: bool,
    vprint: Callable[..., None] = print,
) -> Optional[Tuple[str, str]]:
    """
    Writes a corrected SRT for the video (`-cc`). `source` is the SRT to
    correct ("youtube", an STT model, or None for the most relevant AI SRT);
    the other transcript is used as the reference. Skips if the output exists.
    Returns (path, source used), or None if no captions were produced.
    """
    srts = _find_srts(
        row,
        storage,
        dirs["srt"],
        video_id,
        safe_title,
        [name for name in (source, transcript_arg, "youtube") if name],
    )
    ai_sources = sorted(
        (name for name in srts if name != "youtube"),
        key=lambda name: name != transcript_arg,
    )
    chosen = source if source in srts else None
    if chosen is None:
        chosen = (
            ai_sources[0] if ai_sources else ("youtube" if "youtube" in srts else None)
        )
        if chosen is None:
            print(f"Warning: No SRT found for {video_id}. Skipping corrected captions.")
            return None
        if source:
            print(
                f"Warning: No {source} SRT found for {video_id}; correcting the "
                f"{chosen} SRT instead."
            )
    if chosen == "youtube":
        reference = ai_sources[0] if ai_sources else None
    else:
        reference = "youtube" if "youtube" in srts else None

    col = f"Corrected Captions File {model_name} from {chosen}"
    path = os.path.join(
        dirs["corrected_captions"],
        f"{model_name} - {video_id} - {safe_title} - "
        f"corrected captions (from {chosen}).srt",
    )
    existing = row.get(col)
    if isinstance(existing, str) and existing and storage.exists(existing):
        vprint(f"Corrected captions already exist: {existing}")
        return existing, chosen
    if storage.exists(path):
        row[col] = path
        vprint(f"Corrected captions already exist: {path}")
        return path, chosen

    context = ensure_video_context(
        row=row,
        storage=storage,
        model_name=model_name,
        video_id=video_id,
        safe_title=safe_title,
        url=url,
        video_context_dir=dirs["video_context"],
        verbose=verbose,
        vprint=vprint,
    )
    speaker_lists = [
        (key[len("Speakers ") :], value)
        for key, value in row.items()
        if key.startswith("Speakers ")
        and not key.startswith("Speakers File ")
        and isinstance(value, str)
        and value.strip() not in ("", "nan", 'float("nan")')
    ]
    rprint(
        f"Correcting captions using {model_name} (source: {chosen}"
        + (f", reference: {reference}" if reference else "")
        + (", with on-screen context" if context else "")
        + ")..."
    )
    try:
        result = correct_captions(
            model_name,
            storage.read_text(srts[chosen]),
            reference_srt=storage.read_text(srts[reference]) if reference else "",
            speaker_lists=speaker_lists,
            video_context=context,
            video_title=video_title,
            source_name=_describe(chosen),
            reference_name=_describe(reference) if reference else "",
        )
    except Exception as e:
        print(f"Error correcting captions for {video_id}: {e}")
        return None

    try:
        saved = storage.write_text(path, result.srt)
    except Exception as e:
        print(f"Error writing corrected captions: {e}")
        return None
    row[col] = saved
    details = [f"{result.captions} captions"]
    if result.repaired_cues:
        details.append(f"{result.repaired_cues} cues re-timed")
    if result.added_segments:
        details.append(f"{result.added_segments} missed lines restored")
    if result.fallback_cues:
        details.append(f"{result.fallback_cues} cues left uncorrected")
    rprint(
        f"Saved corrected captions: {format_clickable_path(saved)} "
        f"({', '.join(details)})"
    )
    if verbose:
        input_price, output_price = get_model_pricing(model_name)
        if input_price is not None and output_price is not None:
            cost = (result.input_tokens / 1_000_000) * input_price + (
                result.output_tokens / 1_000_000
            ) * output_price
            row[
                f"{normalize_model_name(model_name)} corrected captions cost "
                f"from {chosen} ($)"
            ] = round(cost, 2)
            vprint(f"Corrected captions cost: ${cost:.2f}")
    return saved, chosen
