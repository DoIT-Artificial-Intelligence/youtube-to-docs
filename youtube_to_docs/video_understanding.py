"""
Summaries, speakers and Q&A generated directly from the video (`--from-video`),
using Gemini video understanding through the Interactions API.

Outputs are stored under the source name `video-{mode}` (e.g. "Summary Text
gemini-3.5-flash-lite from video-agentic"), next to the transcript-based outputs,
so the infographic and TTS steps pick them up like any other summary.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich import print as rprint

from youtube_to_docs.llms import generate_one_sentence_summary, get_model_pricing
from youtube_to_docs.utils import (
    add_question_numbers,
    format_clickable_path,
    normalize_model_name,
)

VIDEO_MODES = ("agentic", "static")

# Long videos can take many minutes in agentic mode, so requests run in the
# background and are polled until they finish or this many seconds pass.
VIDEO_TIMEOUT_SECONDS = 30 * 60
VIDEO_POLL_SECONDS = 5


def parse_video_modes(arg: Optional[str]) -> List[str]:
    """Parses `--from-video` (e.g. 'agentic' or 'agentic,static')."""
    if not arg:
        return []
    modes = [m.strip().lower() for m in arg.split(",") if m.strip()]
    invalid = [m for m in modes if m not in VIDEO_MODES]
    if invalid:
        raise ValueError(
            f"Invalid --from-video mode(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(VIDEO_MODES)}."
        )
    return modes


def video_mode_error(model_name: str) -> Optional[str]:
    """
    Returns why `model_name` can't process video, or None if it can. Agentic
    support (Gemini 3.5 Flash-Lite and later) is left to the API to enforce, so
    newer models work without code changes.
    """
    if not model_name.startswith("gemini-"):
        return (
            f"--from-video requires a Gemini API model (gemini-*); "
            f"skipping {model_name}."
        )
    return None


class VideoSession:
    """
    A stateful Interactions API conversation about one video. The first `ask`
    sends the video; later calls continue via `previous_interaction_id`, so the
    server keeps the video context instead of processing it again.
    """

    def __init__(self, model_name: str, video_url: str, mode: str):
        from google import genai

        self.model_name = model_name
        self.video_url = video_url
        self.mode = mode
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.previous_id: Optional[str] = None

    def ask(self, prompt: str) -> Tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens)."""
        text_part = {"type": "text", "text": prompt}
        kwargs: Dict[str, Any] = {"model": self.model_name, "background": True}
        if self.previous_id:
            kwargs["input"] = [text_part]
            kwargs["previous_interaction_id"] = self.previous_id
        else:
            # Place the prompt after the video, per Gemini's prompting guidance.
            video_part = {
                "type": "video",
                "uri": self.video_url,
                "processing": self.mode,
            }
            kwargs["input"] = [video_part, text_part]

        interaction = self.client.interactions.create(**kwargs)
        deadline = time.time() + VIDEO_TIMEOUT_SECONDS
        while interaction.status in ("in_progress", "queued"):
            if time.time() > deadline:
                return (
                    f"Error: video request timed out after "
                    f"{VIDEO_TIMEOUT_SECONDS // 60} minutes",
                    0,
                    0,
                )
            time.sleep(VIDEO_POLL_SECONDS)
            interaction = self.client.interactions.get(interaction.id)

        if interaction.status != "completed":
            return f"Error: video request ended with status {interaction.status}", 0, 0

        self.previous_id = interaction.id
        input_tokens, output_tokens = _usage_tokens(interaction)
        return interaction.output_text or "", input_tokens, output_tokens


def _usage_tokens(interaction: Any) -> Tuple[int, int]:
    """
    Billable (input, output) tokens. In agentic mode most tokens are navigation
    reasoning (thought tokens, billed as output) and content loaded on demand
    (tool use tokens, billed as input), so both must be counted.
    """
    usage = getattr(interaction, "usage", None)
    if not usage:
        return 0, 0
    total_input = (getattr(usage, "total_input_tokens", 0) or 0) + (
        getattr(usage, "total_tool_use_tokens", 0) or 0
    )
    total_output = (getattr(usage, "total_output_tokens", 0) or 0) + (
        getattr(usage, "total_thought_tokens", 0) or 0
    )
    return total_input, total_output


def build_video_summary_prompt(video_title: str, url: str, language: str) -> str:
    return f"Can you please summarize this video ({video_title}, {url}) in {language}?"


def build_video_speakers_prompt() -> str:
    return (
        "Can you please identify the speakers in this video? Use names and titles "
        "from on-screen captions, name plates, slides or introductions where "
        "available."
        "\n\n"
        "The output should be a markdown string in English like"
        "\n\n"
        "Speaker 1 (title)"
        "\n"
        "Speaker 2 (title)"
        "\n"
        "etc."
        "\n\n"
        "If the speaker is unknown use the placeholder UNKNOWN and if the title "
        "is unknown use the placeholder UNKNOWN. "
        'If No speaker(s) are detected set it to float("nan"). '
        "Return only the list."
    )


def build_video_context_prompt() -> str:
    return (
        "List the information that is shown on screen in this video, for use in "
        "correcting its captions. Use exactly these three markdown sections:"
        "\n\n"
        "## People\n"
        "One line per person who speaks or is named on screen: their full name "
        "and title exactly as shown on name plates, captions or slides, then the "
        "MM:SS (or HH:MM:SS) times when they are speaking, e.g. "
        "`- Sen. Jane Doe (Vice Chair) — speaking at 03:07, 12:40, 31:02`."
        "\n\n"
        "## Names and terms\n"
        "Exact on-screen spellings of names, titles, bill numbers, agencies, "
        "places, organizations and acronyms, one per line."
        "\n\n"
        "## On-screen events\n"
        "Timestamped information that is visible but may not be spoken, such as "
        "vote tallies, slide titles or documents shown, e.g. "
        "`- 14:22 — Vote board: 6 yes, 3 no`."
        "\n\n"
        "Only include what you can see or hear in the video. Return only the three "
        "sections."
    )


def build_video_qa_prompt(url: str, language: str) -> str:
    return (
        f"Can you please extract the questions and answers from this video in "
        f"{language}?"
        "\n\n"
        "The output should be a markdown table like:"
        "\n\n"
        "| questioner(s) | question | responder(s) | answer | "
        "timestamp | timestamp url |"
        "\n"
        "|---|---|---|---|---|---|"
        "\n"
        "| Speaker 1 | What is... | Speaker 2 | It is... | 01:23 | "
        "[Link](https://youtu.be/...&t=83) |\n"
        "\n\n"
        "If the questioner or responder is unknown use the placeholder UNKNOWN. "
        "Use people's names and titles (from the speakers you identified) in the "
        "questioner and responder fields. "
        'If no Q&A pairs are detected set it to float("nan"). '
        "For the 'timestamp' column, use the time in the video in the format "
        "MM:SS or HH:MM:SS. "
        "For the 'timestamp url' column, use the base YouTube URL provided below "
        "and append the timestamp in seconds (e.g. &t=123 or ?t=123). "
        "Format this column as a markdown hyperlink with the text 'Link'. "
        "If the base URL already contains a '?', use '&t=' otherwise use '?t='. "
        f"Base URL: {url}"
        "\n\n"
        "Return only the table."
    )


def process_from_video(
    *,
    row: Dict[str, Any],
    storage: Any,
    model_name: str,
    modes: List[str],
    video_id: str,
    video_title: str,
    safe_title: str,
    url: str,
    language: str,
    col_suffix: str,
    lang_str: str,
    dirs: Dict[str, str],
    verbose: bool,
    vprint: Callable[..., None] = print,
) -> None:
    """Fills `row` with video-sourced summary, one-sentence summary, speakers,
    on-screen context (used by -scc) and Q&A for each requested mode, skipping
    outputs that already exist."""
    for mode in modes:
        error = video_mode_error(model_name)
        if error:
            rprint(f"[yellow]{error}[/yellow]")
            continue

        source = f"video-{mode}"
        session: Optional[VideoSession] = None

        def cost_col(kind: str) -> str:
            return (
                f"{normalize_model_name(model_name)} {kind} cost from "
                f"{source}{col_suffix} ($)"
            )

        def record_cost(kind: str, input_tokens: int, output_tokens: int) -> None:
            if not verbose:
                return
            input_price, output_price = get_model_pricing(model_name)
            if input_price is None or output_price is None:
                return
            cost = (input_tokens / 1_000_000) * input_price + (
                output_tokens / 1_000_000
            ) * output_price
            row[cost_col(kind)] = round(cost, 4)
            vprint(
                f"{kind.capitalize()} cost ({source}): ${cost:.4f} "
                f"({input_tokens} input / {output_tokens} output tokens)"
            )

        def ask(prompt: str) -> Tuple[str, int, int]:
            nonlocal session
            if session is None:
                session = VideoSession(model_name, url, mode)
            try:
                return session.ask(prompt)
            except Exception as e:
                return f"Error: {e}", 0, 0

        def save(directory: str, kind: str, text: str) -> str:
            filename = (
                f"{model_name} - {video_id} - {safe_title} - "
                f"{kind} (from {source}){lang_str}"
                + (".txt" if kind == "speakers" else ".md")
            )
            try:
                saved = storage.write_text(os.path.join(directory, filename), text)
                rprint(f"Saved {kind} ({source}): {format_clickable_path(saved)}")
                return saved
            except Exception as e:
                print(f"Error writing {kind} ({source}): {e}")
                return ""

        # --- Summary (first request: loads the video) ---
        sum_col = f"Summary Text {model_name} from {source}{col_suffix}"
        sum_file_col = f"Summary File {model_name} from {source}{col_suffix}"
        if row.get(sum_file_col) and not row.get(sum_col):
            try:
                row[sum_col] = storage.read_text(str(row[sum_file_col]))
            except Exception as e:
                vprint(f"Warning: Failed to read {row[sum_file_col]}: {e}")
        if not row.get(sum_col):
            prompt = build_video_summary_prompt(video_title, url, language)
            prompt_col = f"Summary Prompt Path {model_name} from {source}{col_suffix}"
            if not row.get(prompt_col):
                row[prompt_col] = save(
                    dirs["summary_prompts"], "summary-prompt", prompt
                )
            rprint(
                f"Generating summary using model: {model_name} "
                f"(Source: video, {mode} processing)"
            )
            start = time.time()
            text, input_tokens, output_tokens = ask(prompt)
            vprint(f"Video summary took {time.time() - start:.0f}s")
            if text.startswith("Error"):
                print(f"Video summary failed ({source}): {text}")
                continue
            row[sum_col] = text
            row[sum_file_col] = save(dirs["summaries"], "summary", text)
            record_cost("summary", input_tokens, output_tokens)

        # --- One sentence summary (from the summary text; no video needed) ---
        os_col = f"One Sentence Summary {model_name} from {source}{col_suffix}"
        if row.get(sum_col) and not row.get(os_col):
            text, input_tokens, output_tokens = generate_one_sentence_summary(
                model_name, row[sum_col], language=language
            )
            row[os_col] = text
            if text and not text.startswith("Error"):
                row[
                    f"One Sentence Summary File {model_name} from {source}{col_suffix}"
                ] = save(dirs["one_sentence"], "one-sentence-summary", text)
            record_cost("one sentence summary", input_tokens, output_tokens)

        # --- Speakers (follow-up in the same video session) ---
        # Speakers are language independent, matching the transcript sources.
        spk_col = f"Speakers {model_name} from {source}"
        if language == "en" and not row.get(spk_col):
            rprint(f"Extracting speakers using model: {model_name} (Source: video)")
            text, input_tokens, output_tokens = ask(build_video_speakers_prompt())
            if not text.startswith("Error"):
                row[spk_col] = text
                row[f"Speakers File {model_name} from {source}"] = save(
                    dirs["speakers"], "speakers", text
                )
                record_cost("speaker extraction", input_tokens, output_tokens)
            else:
                print(f"Video speaker extraction failed ({source}): {text}")

        # --- On-screen context for caption correction (-scc) ---
        # Language independent, like speakers.
        ctx_col = f"Video Context {model_name} from {source}"
        if language == "en" and not row.get(ctx_col):
            rprint(f"Extracting on-screen context using model: {model_name}")
            text, input_tokens, output_tokens = ask(build_video_context_prompt())
            if not text.startswith("Error"):
                row[ctx_col] = text
                row[f"Video Context File {model_name} from {source}"] = save(
                    dirs["video_context"], "video-context", text
                )
                record_cost("video context", input_tokens, output_tokens)
            else:
                print(f"Video context extraction failed ({source}): {text}")

        # --- Q&A (follow-up in the same video session) ---
        qa_col = f"QA Text {model_name} from {source}{col_suffix}"
        if not row.get(qa_col):
            rprint(f"Generating Q&A using model: {model_name} (Source: video)")
            text, input_tokens, output_tokens = ask(
                build_video_qa_prompt(url, language)
            )
            if not text.startswith("Error"):
                if "|" in text:
                    text = add_question_numbers(text)
                row[qa_col] = text
                row[f"QA File {model_name} from {source}{col_suffix}"] = save(
                    dirs["qa"], "qa", text
                )
                record_cost("qa", input_tokens, output_tokens)
            else:
                print(f"Video Q&A failed ({source}): {text}")
