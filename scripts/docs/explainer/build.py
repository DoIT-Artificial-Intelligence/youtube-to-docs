"""Builds the explainer video on the How This Works page.

Each slide is an HTML page rendered to a 1920x1080 image with headless
Chrome, so the on-screen text is exact. The narration is read one sentence
at a time with Gemini TTS, which gives every caption cue its exact start and
end. Outputs go to docs/assets/explainer/:

- explainer.mp4: the slides with the narration
- captions.vtt: one caption cue per sentence
- transcript.md: the narration, included on the page
- poster.webp: the first slide, shown before the video plays

TTS audio is cached in a work directory, so changing a slide's design
doesn't regenerate the speech. Run with:

    uv run --extra all scripts/docs/explainer/build.py
"""

import argparse
import hashlib
import html
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field

from youtube_to_docs.tts import generate_speech

TTS_MODEL = "gemini-3.8-flash-tts"
TTS_VOICE = "Kore"
WIDTH, HEIGHT = 1920, 1080
# Pauses between sentences and between slides, in seconds.
SENTENCE_GAP = 0.35
SLIDE_GAP = 0.9
# WCAG-style caption layout: at most two lines of 42 characters.
CAPTION_LINE = 42

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../docs/assets/explainer")
)


@dataclass
class Line:
    """One narrated sentence. `caption` is shown when it differs from speech."""

    spoken: str
    caption: str = ""

    @property
    def text(self) -> str:
        return self.caption or self.spoken


@dataclass
class Slide:
    title: str
    body: str
    lines: list[Line] = field(default_factory=list)
    step: str = ""
    cls: str = ""


def chips(*items: str) -> str:
    return (
        '<div class="chips">' + "".join(f"<span>{i}</span>" for i in items) + "</div>"
    )


def cards(*items: tuple[str, str, str]) -> str:
    return (
        '<div class="cards">'
        + "".join(
            f'<div class="card"><div class="icon">{icon}</div>'
            f"<h3>{title}</h3><p>{text}</p></div>"
            for icon, title, text in items
        )
        + "</div>"
    )


SLIDES = [
    Slide(
        cls="title",
        title="How youtube-to-docs works",
        body=(
            '<p class="lede">Hours of video. '
            '<span class="grad">Minutes to read.</span></p>'
            '<p class="sub">Turn any YouTube video, playlist or channel into '
            "transcripts, Q&amp;A, summaries, infographics, audio and accessible "
            "captions.</p>"
        ),
        lines=[
            Line(
                "YouTube to docs turns videos into documents to search, share "
                "and hear.",
                "youtube-to-docs turns videos into documents to search, share "
                "and hear.",
            ),
            Line("Here's how it works, in six steps."),
        ],
    ),
    Slide(
        step="1",
        title="Point it at YouTube",
        body=chips("Video URL", "Video ID", "Playlist", "@channel")
        + '<pre class="cmd">$ uvx youtube-to-docs @mga-hgo1740</pre>'
        + '<p class="out">Playlists and channels expand into every video</p>',
        lines=[
            Line("Start with a video link, a video ID, a playlist or a whole channel."),
            Line("Playlists and channels are expanded, and each video runs in turn."),
        ],
    ),
    Slide(
        step="2",
        title="Get the words",
        body=cards(
            ("▶", "YouTube captions", "The default. Free and fast."),
            (
                "🎙",
                "Speech-to-text",
                "Gemini, Google Cloud Chirp 3 or AWS Transcribe, with word "
                "timings and speaker labels.",
            ),
        )
        + '<p class="out">Output: a transcript and a timed SRT file</p>',
        lines=[
            Line("Next, it gets the words."),
            Line("By default, it uses YouTube's own captions."),
            Line("Or it transcribes the audio again with Gemini, Google Cloud or AWS."),
            Line("That gives real word timings and speaker labels."),
            Line("Either way, you get a transcript and a timed SRT file."),
        ],
    ),
    Slide(
        step="3",
        title="Let the model read",
        body=cards(
            ("👥", "Speakers", "Names and titles"),
            ("💬", "Q&amp;A", "Who asked, who answered, and when"),
            ("📝", "Summaries", "Full and one-sentence"),
            ("🏷", "Tags", "Topics for search"),
        )
        + chips("Gemini", "Vertex AI", "AWS Bedrock", "Azure Foundry"),
        lines=[
            Line("Then a language model of your choice reads the transcript."),
            Line("It can run on Gemini, Vertex AI, AWS Bedrock or Azure Foundry."),
            Line("It names the speakers and their titles."),
            Line("It pairs each question with its answer and a timestamp link."),
            Line("It writes a summary, a one-sentence summary and topic tags."),
            Line("Gemini can also watch the video, to read slides and name plates."),
        ],
    ),
    Slide(
        step="4",
        title="Turn it into media",
        body=cards(
            ("🖼", "Infographic", "16:9, with alt text"),
            ("🎧", "Audio summary", "The summary, read aloud"),
            ("🎬", "Video", "The infographic and audio together"),
            ("🌐", "Any language", "Everything again in a second language"),
        ),
        lines=[
            Line("From the summary, it can draw an infographic."),
            Line("Another model looks at the image and writes its alt text."),
            Line("It can read the summary aloud, and combine both into a video."),
            Line("And it can make everything again in a second language."),
        ],
    ),
    Slide(
        step="5",
        title="Make the video accessible",
        body='<pre class="cmd">$ youtube-to-docs &lt;url&gt; -cc</pre>'
        + cards(
            (
                "CC",
                "Corrected captions",
                "WCAG 2.1 AA and Section 508. Every speaker named.",
            ),
            (
                "AD",
                "Audio description",
                "Slides, charts and on-screen text described in the pauses.",
            ),
        ),
        lines=[
            Line(
                "Add the C C flag to make the original video accessible.",
                "Add the -cc flag to make the original video accessible.",
            ),
            Line("Captions are corrected, and each speaker is named."),
            Line(
                "They meet WCAG 2.1 Level double A and Section 508.",
                "They meet WCAG 2.1 Level AA and Section 508.",
            ),
            Line("Slides and on-screen text are described for blind viewers."),
        ],
    ),
    Slide(
        step="6",
        title="You stay in control",
        body=cards(
            ("✎", "Edit any output", "Transcripts, captions, Q&amp;A and summaries"),
            ("↻", "Reruns keep edits", "Later steps use your corrected files"),
            ("$", "Costs recorded", "One CSV lists every file and its cost"),
        )
        + chips("Your computer", "Google Drive", "SharePoint", "Hugging Face"),
        lines=[
            Line("Every output is a file that you can review and edit."),
            Line("Run it again, and the later steps use your edits."),
            Line("A CSV lists every file and what each step cost."),
            Line(
                "Save it all to your computer, Google Drive, SharePoint or "
                "Hugging Face."
            ),
        ],
    ),
    Slide(
        title="Get started",
        body='<pre class="cmd">$ uvx --with "youtube-to-docs[all]" \\\n'
        "    youtube-to-docs &lt;url&gt; -a gemini-pro -cc</pre>"
        + chips("Command line", "Web app", "MCP server", "Claude Code plugin")
        + '<p class="sub">doit-artificial-intelligence.github.io/youtube-to-docs</p>',
        lines=[
            Line(
                "Run it from the command line, the web app, or as a plugin "
                "for your AI assistant."
            ),
            Line("One command takes you from a link to a full set of documents."),
        ],
    ),
]

CSS = """
* { box-sizing: border-box; margin: 0; }
body {
  width: 1920px; height: 1080px; overflow: hidden;
  font-family: Inter, "DejaVu Sans", sans-serif; color: #e8ebf5;
  background: radial-gradient(circle at 85% 10%, #1e2a5a 0%, #0b1020 55%);
  /* The extra bottom padding keeps slide text clear of the captions. */
  padding: 150px 150px 230px; display: flex; flex-direction: column;
  justify-content: center; gap: 56px;
}
body::after {
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 10px;
  background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981, #0ea5e9, #8b5cf6);
}
.brand {
  position: absolute; top: 64px; left: 150px;
  font-size: 32px; color: #a8b3cf; font-weight: 600; letter-spacing: .02em;
}
.title h1 { font-size: 64px; color: #a5b4fc; }
.title .lede { font-size: 112px; white-space: nowrap; }
.title .sub { font-size: 46px; }
.step {
  display: inline-flex; align-items: center; justify-content: center;
  width: 92px; height: 92px; border-radius: 50%; background: #4f46e5;
  font-size: 50px; font-weight: 800; margin-right: 32px; vertical-align: middle;
}
h1 { font-size: 96px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.lede {
  font-size: 104px; font-weight: 800; letter-spacing: -0.03em; line-height: 1.05;
}
.grad {
  background: linear-gradient(90deg, #f59e0b, #10b981, #0ea5e9, #8b5cf6);
  -webkit-background-clip: text; color: transparent;
}
.sub { font-size: 40px; color: #c3cbe0; max-width: 1500px; line-height: 1.4; }
.chips { display: flex; flex-wrap: wrap; gap: 22px; }
.chips span {
  font-size: 40px; font-weight: 600; padding: 18px 34px; border-radius: 999px;
  background: #161d33; border: 2px solid #3b4670;
}
.cards {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(0, 1fr)); gap: 32px;
}
.card {
  background: #161d33; border: 2px solid #273049; border-radius: 28px;
  padding: 44px 42px;
}
.card .icon {
  font-size: 58px; font-weight: 800; width: 104px; height: 104px; border-radius: 24px;
  background: #262f5c; display: flex; align-items: center; justify-content: center;
  margin-bottom: 26px; color: #c7d2fe;
}
.card h3 { font-size: 48px; font-weight: 700; margin-bottom: 16px; }
.card p { font-size: 36px; color: #c3cbe0; line-height: 1.4; }
.cmd {
  font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; font-size: 44px;
  background: #060912; border: 2px solid #273049; border-radius: 18px;
  padding: 30px 38px; color: #a5f3c4; white-space: pre-wrap;
}
.out { font-size: 40px; color: #a5b4fc; font-weight: 600; }
"""


def slide_html(slide: Slide) -> str:
    step = f'<span class="step">{slide.step}</span>' if slide.step else ""
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<style>{CSS}</style></head><body class="{slide.cls}">'
        f'<div class="brand">youtube-to-docs</div>'
        f"<h1>{step}{slide.title}</h1>{slide.body}</body></html>"
    )


def render(slide: Slide, path: str, chrome: str, work: str) -> None:
    page = os.path.join(work, os.path.basename(path).replace(".png", ".html"))
    with open(page, "w", encoding="utf-8") as f:
        f.write(slide_html(slide))
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size={WIDTH},{HEIGHT}",
            f"--screenshot={path}",
            f"file://{page}",
        ],
        check=True,
        capture_output=True,
    )


def trim_silence(pcm: bytes, threshold: int = 300) -> bytes:
    """Strips near-silent samples from both ends of 16-bit mono PCM."""
    samples = memoryview(pcm).cast("h")
    start, end = 0, len(samples)
    while start < end and abs(samples[start]) < threshold:
        start += 1
    while end > start and abs(samples[end - 1]) < threshold:
        end -= 1
    # Keep 50 ms either side so word edges aren't clipped.
    pad = 1200
    return bytes(samples[max(0, start - pad) : min(len(samples), end + pad)])


def speech(text: str, work: str) -> tuple[bytes, int]:
    """Returns the PCM and sample rate for one sentence, cached on disk."""
    key = hashlib.sha256(f"{TTS_MODEL}|{TTS_VOICE}|{text}".encode()).hexdigest()[:16]
    cache = os.path.join(work, f"tts-{key}.wav")
    if not os.path.exists(cache):
        pcm, rate = generate_speech(text, TTS_MODEL, TTS_VOICE)
        if not pcm:
            raise RuntimeError(f"TTS failed for: {text}")
        with wave.open(cache, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(trim_silence(pcm))
    with wave.open(cache, "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


def timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def caption_lines(text: str) -> str:
    """Breaks a caption into at most two lines of CAPTION_LINE characters."""
    if len(text) <= CAPTION_LINE:
        return text
    words = text.split()
    best = None
    for i in range(1, len(words)):
        first, second = " ".join(words[:i]), " ".join(words[i:])
        if len(first) <= CAPTION_LINE and len(second) <= CAPTION_LINE:
            balance = abs(len(first) - len(second))
            if best is None or balance < best[0]:
                best = (balance, f"{first}\n{second}")
    if best is None:
        raise ValueError(f"Caption is longer than two lines: {text}")
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chrome", default=os.environ.get("CHROME", "google-chrome"))
    parser.add_argument(
        "--work-dir",
        default=os.path.join(tempfile.gettempdir(), "ytd-explainer"),
        help="Where slides and cached TTS audio are kept between runs.",
    )
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    from static_ffmpeg import run

    ffmpeg, _ = run.get_or_fetch_platform_executables_else_raise()

    audio = bytearray()
    rate = 24000
    cues: list[tuple[float, float, str]] = []
    concat = []
    transcript = []

    for n, slide in enumerate(SLIDES):
        image = os.path.join(args.work_dir, f"slide-{n:02}.png")
        render(slide, image, args.chrome, args.work_dir)
        slide_start = len(audio) / 2 / rate
        heading = f"{slide.step}. {slide.title}" if slide.step else slide.title
        transcript.append(
            f"**{heading}**\n\n" + " ".join(line.text for line in slide.lines)
        )

        for i, line in enumerate(slide.lines):
            pcm, rate = speech(line.spoken, args.work_dir)
            start = len(audio) / 2 / rate
            audio += pcm
            end = len(audio) / 2 / rate
            cues.append((start, end, caption_lines(line.text)))
            gap = SLIDE_GAP if i == len(slide.lines) - 1 else SENTENCE_GAP
            audio += b"\x00\x00" * round(gap * rate)

        concat.append(
            f"file '{image}'\nduration {len(audio) / 2 / rate - slide_start:.3f}"
        )

    # The concat demuxer needs the last image repeated to hold its duration.
    concat.append(f"file '{image}'")
    list_path = os.path.join(args.work_dir, "slides.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(concat) + "\n")
    wav_path = os.path.join(args.work_dir, "narration.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(audio))

    video = os.path.join(OUT_DIR, "explainer.mp4")
    subprocess.run(
        [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-i", wav_path, "-c:v", "libx264", "-tune", "stillimage",
            "-r", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-t", f"{len(audio) / 2 / rate:.3f}", "-movflags", "+faststart", video,
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    subprocess.run(
        [ffmpeg, "-y", "-i", os.path.join(args.work_dir, "slide-00.png"),
         "-vf", "scale=1280:-1", os.path.join(OUT_DIR, "poster.webp")],
        check=True,
        capture_output=True,
    )  # fmt: skip

    with open(os.path.join(OUT_DIR, "captions.vtt"), "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for start, end, text in cues:
            f.write(f"{timestamp(start)} --> {timestamp(end)}\n{text}\n\n")
    with open(os.path.join(OUT_DIR, "transcript.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(html.unescape(t) for t in transcript) + "\n")

    print(f"Wrote {video} ({len(audio) / 2 / rate:.1f} s, {len(cues)} captions)")


if __name__ == "__main__":
    main()
