"""MkDocs hook: build the Pricing page from youtube_to_docs/prices.py.

The page has a `<!-- PRICING_DATA -->` marker and `{{ updated_at }}` placeholders.
At build time they are replaced with the prices as JSON (plus each model's type
and its price in the vendor's own unit), so the page never drifts from the
prices the CLI uses to estimate costs.
"""

import json
import re

from youtube_to_docs.prices import PRICES

# Assumptions used to turn token prices into per-video and per-hour estimates.
# Shown on the page next to the estimates.
TRANSCRIPT_TOKENS_PER_MINUTE = 200  # Measured: 37,663 tokens for 3h11m of speech.
SUMMARY_OUTPUT_TOKENS = 2_000  # Benchmark summaries ran 800-2,900 tokens.
SPOKEN_CHARS_PER_MINUTE = 900  # About 150 words per minute.
AUDIO_TOKENS_PER_SECOND = 32  # Gemini audio input and output.
TOKENS_PER_CHAR = 0.25  # Starred character prices: 1M tokens ~ 4M characters.
TOKENS_PER_MINUTE_STARRED = 200  # Starred per-minute prices: 1M tokens ~ 5,000 min.

# Real cost of one infographic from a gemini-4-argon summary of a 3h49m hearing
# (bnjjmvy1RlU), read from the cost columns of youtube-docs.csv.
MEASURED_INFOGRAPHIC_COST = {
    "gemini-3-pro-image": 0.26,
    "gemini-nano-banana-2.1": 0.09,
    "gemini-3.1-flash-lite-image": 0.07,
}


def model_type(item: dict) -> str:
    name = item["name"].lower()
    if any(k in name or k in item["id"] for k in ("image", "canvas", "banana")):
        return "image"
    if "tts" in name or "polly" in name:
        return "tts"
    if "stt" in name or "transcribe" in name:
        return "stt"
    if "translat" in name or "translat" in item["id"]:
        return "translate"
    return "text"


def enrich(item: dict) -> dict:
    kind = model_type(item)
    starred = item["name"].endswith("*")
    total = (item["input"] or 0) + (item["output"] or 0)
    out = {
        "id": item["id"],
        "vendor": item["vendor"],
        # The tab already says the type, so drop suffixes like "(STT)".
        "name": re.sub(r" \((STT|TTS|image gen)\)$", "", item["name"].rstrip("*")),
        "type": kind,
        "input": item["input"],
        "output": item["output"],
        "cached": item.get("input_cached"),
        "note": item.get("note"),
        "note_label": "tiered" if "billed at" in (item.get("note") or "") else "promo",
        "converted": starred,
    }
    if kind == "stt":
        if starred:
            per_minute = total * TOKENS_PER_MINUTE_STARRED / 1_000_000
            out["unit_price"], out["unit"] = per_minute, "per minute"
            out["per_hour"] = per_minute * 60
        else:
            audio_in = AUDIO_TOKENS_PER_SECOND * 3600 * item["input"]
            text_out = TRANSCRIPT_TOKENS_PER_MINUTE * 60 * item["output"]
            out["per_hour"] = (audio_in + text_out) / 1_000_000
    elif kind == "tts":
        chars_per_hour = SPOKEN_CHARS_PER_MINUTE * 60
        if starred:
            per_m_chars = total * TOKENS_PER_CHAR
            out["unit_price"], out["unit"] = per_m_chars, "per 1M characters"
            out["per_hour"] = per_m_chars * chars_per_hour / 1_000_000
        else:
            text_in = chars_per_hour * TOKENS_PER_CHAR * item["input"]
            audio_out = AUDIO_TOKENS_PER_SECOND * 3600 * item["output"]
            out["per_hour"] = (text_in + audio_out) / 1_000_000
    elif kind == "translate":
        per_m_chars = total * TOKENS_PER_CHAR
        out["unit_price"], out["unit"] = per_m_chars, "per 1M characters"
        out["per_hour"] = per_m_chars * SPOKEN_CHARS_PER_MINUTE * 60 / 1_000_000
    elif kind == "image":
        out["measured"] = MEASURED_INFOGRAPHIC_COST.get(item["id"])
    return out


def on_page_markdown(markdown, page, **kwargs):
    if page.file.src_path != "pricing.md":
        return markdown
    data = {
        "updated_at": PRICES["updated_at"],
        "assumptions": {
            "tokens_per_minute": TRANSCRIPT_TOKENS_PER_MINUTE,
            "summary_output_tokens": SUMMARY_OUTPUT_TOKENS,
        },
        "prices": [enrich(p) for p in PRICES["prices"]],
    }
    payload = json.dumps(data).replace("</", "<\\/")
    script = f'<script id="pricing-data" type="application/json">{payload}</script>'
    return markdown.replace("<!-- PRICING_DATA -->", script).replace(
        "{{ updated_at }}", PRICES["updated_at"]
    )
