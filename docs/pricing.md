---
title: Pricing
hide:
  - navigation
  - toc
description: What it costs to turn a YouTube video into documents with each AI model, plus the full price list for text, image, speech and translation models.
---

# Pricing

<p class="ytd-lede">youtube-to-docs is free and open source. You pay only your AI provider, at their list prices, and every run records its estimated cost in the output CSV. Prices last updated <strong>{{ updated_at }}</strong>.</p>

<div class="ytd-kpis">
  <div class="ytd-kpi"><span>Summarize a 1-hour video</span><b id="kpi-cheapest">&nbsp;</b><small id="kpi-cheapest-model">&nbsp;</small></div>
  <div class="ytd-kpi"><span>Real run: a 3h49m hearing</span><b>$1.01</b><small>Gemini 4 Argon: speakers, Q&amp;A, summary, tags and an infographic</small></div>
  <div class="ytd-kpi"><span>Transcribe an hour of audio</span><b id="kpi-stt">&nbsp;</b><small id="kpi-stt-model">&nbsp;</small></div>
</div>

## What a video costs

<p class="ytd-lede">The estimated cost of one summary pass over a video's transcript for each text model. Move the slider to match your video's length.</p>

<div class="pricing-controls">
  <label class="pricing-length" for="pricing-minutes">Video length <output id="pricing-minutes-label">1 hour</output>
    <input type="range" id="pricing-minutes" min="15" max="240" step="15" value="60">
  </label>
  <div class="pricing-chips" id="pricing-vendors" role="group" aria-label="Filter by vendor"></div>
</div>

<div class="pricing-bars" id="pricing-bars" role="list" aria-label="Estimated summary cost by model"></div>
<div class="pricing-tooltip" id="pricing-tooltip" hidden></div>

??? note "How the estimate works"

    A summary pass sends the whole transcript and gets about 2,000 tokens back. Transcripts run about **200 tokens per minute** of speech; the [benchmark](performance.md) video is 3h11m and 37,663 tokens. A full run makes several passes (speakers, Q&A, summary, one-sentence summary and tags), and Q&A also sends the timed captions, so a full text run costs roughly 4–5 times one pass. Claude models count about a third more tokens for the same text.

## Price list

<div class="pricing-controls">
  <input type="search" id="pricing-search" placeholder="Search models or vendors" aria-label="Search models or vendors">
  <div class="pricing-tabs" id="pricing-tabs" role="tablist" aria-label="Model type"></div>
</div>

<div class="pricing-table-wrap"><table class="pricing-table" id="pricing-table"></table></div>
<p class="pricing-count" id="pricing-count" aria-live="polite"></p>

??? note "Units, conversions and sources"

    - Token prices are USD per 1 million tokens. **Cached input** applies when the same prompt prefix is reused.
    - Speech-to-text, text-to-speech and translation services are shown in the vendor's own unit (per minute or per 1M characters). The CLI converts them to token equivalents (about 4 characters or 0.2 seconds of audio per token) to estimate costs.
    - The *per hour* estimates assume 150 spoken words (about 900 characters) a minute, and that Gemini audio uses 32 tokens per second.
    - Infographic costs were measured on a real run at the default settings (the highest resolution each model supports, and high thinking for Gemini) and include the image prompt.
    - Prices come from the vendors' pricing pages and [llm-prices.com](https://www.llm-prices.com/), and they live in [`youtube_to_docs/prices.py`](https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs/blob/main/youtube_to_docs/prices.py), which is the same file the CLI uses. Spot a stale price? [Open an issue](https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs/issues).

<!-- PRICING_DATA -->
