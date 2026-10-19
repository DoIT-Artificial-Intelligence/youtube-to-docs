# Performance

## Summaries

How long each model takes to summarize the same transcript, and what it costs. The test uses the YouTube transcript of [atmGAHYpf_c](https://www.youtube.com/watch?v=atmGAHYpf_c), about 38,000 input tokens. Claude models count tokens differently, so they report more input tokens for the same text.

- **Time** is the wall-clock time for one `generate_summary` call. *failed* means the call failed or the provider wasn't configured.
- **Cost** is in USD, from the token counts and the prices on the [Pricing](pricing.md) page.

To re-run it, see [`scripts/performance`](https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs/tree/main/scripts/performance).

<div class="ytd-perf" markdown>

--8<-- "docs/assets/time_to_summarize.md"

</div>

## Infographics

How long each image model takes to draw an infographic from the same summary, and what it costs. The summary is of the same video, written by `gemini-3.5-flash-lite` (the default `-m` model). Each model runs at its default: 16:9 landscape at the highest resolution it supports, with high thinking for Gemini.

- **Time** is the wall-clock time for one `generate_infographic` call. *failed* means the call failed or the provider wasn't configured.
- **Cost** is in USD, from the token counts and the prices on the [Pricing](pricing.md) page.
- **Size** is the width and height of the image in pixels. `gemini-3.1-flash-lite-image` only supports 1K.

<div class="ytd-perf" markdown>

--8<-- "docs/assets/time_to_infographic.md"

</div>

`gemini-3-pro-image`, the default for `-i`, was the only model with no text errors on both videos we compared (this one and a 3h49m hearing). Image models can still misspell words, so check the text before you publish an infographic. Select an image to open the preview at full width.

<div class="ytd-gallery" markdown>

<figure markdown>
<a href="../assets/infographics/gemini-3-pro-image.webp"><img src="../assets/infographics/gemini-3-pro-image.webp" alt="Infographic titled MJM Committee Session Summary: AI in Maryland State Government (10/1/2025 #1), naming co-chairs Senator Katie Fry Hester and Delegate Ann Kaiser. Panels cover the federal overview from NCSL, a Maryland DoIT update (responsible AI policy, 1 in 4 employees using enterprise AI, 100+ agency use cases), state agency chatbots, national best practices from Denver, Oregon and Google Public Sector, and committee Q&A themes on data, bias and cost savings." loading="lazy" /></a>
<figcaption><code>gemini-3-pro-image</code>: Default for <code>-i</code>. Clear 16:9 layout with no text errors found.</figcaption>
</figure>

<figure markdown>
<a href="../assets/infographics/gemini-nano-banana-2.1.webp"><img src="../assets/infographics/gemini-nano-banana-2.1.webp" alt="Infographic titled MJM Committee Session, 10/1/2025 #1: Artificial Intelligence (AI) in Maryland State Government, on a dark teal background. Large figures read 1 in 4 state employees using enterprise AI, 100+ agency use cases and 700+ community of practice members, alongside panels on the federal AI action plan, Maryland agencies, national best practices and committee Q&A." loading="lazy" /></a>
<figcaption><code>gemini-nano-banana-2.1</code>: Bold, large headline numbers that stay readable as a thumbnail.</figcaption>
</figure>

<figure markdown>
<a href="../assets/infographics/gemini-3.1-flash-lite-image.webp"><img src="../assets/infographics/gemini-3.1-flash-lite-image.webp" alt="Infographic titled MJM Committee Session Summary, styled as a dark blue dashboard, covering the federal overview, Maryland DoIT progress, state agency implementations, national best practices and committee Q&A. Some labels are garbled and the risk classification panel is repeated." loading="lazy" /></a>
<figcaption><code>gemini-3.1-flash-lite-image</code>: Fastest and cheapest, but 1K only, with some garbled and repeated text.</figcaption>
</figure>

<figure markdown>
<a href="../assets/infographics/foundry-gpt-image-2.5-flare.webp"><img src="../assets/infographics/foundry-gpt-image-2.5-flare.webp" alt="Infographic titled MJM Committee Session, 10/1/2025 #1, Joint Committee on Cybersecurity, IT, and Biotechnology. Four numbered columns list key takeaways from the federal overview, the Maryland DoIT update, state agency implementations and national best practices, naming each speaker, with committee Q&A highlights and a Big Picture panel along the bottom." loading="lazy" /></a>
<figcaption><code>foundry-gpt-image-2.5-flare</code>: The most detail, including speakers and their roles, in a dense layout.</figcaption>
</figure>

<figure markdown>
<a href="../assets/infographics/foundry-gpt-image-2.5-sunburst.webp"><img src="../assets/infographics/foundry-gpt-image-2.5-sunburst.webp" alt="Infographic titled MJM Committee Session, 10/1/2025 #1, with a Maryland flag and State House banner. Colored columns cover the federal overview, Maryland DoIT update, state agency implementations and other states' examples, with call-out figures such as 40% of 311 interactions, 90% customer satisfaction and 26 FTE saved per month, and a closing panel reading Smarter tools. Stronger services. A brighter Maryland." loading="lazy" /></a>
<figcaption><code>foundry-gpt-image-2.5-sunburst</code>: The most polished, poster-like design, but with several misspellings.</figcaption>
</figure>

</div>
