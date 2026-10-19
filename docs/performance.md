# Performance

How long each model takes to summarize the same transcript, and what it costs. The test uses the YouTube transcript of [atmGAHYpf_c](https://www.youtube.com/watch?v=atmGAHYpf_c), about 38,000 input tokens. Claude models count tokens differently, so they report more input tokens for the same text.

- **time (seconds)** is the wall-clock time for one `generate_summary` call. `999` means the call failed or the provider wasn't configured.
- **total_cost** is in USD, from the token counts and the prices on the [Pricing](pricing.md) page.

To re-run it, see [`scripts/performance`](https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs/tree/main/scripts/performance).

--8<-- "docs/assets/time_to_summarize.md"
