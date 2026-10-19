# Performance Scripts

This directory contains scripts for benchmarking the performance of various models and processes in the `youtube-to-docs` project.

## Scripts

### `time_to_summary.py`

This script benchmarks the time it takes for different LLM models to generate a summary for a specific YouTube video transcript.

#### Usage

To run the script, use `uv run`:

```bash
uv run --extra all scripts/performance/time_to_summary.py
```

#### What it does:
1.  Fetches the transcript for a hardcoded video ID (`atmGAHYpf_c`).
2.  Iterates through a predefined list of models (Bedrock, including OpenAI GPT models on Bedrock, Gemini, Vertex, Foundry).
3.  Measures the time taken by `generate_summary` for each model.
4.  If a model fails, returns an error message (e.g. a retired model) or returns no output tokens, the time is set to `999`.
5.  Outputs the results to a CSV file named `time_to_summarize.csv`.

#### Output
The output CSV `time_to_summarize.csv` contains the following columns:
- `model`: The name of the LLM model.
- `time (seconds)`: The duration of the summarization task.
- `input_tokens`: The number of input tokens used.
- `output_tokens`: The number of output tokens generated.
- `input_price_per_1m`: The price per 1 million input tokens.
- `output_price_per_1m`: The price per 1 million output tokens.
- `total_cost`: The calculated cost of the summarization task.
- `date (today)`: The date the benchmark was run.

The results are sorted by `time (seconds)` in ascending order.
