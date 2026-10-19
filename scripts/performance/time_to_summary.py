import os
import time
from datetime import date

import polars as pl

from youtube_to_docs.llms import generate_summary, get_model_pricing
from youtube_to_docs.transcript import (
    fetch_transcript,
    get_video_details,
    get_youtube_service,
)

VIDEO_ID = "atmGAHYpf_c"
MODELS = [
    "bedrock-nova-2-lite-v1",
    "bedrock-claude-haiku-5-5",
    "bedrock-claude-sonnet-5-5",
    "bedrock-claude-opus-5-5",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-4-argon",
    "vertex-gemini-3.5-flash-lite",
    "vertex-gemini-3.8-flash",
    "vertex-gemini-4-argon",
    "vertex-claude-haiku-5-5",
    "vertex-claude-sonnet-5-5",
    "vertex-claude-opus-5-5",
    "foundry-gpt-6-luna",
    "foundry-gpt-6.1-sol",
    "foundry-gpt-6-astra",
]


def main():
    print(f"Fetching transcript for {VIDEO_ID}...")

    yt_service = get_youtube_service()
    details = get_video_details(VIDEO_ID, yt_service)

    if details and details[0]:
        video_title = details[0]
        url = details[6]
    else:
        video_title = "Unknown Title"
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"

    transcript_data = fetch_transcript(VIDEO_ID)
    if not transcript_data:
        print("Failed to fetch transcript.")
        return

    transcript_text = transcript_data[0]
    print(f"Transcript fetched. Length: {len(transcript_text)} characters.")

    results = []
    today = date.today()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Output to docs/assets
    docs_assets_dir = os.path.abspath(os.path.join(script_dir, "../../docs/assets"))
    os.makedirs(docs_assets_dir, exist_ok=True)

    output_path = os.path.join(docs_assets_dir, "time_to_summarize.csv")
    md_output_path = os.path.join(docs_assets_dir, "time_to_summarize.md")

    for model in MODELS:
        print(f"Testing model: {model}")
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0
        input_price = 0.0
        output_price = 0.0
        cost = 0.0
        try:
            summary, input_tokens, output_tokens = generate_summary(
                model, transcript_text, video_title, url
            )

            # Providers return errors as text (e.g. "Bedrock API Error 404: ..."
            # for a retired model) rather than raising, so also treat an empty
            # response as a failure.
            if (
                summary.startswith("Error:")
                or "API Error" in summary[:100]
                or output_tokens == 0
            ):
                elapsed_time = 999.0
            else:
                elapsed_time = round(time.time() - start_time, 2)
                input_price, output_price = get_model_pricing(model)
                if input_price is not None and output_price is not None:
                    cost = round(
                        (input_tokens * input_price + output_tokens * output_price)
                        / 1_000_000,
                        2,
                    )

        except Exception as e:
            print(f"Model {model} failed: {e}")
            elapsed_time = 999.0

        results.append(
            {
                "model": model,
                "time (seconds)": elapsed_time,
                "total_cost": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_price_per_1m": input_price if input_price is not None else 0.0,
                "output_price_per_1m": output_price
                if output_price is not None
                else 0.0,
                "date (today)": today,
            }
        )

        df = pl.DataFrame(results)
        df = df.select(
            [
                "model",
                "time (seconds)",
                "total_cost",
                "input_tokens",
                "output_tokens",
                "input_price_per_1m",
                "output_price_per_1m",
                "date (today)",
            ]
        )
        df = df.sort("time (seconds)")
        df.write_csv(output_path)

        # Write markdown table
        with open(md_output_path, "w") as f:
            # Header
            cols = df.columns
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            # Rows
            for row in df.iter_rows():
                f.write("| " + " | ".join(str(x) for x in row) + " |\n")

        print(f"Results updated in {output_path} and {md_output_path}")


if __name__ == "__main__":
    main()
