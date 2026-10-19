import io
import os
import time
from datetime import date

import polars as pl
from PIL import Image
from tables import write_infographic_table

from youtube_to_docs.infographic import generate_infographic
from youtube_to_docs.llms import generate_summary, get_model_pricing
from youtube_to_docs.transcript import (
    fetch_transcript,
    get_video_details,
    get_youtube_service,
)

VIDEO_ID = "atmGAHYpf_c"
# The default -m model, so every image model draws the same summary.
SUMMARY_MODEL = "gemini-3.5-flash-lite"
MODELS = [
    "gemini-3-pro-image",
    "gemini-nano-banana-2.1",
    "gemini-3.1-flash-lite-image",
    "foundry-gpt-image-2.5-sunburst",
    "foundry-gpt-image-2.5-flare",
]
# Width of the WebP previews shown on the Performance page.
PREVIEW_WIDTH = 1600


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

    print(f"Summarizing with {SUMMARY_MODEL}...")
    summary, _, _ = generate_summary(
        SUMMARY_MODEL, transcript_data[0], video_title, url
    )
    if summary.startswith("Error:"):
        print(f"Failed to summarize: {summary}")
        return

    results = []
    today = date.today()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_assets_dir = os.path.abspath(os.path.join(script_dir, "../../docs/assets"))
    previews_dir = os.path.join(docs_assets_dir, "infographics")
    os.makedirs(previews_dir, exist_ok=True)

    output_path = os.path.join(docs_assets_dir, "time_to_infographic.csv")
    md_output_path = os.path.join(docs_assets_dir, "time_to_infographic.md")

    for model in MODELS:
        print(f"Testing model: {model}")
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        size = ""
        try:
            image_bytes, input_tokens, output_tokens = generate_infographic(
                model, summary, video_title
            )
            if not image_bytes:
                elapsed_time = 999.0
            else:
                elapsed_time = round(time.time() - start_time, 2)
                input_price, output_price = get_model_pricing(model)
                if input_price is not None and output_price is not None:
                    cost = round(
                        (input_tokens * input_price + output_tokens * output_price)
                        / 1_000_000,
                        4,
                    )
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                size = f"{image.width}x{image.height}"
                image.thumbnail((PREVIEW_WIDTH, PREVIEW_WIDTH))
                image.save(
                    os.path.join(previews_dir, f"{model}.webp"), "WEBP", quality=80
                )

        except Exception as e:
            print(f"Model {model} failed: {e}")
            elapsed_time = 999.0

        results.append(
            {
                "model": model,
                "time (seconds)": elapsed_time,
                "total_cost": cost,
                "size": size,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "date (today)": today,
            }
        )

        df = pl.DataFrame(results).sort("time (seconds)")
        df.write_csv(output_path)

        write_infographic_table(df, md_output_path)

        print(f"Results updated in {output_path} and {md_output_path}")


if __name__ == "__main__":
    main()
