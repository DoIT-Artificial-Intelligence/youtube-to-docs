import json
import os
import re

import requests
from bs4 import BeautifulSoup
from google import genai


def fetch_pricing_page():
    url = "https://ai.google.dev/gemini-api/docs/pricing"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_price_string(price_str):
    """Extracts the first dollar amount from a string."""
    # Look for $X.XX or $X
    match = re.search(r"\$([\d\.]+)", price_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def fetch_sdk_models():
    """Fetches models from the Google GenAI SDK."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not set. Cannot fetch SDK models.")
        return []

    try:
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        # Return just the model IDs (e.g., "gemini-1.5-flash")
        # stripping "models/" prefix
        return [m.name.split("/")[-1] for m in models]
    except Exception as e:
        print(f"Error fetching SDK models: {e}")
        return []


def scrape_pricing_data():
    """Scrapes pricing data from the web and returns a dictionary."""
    print("Fetching pricing page...")
    html_content = fetch_pricing_page()
    soup = BeautifulSoup(html_content, "html.parser")

    scraped_prices = {}  # {model_id_from_h2: {'input': float, 'output': float}}

    h2_tags = soup.find_all("h2")
    print(f"Found {len(h2_tags)} sections on pricing page.")

    for h2 in h2_tags:
        web_model_id = h2.get("id")
        if not web_model_id:
            continue

        next_table = h2.find_next("table", class_="pricing-table")
        if not next_table:
            continue

        # Check if table belongs to this h2
        prev_h2 = next_table.find_previous("h2")
        if prev_h2 != h2:
            continue

        table = next_table

        # Determine Paid Tier column index
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        paid_col_idx = -1
        for i, h in enumerate(headers):
            if "Paid Tier" in h:
                paid_col_idx = i
                break

        if paid_col_idx == -1:
            paid_col_idx = 2  # Default fallback

        input_price = None
        output_price = None

        rows = table.find("tbody").find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if not cells or len(cells) <= paid_col_idx:
                continue

            label = cells[0].get_text(strip=True).lower()
            price_text = cells[paid_col_idx].get_text(" ", strip=True)

            if "input price" in label and input_price is None:
                extracted = parse_price_string(price_text)
                if extracted is not None:
                    input_price = extracted

            if "output price" in label and output_price is None:
                extracted = parse_price_string(price_text)
                if extracted is not None:
                    output_price = extracted

        if input_price is not None or output_price is not None:
            scraped_prices[web_model_id] = {
                "input": input_price,
                "output": output_price,
            }

    return scraped_prices


def update_prices():
    # 1. Fetch SDK Models
    sdk_models = fetch_sdk_models()
    if not sdk_models:
        print("No SDK models found. Using existing JSON or falling back to empty list.")
        # Try to load existing JSON to get model list if SDK fails
        if os.path.exists("scripts/prices/gemini-prices.json"):
            with open("scripts/prices/gemini-prices.json", "r") as f:
                existing_data = json.load(f)
                sdk_models = [item["Model"] for item in existing_data]
        else:
            sdk_models = []

    # 2. Scrape Pricing Data
    scraped_data = scrape_pricing_data()

    # 3. Combine Data
    final_output = []

    for model_id in sdk_models:
        entry = {
            "Model": model_id,
            "Input Price (USD per 1M tokens)": None,
            "Output Price (USD per 1M tokens)": None,
        }

        # Try to find match in scraped data
        # Exact match
        match = scraped_data.get(model_id)

        if not match:
            # Try fuzzy match / version stripping
            # e.g. SDK: gemini-1.5-pro-001 -> Web: gemini-1.5-pro
            # e.g. SDK: gemini-2.0-flash-exp -> Web: gemini-2.0-flash
            # (maybe? or prefer exact)

            # Remove -001, -latest, etc.
            # But be careful not to match gemini-1.5-flash to gemini-1.5-pro

            # Simple heuristic: look for web keys that are prefixes of sdk model id
            # Sort web keys by length descending to match longest prefix first
            sorted_web_keys = sorted(scraped_data.keys(), key=len, reverse=True)
            for web_key in sorted_web_keys:
                if model_id.startswith(web_key):
                    # Check if the suffix is just versioning stuff
                    suffix = model_id[len(web_key) :]
                    if suffix.startswith("-") or not suffix:
                        match = scraped_data[web_key]
                        break

        if match:
            entry["Input Price (USD per 1M tokens)"] = match["input"]
            entry["Output Price (USD per 1M tokens)"] = match["output"]

        final_output.append(entry)

    # Sort
    final_output.sort(key=lambda x: x["Model"])

    # Write to JSON
    json_path = "scripts/prices/gemini-prices.json"
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Done. Wrote {len(final_output)} models to {json_path}")


if __name__ == "__main__":
    update_prices()
