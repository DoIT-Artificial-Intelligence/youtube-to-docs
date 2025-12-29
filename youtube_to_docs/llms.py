import os
import re
from typing import Tuple

import google.auth
import requests
from google import genai
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.genai import types
from openai import OpenAI


def normalize_model_name(model_name: str) -> str:
    """
    Normalizes a model name by stripping prefixes and suffixes.
    Suffixes handled: @20251001, -20251001-v1, -v1.
    Prefixes handled: vertex-, bedrock-, foundry-.
    """
    # Strip prefixes
    normalized = model_name
    prefixes = ["vertex-", "bedrock-", "foundry-"]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    # Strip suffixes using regex: (@\d{8}|-\d{8}-v\d+|-v\d+)$
    normalized = re.sub(r"(@\d{8}|-\d{8}-v\d+|-v\d+)$", "", normalized)

    return normalized


def get_model_pricing(model_name: str) -> Tuple[float | None, float | None]:
    """
    Fetches model pricing from llm-prices repository.
    Returns (input_price_per_1m, output_price_per_1m).
    """
    try:
        url = "https://www.llm-prices.com/current-v1.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])

            # Add https://github.com/simonw/llm-prices/pull/48/files here until merge
            prices.extend(
                [
                    {
                        "id": "amazon-nova-2-omni-preview",
                        "name": "Amazon Nova 2 Omni (Preview)",
                        "input": 0.3,
                        "output": 2.5,
                    },
                    {
                        "id": "amazon-nova-2-pro-preview",
                        "name": "Amazon Nova 2 Pro (Preview)",
                        "input": 1.25,
                        "output": 10.0,
                    },
                    {
                        "id": "amazon-nova-2-lite",
                        "name": "Amazon Nova 2 Lite",
                        "input": 0.3,
                        "output": 2.5,
                    },
                ]
            )

            # 1. Try exact match first
            for p in prices:
                if p["id"] == model_name:
                    return p["input"], p["output"]

            # 2. Try normalized name
            normalized_name = normalize_model_name(model_name)

            # Check aliases
            aliases = {
                "claude-haiku-4-5": "claude-4.5-haiku",
                "nova-2-lite": "amazon-nova-2-lite",
            }
            search_name = aliases.get(normalized_name, normalized_name)

            for p in prices:
                if p["id"] == search_name:
                    return p.get("input"), p.get("output")

            print(
                f"model {model_name} is not found in "
                "https://www.llm-prices.com/current-v1.json"
            )

    except Exception as e:
        print(f"Error fetching pricing data: {e}")

    return None, None


def generate_summary(
    model_name: str, transcript: str, video_title: str, url: str
) -> Tuple[str, int, int]:
    """Generates a summary and returns (summary_text, input_tokens, output_tokens)."""
    summary_text = ""
    input_tokens = 0
    output_tokens = 0
    prompt = (
        f"I have included a transcript for {url} ({video_title})"
        "\n\n"
        "Can you please summarize this?"
        "\n\n"
        f"{transcript}"
    )

    if model_name.startswith("gemini"):
        try:
            GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
            google_genai_client = genai.Client(api_key=GEMINI_API_KEY)
            response = google_genai_client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=prompt)]
                    )
                ],
            )
            summary_text = response.text or ""
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0
        except KeyError:
            print("Error: GEMINI_API_KEY not found")
            summary_text = "Error: GEMINI_API_KEY not found"
        except Exception as e:
            print(f"Gemini API Error: {e}")
            summary_text = f"Error: {e}"

    elif model_name.startswith("vertex"):
        try:
            vertex_project_id = os.environ["PROJECT_ID"]
            vertex_credentials, _ = google.auth.default()
            actual_model_name = model_name.replace("vertex-", "")

            if actual_model_name.startswith("claude"):
                if vertex_credentials.expired:
                    vertex_credentials.refresh(GoogleAuthRequest())
                access_token = vertex_credentials.token
                endpoint = (
                    "https://us-east5-aiplatform.googleapis.com/v1/"
                    f"projects/{vertex_project_id}/locations/us-east5/"
                    f"publishers/anthropic/models/{actual_model_name}:rawPredict"
                )
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=utf-8",
                }
                payload = {
                    "anthropic_version": "vertex-2023-10-16",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64_000,
                    "stream": False,
                }
                response = requests.post(endpoint, headers=headers, json=payload)
                if response.status_code == 200:
                    response_json = response.json()
                    content_blocks = response_json.get("content", [])
                    if (
                        content_blocks
                        and isinstance(content_blocks, list)
                        and "text" in content_blocks[0]
                    ):
                        summary_text = content_blocks[0]["text"]
                    else:
                        summary_text = f"Unexpected response format: {response.text}"

                    usage = response_json.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                else:
                    summary_text = (
                        f"Vertex API Error {response.status_code}: {response.text}"
                    )
                    print(summary_text)

        except KeyError:
            print(
                "Error: PROJECT_ID environment variable required for GCPVertex models."
            )
            summary_text = "Error: PROJECT_ID required"
        except Exception as e:
            print(f"Vertex Request Error: {e}")
            summary_text = f"Error: {e}"

    elif model_name.startswith("bedrock"):
        try:
            aws_bearer_token_bedrock = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
            actual_model_name = model_name.replace("bedrock-", "")
            if actual_model_name.startswith("claude"):
                actual_model_name = f"us.anthropic.{actual_model_name}:0"
            elif actual_model_name.startswith("nova"):
                actual_model_name = f"us.amazon.{actual_model_name}:0"

            endpoint = (
                f"https://bedrock-runtime.us-east-1.amazonaws.com/model/"
                f"{actual_model_name}/converse"
            )
            response = requests.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {aws_bearer_token_bedrock}",
                },
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}],
                        }
                    ],
                    "max_tokens": 64_000,
                },
            )
            if response.status_code == 200:
                response_json = response.json()
                try:
                    content_blocks = response_json["output"]["message"]["content"]
                    if (
                        content_blocks
                        and isinstance(content_blocks, list)
                        and "text" in content_blocks[0]
                    ):
                        summary_text = content_blocks[0]["text"]
                    else:
                        summary_text = f"Unexpected content format: {response_json}"

                    usage = response_json.get("usage", {})
                    input_tokens = usage.get("inputTokens", 0)
                    output_tokens = usage.get("outputTokens", 0)
                except KeyError:
                    summary_text = f"Unexpected response structure: {response_json}"
            else:
                summary_text = (
                    f"Bedrock API Error {response.status_code}: {response.text}"
                )
        except KeyError:
            print(
                "Error: AWS_BEARER_TOKEN_BEDROCK environment variable required for "
                "AWS Bedrock models."
            )
            summary_text = "Error: AWS_BEARER_TOKEN_BEDROCK required"
        except Exception as e:
            print(f"Bedrock Request Error: {e}")
            summary_text = f"Error: {e}"

    elif model_name.startswith("foundry"):
        try:
            AZURE_FOUNDRY_ENDPOINT = os.environ["AZURE_FOUNDRY_ENDPOINT"]
            AZURE_FOUNDRY_API_KEY = os.environ["AZURE_FOUNDRY_API_KEY"]
            actual_model_name = model_name.replace("foundry-", "")
            client = OpenAI(
                base_url=AZURE_FOUNDRY_ENDPOINT, api_key=AZURE_FOUNDRY_API_KEY
            )
            completion = client.chat.completions.create(
                model=actual_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            summary_text = completion.choices[0].message.content
            if completion.usage:
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
        except KeyError:
            print(
                "Error: AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY "
                "environment variables required."
            )
            summary_text = "Error: Foundry vars required"
        except Exception as e:
            print(f"Foundry Request Error: {e}")
            summary_text = f"Error: {e}"

    return summary_text, input_tokens, output_tokens
