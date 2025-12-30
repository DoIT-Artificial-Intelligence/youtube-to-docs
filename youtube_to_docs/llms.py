import os
import re
from typing import Any, Dict, List, Tuple, cast

import google.auth
import requests
from google import genai
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.genai import types
from openai import OpenAI

from youtube_to_docs.prices import PRICES


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
    Fetches model pricing from local prices.py.
    Returns (input_price_per_1m, output_price_per_1m).
    """
    try:
        prices = cast(List[Dict[str, Any]], PRICES.get("prices", []))
        aliases = cast(Dict[str, str], PRICES.get("aliases", {}))

        # 1. Try exact match first
        for p in prices:
            if p["id"] == model_name:
                return p["input"], p["output"]

        # 2. Try normalized name
        normalized_name = normalize_model_name(model_name)

        # Check aliases
        search_name = aliases.get(normalized_name, normalized_name)

        for p in prices:
            if p["id"] == search_name:
                return p.get("input"), p.get("output")

        print(f"model {model_name} is not found in youtube_to_docs/prices.py")

    except Exception as e:
        print(f"Error accessing pricing data: {e}")

    return None, None


def _query_llm(model_name: str, prompt: str) -> Tuple[str, int, int]:
    """
    Generic function to query the specified LLM model.
    Returns (response_text, input_tokens, output_tokens).
    """
    response_text = ""
    input_tokens = 0
    output_tokens = 0

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
            response_text = response.text or ""
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0
        except KeyError:
            print("Error: GEMINI_API_KEY not found")
            response_text = "Error: GEMINI_API_KEY not found"
        except Exception as e:
            print(f"Gemini API Error: {e}")
            response_text = f"Error: {e}"

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
                        response_text = content_blocks[0]["text"]
                    else:
                        response_text = f"Unexpected response format: {response.text}"

                    usage = response_json.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                else:
                    response_text = (
                        f"Vertex API Error {response.status_code}: {response.text}"
                    )
                    print(response_text)

        except KeyError:
            print(
                "Error: PROJECT_ID environment variable required for GCPVertex models."
            )
            response_text = "Error: PROJECT_ID required"
        except Exception as e:
            print(f"Vertex Request Error: {e}")
            response_text = f"Error: {e}"

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
                        response_text = content_blocks[0]["text"]
                    else:
                        response_text = f"Unexpected content format: {response_json}"

                    usage = response_json.get("usage", {})
                    input_tokens = usage.get("inputTokens", 0)
                    output_tokens = usage.get("outputTokens", 0)
                except KeyError:
                    response_text = f"Unexpected response structure: {response_json}"
            else:
                response_text = (
                    f"Bedrock API Error {response.status_code}: {response.text}"
                )
        except KeyError:
            print(
                "Error: AWS_BEARER_TOKEN_BEDROCK environment variable required for "
                "AWS Bedrock models."
            )
            response_text = "Error: AWS_BEARER_TOKEN_BEDROCK required"
        except Exception as e:
            print(f"Bedrock Request Error: {e}")
            response_text = f"Error: {e}"

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
            response_text = completion.choices[0].message.content
            if completion.usage:
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
        except KeyError:
            print(
                "Error: AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY "
                "environment variables required."
            )
            response_text = "Error: Foundry vars required"
        except Exception as e:
            print(f"Foundry Request Error: {e}")
            response_text = f"Error: {e}"

    return response_text, input_tokens, output_tokens


def generate_transcript(
    model_name: str, audio_path: str, url: str, language: str = "en"
) -> Tuple[str, int, int]:
    """
    Generates a transcript from an audio file using the specified model.
    Currently only supports Gemini models.
    Returns (transcript_text, input_tokens, output_tokens).
    """
    if not model_name.startswith("gemini"):
        return f"Error: STT not yet implemented for model {model_name}", 0, 0

    try:
        GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
        client = genai.Client(api_key=GEMINI_API_KEY)

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        prompt = (
            f"Can you extract the transcript for {url} from this audio in {language}? "
            "Start the response immediately with the transcript."
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="audio/x-m4a",
                        data=audio_bytes,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig()

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )

        response_text = response.text or ""
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return response_text, input_tokens, output_tokens

    except KeyError:
        return "Error: GEMINI_API_KEY not found", 0, 0
    except Exception as e:
        print(f"Gemini STT Error: {e}")
        return f"Error: {e}", 0, 0


def generate_summary(
    model_name: str,
    transcript: str,
    video_title: str,
    url: str,
    language: str = "en",
) -> Tuple[str, int, int]:
    """Generates a summary and returns (summary_text, input_tokens, output_tokens)."""
    prompt = (
        f"I have included a transcript for {url} ({video_title})"
        "\n\n"
        f"Can you please summarize this in {language}?"
        "\n\n"
        f"{transcript}"
    )
    return _query_llm(model_name, prompt)


def extract_speakers(model_name: str, transcript: str) -> Tuple[str, int, int]:
    """
    Extracts speakers from the transcript.
    Returns (speakers_markdown, input_tokens, output_tokens).
    """
    prompt = (
        "I have included a transcript."
        "\n\n"
        "Can you please identify the speakers in the transcript?"
        "\n\n"
        "The output should be a markdown string in English like"
        "\n\n"
        "Speaker 1 (title)"
        "\n"
        "Speaker 2 (title)"
        "\n"
        "etc."
        "\n\n"
        "If the speaker is unknown use the placeholder UNKNOWN and if the title "
        "is unknown use the placeholder UNKNOWN. "
        'If No speaker(s) are detected set it to float("nan").'
        "\n\n"
        f"Transcript: {transcript}"
    )
    return _query_llm(model_name, prompt)


def add_question_numbers(markdown_table: str) -> str:
    """
    Adds a 'question number' column to the markdown table.
    """
    lines = markdown_table.strip().split("\n")
    if not lines:
        return markdown_table

    # Check if it's a valid table (has header and separator)
    if len(lines) < 2:
        return markdown_table

    new_lines = []
    question_counter = 1

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if i == 0:
            # Header row
            # Ensure it starts with | (some LLMs might miss it)
            if not stripped_line.startswith("|"):
                stripped_line = "|" + stripped_line
            new_lines.append(f"| question number {stripped_line}")
        elif i == 1 and ("---" in stripped_line or "-|-" in stripped_line):
            # Separator row
            if not stripped_line.startswith("|"):
                stripped_line = "|" + stripped_line
            new_lines.append(f"|---{stripped_line}")
        else:
            # Data row
            if stripped_line.startswith("|"):
                new_lines.append(f"| {question_counter} {stripped_line}")
                question_counter += 1
            else:
                # Handle potential malformed table rows or text outside the table
                if "|" in stripped_line:  # It has columns but maybe missing start pipe
                    new_lines.append(f"| {question_counter} | {stripped_line}")
                    question_counter += 1
                else:
                    new_lines.append(line)

    return "\n".join(new_lines)


def generate_qa(
    model_name: str, transcript: str, speakers: str, language: str = "en"
) -> Tuple[str, int, int]:
    """
    Extracts Q&A pairs from the transcript.
    Returns (qa_markdown, input_tokens, output_tokens).
    """
    prompt = (
        "I have included a transcript."
        "\n\n"
        "Can you please extract the questions and answers from the transcript "
        f"in {language}?"
        "\n\n"
        "The output should be a markdown table like:"
        "\n\n"
        "| questioner(s) | question | responder(s) | answer |"
        "\n"
        "|---|---|---|---|"
        "\n"
        "| Speaker 1 | What is... | Speaker 2 | It is... |"
        "\n\n"
        "If the questioner or responder is unknown use the placeholder UNKNOWN. "
        "Use people's name and titles in the questioner and responder fields. "
        'If no Q&A pairs are detected set it to float("nan").'
        "\n\n"
        f"Speakers detected: {speakers}"
        "\n\n"
        f"Transcript: {transcript}"
    )
    response_text, input_tokens, output_tokens = _query_llm(model_name, prompt)

    if (
        response_text.strip() != "nan"
        and response_text.strip() != 'float("nan")'
        and "|" in response_text
    ):
        response_text = add_question_numbers(response_text)

    return response_text, input_tokens, output_tokens
