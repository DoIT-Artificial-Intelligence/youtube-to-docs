import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

import requests

from youtube_to_docs.prices import PRICES
from youtube_to_docs.utils import add_question_numbers, normalize_model_name


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

    if model_name.startswith("nova") or model_name.startswith("claude"):
        model_name = "bedrock-" + model_name

    if model_name.startswith("gemini"):
        try:
            from google import genai
            from google.genai import types

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
            import subprocess

            import google.auth
            from google.auth.exceptions import RefreshError
            from google.auth.transport.requests import AuthorizedSession

            vertex_project_id = os.environ["PROJECT_ID"]
            actual_model_name = model_name.replace("vertex-", "")

            if actual_model_name.startswith("claude"):
                endpoint = (
                    "https://us-east5-aiplatform.googleapis.com/v1/"
                    f"projects/{vertex_project_id}/locations/us-east5/"
                    f"publishers/anthropic/models/{actual_model_name}:rawPredict"
                )
                # ... existing claude logic ...
                payload = {
                    "anthropic_version": "vertex-2023-10-16",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64_000,
                    "stream": False,
                }
                headers = {"Content-Type": "application/json; charset=utf-8"}

                vertex_api_key = os.environ.get("VERTEXAI_API_KEY")
                response = None

                if vertex_api_key:
                    # Use API Key if available
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        params={"key": vertex_api_key},
                    )
                    # If API key is not supported or fails, we will try ADC fallback
                    if response.status_code != 200:
                        print(
                            f"Vertex API Key failed (Status {response.status_code})."
                            " Falling back to ADC..."
                        )
                        response = None

                if response is None:
                    # Fallback to Application Default Credentials
                    vertex_credentials, _ = google.auth.default()
                    authed_session = AuthorizedSession(vertex_credentials)

                    try:
                        response = authed_session.post(
                            endpoint, json=payload, headers=headers
                        )
                    except RefreshError:
                        print(
                            "Vertex AI Credentials expired. Launching gcloud login..."
                        )
                        try:
                            # Run gcloud login interactively
                            subprocess.run(
                                ["gcloud", "auth", "application-default", "login"],
                                check=True,
                            )
                            # Reload credentials and retry
                            vertex_credentials, _ = google.auth.default()
                            authed_session = AuthorizedSession(vertex_credentials)
                            response = authed_session.post(
                                endpoint, json=payload, headers=headers
                            )
                        except Exception as e:
                            error_msg = f"Re-authentication failed: {e}"
                            print(error_msg)
                            return f"Error: {error_msg}", 0, 0

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
            elif actual_model_name.startswith("gemini"):
                from google import genai
                from google.genai import types

                vertex_location = os.environ.get("VERTEX_LOCATION", "us-east5")
                client = genai.Client(
                    vertexai=True,
                    project=vertex_project_id,
                    location=vertex_location,
                    http_options=types.HttpOptions(api_version="v1"),
                )
                response = client.models.generate_content(
                    model=actual_model_name,
                    contents=prompt,
                )
                response_text = response.text or ""
                if response.usage_metadata:
                    input_tokens = response.usage_metadata.prompt_token_count or 0
                    output_tokens = response.usage_metadata.candidates_token_count or 0

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
            if "claude" in actual_model_name:
                if not actual_model_name.startswith(
                    "anthropic."
                ) and not actual_model_name.startswith("us.anthropic."):
                    actual_model_name = f"us.anthropic.{actual_model_name}:0"
            elif "nova" in actual_model_name:
                if not actual_model_name.startswith(
                    "amazon."
                ) and not actual_model_name.startswith("us.amazon."):
                    actual_model_name = f"us.amazon.{actual_model_name}:0"
                if not actual_model_name.endswith(":0"):
                    actual_model_name = f"{actual_model_name}:0"
            elif "llama" in actual_model_name:
                if not actual_model_name.startswith("meta."):
                    actual_model_name = f"meta.{actual_model_name}"

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
            from openai import OpenAI

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
    model_name: str,
    audio_path: str,
    url: str,
    language: str = "en",
    srt: bool = False,
) -> Tuple[str, int, int]:
    """
    Generates a transcript from an audio file using the specified model.
    Currently only supports Gemini models.
    Returns (transcript_text, input_tokens, output_tokens).
    """
    if model_name.startswith("nova") or model_name.startswith("claude"):
        model_name = "bedrock-" + model_name

    if model_name.startswith("gcp-"):
        return _transcribe_gcp(model_name, audio_path, url, language, srt)

    if not model_name.startswith("gemini"):
        return f"Error: STT not yet implemented for model {model_name}", 0, 0

    try:
        from google import genai
        from google.genai import types

        GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
        client = genai.Client(api_key=GEMINI_API_KEY)

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        prompt = (
            f"Can you extract the transcript for {url} from this audio in {language}? "
            "Start the response immediately with the transcript. "
            "Provide the transcript as a single continuous string of text "
            "without line breaks or speaker labels."
        )
        if srt:
            prompt = (
                f"Can you extract the transcript for {url} from this audio in "
                f"{language}? Start the response immediately with the "
                "transcript. \n\nPlease provide the transcript in SRT format "
                "with accurate time stamps."
            )
        else:
            prompt = (
                f"Can you extract the transcript for {url} from this audio in "
                f"{language}? Start the response immediately with the "
                "transcript. Provide the transcript as a single continuous "
                "string of text without line breaks or speaker labels."
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

        print(f"Starting transcription with model: {model_name}...")
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


def _transcribe_gcp(
    model_name: str,
    audio_path: str,
    url: str,
    language: str = "en",
    srt: bool = False,
) -> Tuple[str, int, int]:
    """
    Transcribes audio using Google Cloud Speech-to-Text V2 API.
    Requires 'YTD_GCS_BUCKET_NAME' env var for temporary storage.
    """
    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import speech_v2, storage
        from google.cloud.speech_v2.types import cloud_speech
    except ImportError:
        return (
            "Error: google-cloud-speech and google-cloud-storage are required for "
            "GCP models. Install with `pip install '.[gcp]'`",
            0,
            0,
        )

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    bucket_name = os.environ.get("YTD_GCS_BUCKET_NAME", "youtube-to-docs")

    if not project_id:
        return "Error: GOOGLE_CLOUD_PROJECT environment variable is required.", 0, 0

    # Extract model ID (e.g. gcp-chirp3 -> chirp_3 or just pass as is if mapped?)
    actual_model = model_name.replace("gcp-", "").replace("-", "_")
    if actual_model == "chirp3":
        actual_model = "chirp_3"  # specific fix based on user example

    # Determine location
    # Chirp models are often regional (e.g. us-central1), not global.
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not location:
        if "chirp" in actual_model:
            location = "us"
        else:
            location = "global"

    # 1. Upload to GCS
    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob_name = f"temp/ytd_audio_{uuid.uuid4()}.m4a"
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(audio_path)
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        return f"Error uploading to GCS: {e}", 0, 0

    transcript_text = ""

    try:
        # 2. Transcribe
        client_options = None
        if location != "global":
            api_endpoint = f"{location}-speech.googleapis.com"
            client_options = ClientOptions(api_endpoint=api_endpoint)

        # Instantiates a client
        client = speech_v2.SpeechClient(client_options=client_options)

        # Map 'en' to 'en-US' for GCP V2 if needed
        if language == "en":
            language = "en-US"

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[language],
            model=actual_model,
        )

        if srt:
            config.features = speech_v2.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            )
        else:
            config.features = speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=True,
            )

        file_metadata = speech_v2.BatchRecognizeFileMetadata(uri=gcs_uri)

        output_bucket_uri = gcs_uri.rsplit("/", 1)[0] + "/transcripts/"

        request = speech_v2.BatchRecognizeRequest(
            recognizer=f"projects/{project_id}/locations/{location}/recognizers/_",
            config=config,
            files=[file_metadata],
            recognition_output_config=speech_v2.RecognitionOutputConfig(
                gcs_output_config=speech_v2.GcsOutputConfig(uri=output_bucket_uri),
            ),
        )

        print(f"Starting transcription with model: {model_name}...")
        operation = client.batch_recognize(request=request)

        # Poll logic
        while not operation.done():
            print("Transcript processing...", end="\r")
            time.sleep(5)

        response = operation.result()

        # 3. Process GCS Output
        # GCP V2 BatchRecognize with GcsOutputConfig writes a JSON file.
        # We need to find the output URI from the response.
        if gcs_uri in response.results:
            batch_result = response.results[gcs_uri]
            output_uri = batch_result.uri
            if not output_uri:
                return f"Error: No output URI found in batch result for {gcs_uri}", 0, 0

            # Read the JSON from GCS
            try:
                # output_uri is like gs://bucket/temp/transcripts/ytd_audio_uuid_transcript_....json
                import json

                bucket_name_out = output_uri.split("/")[2]
                blob_name_out = "/".join(output_uri.split("/")[3:])
                blob_out = storage_client.bucket(bucket_name_out).blob(blob_name_out)
                json_content = blob_out.download_as_text()
                transcript_json = json.loads(json_content)

                # Full text accumulator
                full_text_parts = []
                # SRT accumulator
                srt_entries = []
                srt_counter = 1

                # Helper for SRT formatting
                def format_time(seconds_str):
                    if not seconds_str:
                        return "00:00:00,000"
                    total_seconds = float(seconds_str.replace("s", ""))
                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    seconds = int(total_seconds % 60)
                    milliseconds = int((total_seconds * 1000) % 1000)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

                # The JSON structure for BatchRecognize V2 results is:
                # { "results": [ { "alternatives": [ { "transcript": "...",
                #   "words": [...] } ] } ] }
                results = transcript_json.get("results", [])

                for result in results:
                    alternatives = result.get("alternatives", [])
                    if not alternatives:
                        continue
                    alt = alternatives[0]
                    transcript_part = alt.get("transcript", "")
                    full_text_parts.append(transcript_part)

                    if srt and "words" in alt:
                        words = alt["words"]
                        current_segment_words = []
                        current_segment_len = 0

                        for word_info in words:
                            word = word_info.get("word", "")
                            start = word_info.get("startOffset", "0s")
                            end = word_info.get("endOffset", "0s")

                            current_segment_words.append((word, start, end))
                            current_segment_len += len(word) + 1

                            if (
                                word.endswith(".")
                                or word.endswith("?")
                                or word.endswith("!")
                                or current_segment_len > 80
                            ):
                                if current_segment_words:
                                    seg_text = " ".join(
                                        [w[0] for w in current_segment_words]
                                    )
                                    seg_start = format_time(current_segment_words[0][1])
                                    seg_end = format_time(current_segment_words[-1][2])

                                    srt_entries.append(
                                        f"{srt_counter}\n"
                                        f"{seg_start} --> {seg_end}\n"
                                        f"{seg_text}\n"
                                    )
                                    srt_counter += 1
                                    current_segment_words = []
                                    current_segment_len = 0

                        # Flush remaining
                        if current_segment_words:
                            seg_text = " ".join([w[0] for w in current_segment_words])
                            seg_start = format_time(current_segment_words[0][1])
                            seg_end = format_time(current_segment_words[-1][2])
                            srt_entries.append(
                                f"{srt_counter}\n"
                                f"{seg_start} --> {seg_end}\n"
                                f"{seg_text}\n"
                            )
                            srt_counter += 1

                transcript_text = " ".join(full_text_parts)

                # Cleanup the output JSON
                try:
                    blob_out.delete()
                except Exception:
                    pass

                if srt:
                    return "\n".join(srt_entries), 0, 0

                return transcript_text, 0, 0

            except Exception as e:
                return f"Error parsing GCS transcript JSON: {e}", 0, 0

    except Exception as e:
        return f"Error during transcription: {e}", 0, 0
    finally:
        # 3. Cleanup GCS
        try:
            blob.delete()
        except Exception:
            pass

    return transcript_text, 0, 0


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


def generate_one_sentence_summary(
    model_name: str,
    summary_text: str,
    language: str = "en",
) -> Tuple[str, int, int]:
    """Generates a one sentence summary from the provided summary text."""
    prompt = (
        f"Can you please summarize the following text into one sentence in {language}?"
        "\n\n"
        f"{summary_text}"
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


def generate_qa(
    model_name: str,
    transcript: str,
    speakers: str,
    url: str,
    language: str = "en",
    timing_reference: Optional[str] = None,
) -> Tuple[str, int, int]:
    """
    Extracts Q&A pairs from the transcript.
    Returns (qa_markdown, input_tokens, output_tokens).
    """
    prompt = (
        "I have included a transcript (which might be in SRT format with timestamps)."
        "\n\n"
        "Can you please extract the questions and answers from the transcript "
        f"in {language}?"
        "\n\n"
        "The output should be a markdown table like:"
        "\n\n"
        "| questioner(s) | question | responder(s) | answer | "
        "timestamp | timestamp url |"
        "\n"
        "|---|---|---|---|---|---|"
        "\n"
        "| Speaker 1 | What is... | Speaker 2 | It is... | 01:23 | "
        "[Link](https://youtu.be/...&t=83) |\n"
        "\n\n"
        "If the questioner or responder is unknown use the placeholder UNKNOWN. "
        "Use people's name and titles in the questioner and responder fields. "
        'If no Q&A pairs are detected set it to float("nan").'
        "\n\n"
        "For the 'timestamp' column, use the format MM:SS or HH:MM:SS. "
        "If the 'Timing Reference' below is provided, please use its "
        "timestamps to provide high accuracy timestamps. Otherwise, use "
        "timestamps from the main transcript."
        "For the 'timestamp url' column, use the base YouTube URL provided below "
        "and append the timestamp in seconds (e.g. &t=123 or ?t=123). "
        "Format this column as a markdown hyperlink with the text 'Link' "
        "(e.g. [Link](https://youtu.be/...&t=123)). "
        "If the base URL already contains a '?', use '&t=' otherwise use '?t='. "
        f"Base URL: {url}"
        "\n\n"
        f"Speakers detected: {speakers}"
        "\n\n"
        f"Content Transcript: {transcript}"
    )
    if timing_reference:
        prompt += f"\n\nTiming Reference (SRT): {timing_reference}"

    response_text, input_tokens, output_tokens = _query_llm(model_name, prompt)

    if (
        response_text.strip() != "nan"
        and response_text.strip() != 'float("nan")'
        and "|" in response_text
    ):
        response_text = add_question_numbers(response_text)

    return response_text, input_tokens, output_tokens


def generate_tags(
    model_name: str, summary_text: str, language: str = "en"
) -> Tuple[str, int, int]:
    """
    Generates up to 5 comma-separated tags for the provided summary.
    Returns (tags_string, input_tokens, output_tokens).
    """
    prompt = (
        "I have included a summary."
        "\n\n"
        f"Can you please generate up to 5 comma-separated tags for this summary in "
        f"{language}? "
        "Each tag can be one or more words. "
        "Return ONLY the comma-separated tags string without any introductory or "
        "concluding text."
        "\n\n"
        f"Summary: {summary_text}"
    )
    return _query_llm(model_name, prompt)


def generate_alt_text(
    model_name: str,
    image_bytes: bytes,
    language: str = "en",
) -> Tuple[str, int, int]:
    """
    Generates alt text for an infographic based on the generated image.
    Returns (alt_text, input_tokens, output_tokens).
    """
    if model_name.startswith("nova") or model_name.startswith("claude"):
        model_name = "bedrock-" + model_name

    prompt = (
        f"Please provide a descriptive alt text for this infographic "
        f"in {language}. "
        "The alt text should describe the visual layout and key information "
        "presented, making it accessible for someone who cannot see the image. "
        "Start the response immediately with the alt text."
    )

    if model_name.startswith("gemini"):
        try:
            from google import genai
            from google.genai import types

            GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
            client = genai.Client(api_key=GEMINI_API_KEY)

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            mime_type="image/png",
                            data=image_bytes,
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
            # Post-processing: Remove common prefixes like "Alt text: " or "Alt text - "
            response_text = re.sub(
                r"^(Alt text[:\-\s]+)", "", response_text, flags=re.IGNORECASE
            ).strip()

            input_tokens = 0
            output_tokens = 0
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0

            return response_text, input_tokens, output_tokens

        except KeyError:
            return "Error: GEMINI_API_KEY not found", 0, 0
        except Exception as e:
            print(f"Gemini Alt Text Error: {e}")
            return f"Error: {e}", 0, 0

    elif model_name.startswith("bedrock"):
        try:
            import base64

            aws_bearer_token_bedrock = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
            actual_model_name = model_name.replace("bedrock-", "")

            # Mapping (shared logic with _query_llm could be refactored later)
            if "claude" in actual_model_name:
                if not actual_model_name.startswith(
                    "anthropic."
                ) and not actual_model_name.startswith("us.anthropic."):
                    actual_model_name = f"us.anthropic.{actual_model_name}:0"
            elif "nova" in actual_model_name:
                if not actual_model_name.startswith(
                    "amazon."
                ) and not actual_model_name.startswith("us.amazon."):
                    actual_model_name = f"us.amazon.{actual_model_name}:0"
                if not actual_model_name.endswith(":0"):
                    actual_model_name = f"{actual_model_name}:0"

            endpoint = (
                f"https://bedrock-runtime.us-east-1.amazonaws.com/model/"
                f"{actual_model_name}/converse"
            )

            # Convert images bytes to base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": image_base64},
                                }
                            },
                            {"text": prompt},
                        ],
                    }
                ],
                "max_tokens": 2048,
            }

            response = requests.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {aws_bearer_token_bedrock}",
                },
                json=payload,
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
                        # Post-processing
                        response_text = re.sub(
                            r"^(Alt text[:\-\s]+)",
                            "",
                            response_text,
                            flags=re.IGNORECASE,
                        ).strip()

                        usage = response_json.get("usage", {})
                        return (
                            response_text,
                            usage.get("inputTokens", 0),
                            usage.get("outputTokens", 0),
                        )
                    else:
                        return f"Unexpected content format: {response.text}", 0, 0
                except KeyError:
                    return f"Unexpected response structure: {response.text}", 0, 0
            else:
                return (
                    f"Bedrock API Error {response.status_code}: {response.text}",
                    0,
                    0,
                )
        except KeyError:
            return "Error: AWS_BEARER_TOKEN_BEDROCK required", 0, 0
        except Exception as e:
            print(f"Bedrock Alt Text Error: {e}")
            return f"Error: {e}", 0, 0

    return f"Error: Multimodal alt text not yet implemented for {model_name}", 0, 0
