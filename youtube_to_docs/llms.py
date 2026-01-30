import json
import os
import re
import subprocess
import tempfile
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


def generate_transcript_with_srt(
    model_name: str,
    audio_path: str,
    url: str,
    language: str = "en",
    duration_seconds: Optional[float] = None,
) -> Tuple[str, str, int, int]:
    """
    Generates both transcript text and SRT content from audio in a single call.
    Only supported for GCP models. For other models, call generate_transcript twice.
    Returns (transcript_text, srt_content, input_tokens, output_tokens).
    """
    if model_name.startswith("nova") or model_name.startswith("claude"):
        model_name = "bedrock-" + model_name

    if model_name.startswith("gcp-"):
        return _transcribe_gcp(model_name, audio_path, url, language, duration_seconds)

    # For non-GCP models, return empty SRT (caller should use generate_transcript)
    return "", "", 0, 0


def generate_transcript(
    model_name: str,
    audio_path: str,
    url: str,
    language: str = "en",
    srt: bool = False,
    duration_seconds: Optional[float] = None,
) -> Tuple[str, int, int]:
    """
    Generates a transcript from an audio file using the specified model.
    Currently only supports Gemini models.
    Returns (transcript_text, input_tokens, output_tokens).
    """
    if model_name.startswith("nova") or model_name.startswith("claude"):
        model_name = "bedrock-" + model_name

    if model_name.startswith("gcp-"):
        text, srt_content, in_tok, out_tok = _transcribe_gcp(
            model_name, audio_path, url, language, duration_seconds
        )
        if srt:
            return srt_content, in_tok, out_tok
        return text, in_tok, out_tok

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


def _parse_gcp_time(time_str: str) -> float:
    """Parses a time string (e.g. '10s', '0.100s', or '0:00:02.640000') into seconds."""
    if not time_str:
        return 0.0
    time_str = str(time_str)
    if ":" in time_str:
        # Handle HH:MM:SS.mmmmmm format
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    return float(time_str.replace("s", ""))


def _format_srt_time(seconds: float) -> str:
    """Formats seconds into SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds * 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _process_gcp_batch_result(
    batch_result: Any,
    storage_client: Any,
    offset_seconds: float,
    srt_counter_start: int,
) -> Tuple[str, List[str], int]:
    """
    Processes a single file result from a BatchRecognizeResponse.
    Returns (transcript_text, srt_entries_list, next_srt_counter).
    """

    transcript_text = ""
    srt_entries = []
    next_ctr = srt_counter_start

    if batch_result.error and batch_result.error.code != 0:
        error_msg = batch_result.error.message or "Unknown error"
        print(f"Error in chunk result: {error_msg}")
        return "", [], next_ctr

    # Check for inline result first
    if batch_result.inline_result and batch_result.inline_result.transcript:
        results_list = batch_result.inline_result.transcript.results
        return _process_alternatives(results_list, offset_seconds, srt_counter_start)

    # Fallback to GCS output
    output_uri = batch_result.uri
    if not output_uri:
        print("Error: No output URI or inline result for chunk.")
        return "", [], next_ctr

    try:
        bucket_name_out = output_uri.split("/")[2]
        blob_name_out = "/".join(output_uri.split("/")[3:])
        blob_out = storage_client.bucket(bucket_name_out).blob(blob_name_out)

        # Retry logic for download
        max_retries = 5
        json_content = None
        for attempt in range(max_retries):
            try:
                json_content = blob_out.download_as_text()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (2**attempt))
                else:
                    print(f"Failed to download transcript: {e}")

        if json_content:
            transcript_json = json.loads(json_content)
            results_list = transcript_json.get("results", [])
            transcript_text, srt_entries, next_ctr = _process_alternatives(
                results_list, offset_seconds, srt_counter_start
            )

            # Cleanup output blob
            try:
                blob_out.delete()
            except Exception:
                pass

    except Exception as e:
        print(f"Error processing GCS output: {e}")

    return transcript_text, srt_entries, next_ctr


def _process_alternatives(
    results_list: List[Any], current_offset_sec: float, current_srt_idx: int
) -> Tuple[str, List[str], int]:
    """Helper to process a list of transcript results into text and SRT entries."""
    full_text_parts = []
    srt_entries = []
    srt_counter = current_srt_idx

    for result in results_list:
        alternatives = (
            result.get("alternatives", [])
            if isinstance(result, dict)
            else (result.alternatives if hasattr(result, "alternatives") else [])
        )
        if not alternatives:
            continue

        alt = alternatives[0]
        transcript_part = (
            alt.get("transcript", "") if isinstance(alt, dict) else alt.transcript
        )
        full_text_parts.append(transcript_part)

        words = (
            alt.get("words", [])
            if isinstance(alt, dict)
            else (alt.words if hasattr(alt, "words") else [])
        )

        current_segment_words = []
        current_segment_len = 0

        for word_info in words:
            word = (
                word_info.get("word", "")
                if isinstance(word_info, dict)
                else word_info.word
            )
            start_raw = (
                word_info.get("startOffset", "0s")
                if isinstance(word_info, dict)
                else (
                    word_info.start_offset
                    if hasattr(word_info, "start_offset")
                    else "0s"
                )
            )
            end_raw = (
                word_info.get("endOffset", "0s")
                if isinstance(word_info, dict)
                else (
                    word_info.end_offset if hasattr(word_info, "end_offset") else "0s"
                )
            )

            start_sec = _parse_gcp_time(str(start_raw)) + current_offset_sec
            end_sec = _parse_gcp_time(str(end_raw)) + current_offset_sec

            current_segment_words.append((word, start_sec, end_sec))
            current_segment_len += len(word) + 1

            if (
                word.endswith(".")
                or word.endswith("?")
                or word.endswith("!")
                or current_segment_len > 80
            ):
                if current_segment_words:
                    seg_text = " ".join([w[0] for w in current_segment_words])
                    seg_start = _format_srt_time(current_segment_words[0][1])
                    seg_end = _format_srt_time(current_segment_words[-1][2])

                    srt_entries.append(
                        f"{srt_counter}\n{seg_start} --> {seg_end}\n{seg_text}\n"
                    )
                    srt_counter += 1
                    current_segment_words = []
                    current_segment_len = 0

        # Flush remaining
        if current_segment_words:
            seg_text = " ".join([w[0] for w in current_segment_words])
            seg_start = _format_srt_time(current_segment_words[0][1])
            seg_end = _format_srt_time(current_segment_words[-1][2])
            srt_entries.append(
                f"{srt_counter}\n{seg_start} --> {seg_end}\n{seg_text}\n"
            )
            srt_counter += 1

    return " ".join(full_text_parts), srt_entries, srt_counter


def _transcribe_gcp(
    model_name: str,
    audio_path: str,
    url: str,
    language: str = "en",
    duration_seconds: Optional[float] = None,
) -> Tuple[str, str, int, int]:
    """
    Transcribes audio using Google Cloud Speech-to-Text V2 API.
    Returns (transcript_text, srt_content, input_tokens, output_tokens).
    Requires 'YTD_GCS_BUCKET_NAME' env var for temporary storage.
    """
    try:
        import static_ffmpeg
        from google.api_core.client_options import ClientOptions
        from google.cloud import speech_v2, storage
        from google.cloud.speech_v2.types import cloud_speech

        static_ffmpeg.add_paths()
    except ImportError:
        return (
            "Error: google-cloud-speech, google-cloud-storage, and "
            "static-ffmpeg are required for GCP models. "
            "Install with `pip install '.[gcp,video]'`",
            "",
            0,
            0,
        )

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    bucket_name = os.environ.get("YTD_GCS_BUCKET_NAME", "youtube-to-docs")

    if not project_id:
        return (
            "Error: GOOGLE_CLOUD_PROJECT environment variable is required.",
            "",
            0,
            0,
        )

    actual_model = model_name.replace("gcp-", "").replace("-", "_")
    if actual_model == "chirp3":
        actual_model = "chirp_3"

    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not location:
        if "chirp" in actual_model:
            location = "us"
        else:
            location = "global"

    CHUNK_SIZE_SEC = 900  # 15 minutes
    should_chunk = duration_seconds and duration_seconds > 1800

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    client_options = None
    if location != "global":
        api_endpoint = f"{location}-speech.googleapis.com"
        client_options = ClientOptions(api_endpoint=api_endpoint)

    client = speech_v2.SpeechClient(client_options=client_options)

    if language == "en":
        language = "en-US"

    decoding_config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language],
        model=actual_model,
    )
    decoding_config.features = speech_v2.RecognitionFeatures(
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )

    full_transcript_parts = []
    full_srt_entries = []
    total_in_tok = 0
    total_out_tok = 0

    if should_chunk:
        print(
            f"Audio is long ({duration_seconds}s). "
            "Chunking and processing in parallel..."
        )
        assert duration_seconds is not None

        with tempfile.TemporaryDirectory() as temp_dir:
            num_chunks = int((duration_seconds + CHUNK_SIZE_SEC - 1) // CHUNK_SIZE_SEC)
            chunk_files = []

            # 1. Create Chunks (locally)
            for i in range(num_chunks):
                start_offset = i * CHUNK_SIZE_SEC
                # Use .flac for better quality/reliability with STT
                chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.flac")

                cmd = [
                    "ffmpeg",
                    "-ss",
                    str(start_offset),
                    "-t",
                    str(CHUNK_SIZE_SEC),
                    "-i",
                    audio_path,
                    "-c:a",
                    "flac",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-loglevel",
                    "error",
                    chunk_path,
                ]
                try:
                    subprocess.run(cmd, check=True)
                    chunk_files.append((i, chunk_path, start_offset))
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Failed to create chunk {i}: {e}")

            print(
                f"Created {len(chunk_files)} chunks. "
                "Uploading and submitting batches..."
            )

            # 2. Upload Chunks and Prepare Batches
            # Max files per request is typically limited (e.g. 5 or 15).
            # We'll batch requests to be safe.
            FILES_PER_BATCH = 5

            # Map gcs_uri -> (chunk_index, chunk_offset, blob_object)
            chunk_map = {}

            for i, local_path, offset in chunk_files:
                blob_name = f"temp/ytd_chunk_{uuid.uuid4()}.flac"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_path)
                gcs_uri = f"gs://{bucket_name}/{blob_name}"
                chunk_map[gcs_uri] = (i, offset, blob)

            # 3. Submit Batches
            sorted_uris = sorted(chunk_map.keys(), key=lambda k: chunk_map[k][0])
            all_results_map = {}  # uri -> (text, srt_entries)

            # Process in batches
            for b_idx in range(0, len(sorted_uris), FILES_PER_BATCH):
                batch_uris = sorted_uris[b_idx : b_idx + FILES_PER_BATCH]
                print(
                    f"Submitting batch {b_idx // FILES_PER_BATCH + 1} "
                    f"({len(batch_uris)} files)..."
                )

                batch_files_metadata = [
                    speech_v2.BatchRecognizeFileMetadata(uri=u) for u in batch_uris
                ]

                # Use GCS output config for reliability
                output_bucket_uri = f"gs://{bucket_name}/transcripts/"
                recognition_output_config = speech_v2.RecognitionOutputConfig(
                    gcs_output_config=speech_v2.GcsOutputConfig(uri=output_bucket_uri),
                )

                request = speech_v2.BatchRecognizeRequest(
                    recognizer=f"projects/{project_id}/locations/{location}/recognizers/_",
                    config=decoding_config,
                    files=batch_files_metadata,
                    recognition_output_config=recognition_output_config,
                )

                operation = client.batch_recognize(request=request)

                # Wait for this batch to complete
                # (We could parallelize batches too, but simple batching is usually
                # fast enough)
                print("Waiting for batch completion...")
                response = operation.result()

                # Process results for this batch
                for uri, result in response.results.items():
                    if uri in chunk_map:
                        idx, offset, _ = chunk_map[uri]
                        all_results_map[uri] = result

            # 4. Stitch Results
            # Sort by chunk index to ensure order
            sorted_results = sorted(
                all_results_map.items(), key=lambda item: chunk_map[item[0]][0]
            )

            srt_counter = 1
            for uri, result in sorted_results:
                idx, offset, blob = chunk_map[uri]

                t_text, t_srt_entries, next_ctr = _process_gcp_batch_result(
                    result, storage_client, offset, srt_counter
                )

                # Check for usage metadata in batch result if available
                # Speech V2 BatchRecognizeResponse metadata is at the top level usually
                # but can be per-file in some versions/configs.
                # For now we'll rely on the fact that if it's there, we should sum it.
                if hasattr(result, "metadata") and result.metadata:
                    total_in_tok += getattr(result.metadata, "prompt_token_count", 0)
                    total_out_tok += getattr(
                        result.metadata, "candidates_token_count", 0
                    )

                if t_text:
                    full_transcript_parts.append(t_text)
                if t_srt_entries:
                    full_srt_entries.extend(t_srt_entries)

                srt_counter = next_ctr

                # Cleanup Input Blob
                try:
                    blob.delete()
                except Exception:
                    pass

        return (
            " ".join(full_transcript_parts),
            "\n".join(full_srt_entries),
            total_in_tok,
            total_out_tok,
        )

    else:
        # Non-chunked (single file)
        use_inline = False
        if duration_seconds is not None and duration_seconds < 3600:
            use_inline = True

        blob_name = f"temp/ytd_audio_{uuid.uuid4()}.m4a"
        blob = bucket.blob(blob_name)

        try:
            blob.upload_from_filename(audio_path)
        except Exception as e:
            return f"Error uploading to GCS: {e}", "", 0, 0

        gcs_uri = f"gs://{bucket_name}/{blob_name}"

        file_metadata = speech_v2.BatchRecognizeFileMetadata(uri=gcs_uri)

        if use_inline:
            recognition_output_config = speech_v2.RecognitionOutputConfig(
                inline_response_config=speech_v2.InlineOutputConfig(),
            )
        else:
            output_bucket_uri = f"gs://{bucket_name}/transcripts/"
            recognition_output_config = speech_v2.RecognitionOutputConfig(
                gcs_output_config=speech_v2.GcsOutputConfig(uri=output_bucket_uri),
            )

        request = speech_v2.BatchRecognizeRequest(
            recognizer=f"projects/{project_id}/locations/{location}/recognizers/_",
            config=decoding_config,
            files=[file_metadata],
            recognition_output_config=recognition_output_config,
        )

        print(f"Starting transcription for {gcs_uri}...", flush=True)
        operation = client.batch_recognize(request=request)
        response = operation.result()

        t_text = ""
        t_srt = ""

        if gcs_uri in response.results:
            result = response.results[gcs_uri]
            t_text, t_srt_entries, _ = _process_gcp_batch_result(
                result, storage_client, 0.0, 1
            )
            t_srt = "\n".join(t_srt_entries)
        else:
            t_text = f"Error: No result found for {gcs_uri}"

        try:
            blob.delete()
        except Exception:
            pass

        return t_text, t_srt, 0, 0


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
