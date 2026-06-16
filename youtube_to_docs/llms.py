import json
import mimetypes
import os
import re
import subprocess
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from rich import print as rprint

from youtube_to_docs.providers import (
    BaseProvider,
    LLMProvider,
    MultimodalProvider,
    STTProvider,
    TranslationProvider,
    TTSProvider,
)

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3: Any = None

    class ClientError(Exception):
        pass


from youtube_to_docs.prices import PRICES
from youtube_to_docs.utils import (
    add_question_numbers,
    get_gcp_client,
    normalize_model_name,
)


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


class GeminiProvider(
    BaseProvider, LLMProvider, STTProvider, MultimodalProvider, TTSProvider
):
    def generate_content(self, prompt: str, **kwargs) -> Tuple[str, int, int]:
        try:
            from google import genai
            from google.genai import types

            GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
            google_genai_client = genai.Client(api_key=GEMINI_API_KEY)
            response = google_genai_client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=prompt)]
                    )
                ],
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
            print(f"Gemini API Error: {e}")
            return f"Error: {e}", 0, 0

    def transcribe(
        self,
        audio_path: str,
        url: str,
        language: str = "en",
        duration_seconds: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, str, int, int]:
        srt = kwargs.get("srt", False)
        try:
            from google import genai
            from google.genai import types

            GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
            client = genai.Client(api_key=GEMINI_API_KEY)

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            mime_type = mimetypes.guess_type(audio_path)[0] or "audio/x-m4a"

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
                            mime_type=mime_type,
                            data=audio_bytes,
                        ),
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]

            print(f"Starting transcription with model: {self.model_name}...")
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )

            response_text = response.text or ""
            input_tokens = 0
            output_tokens = 0
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0

            if srt:
                return "", response_text, input_tokens, output_tokens
            return response_text, "", input_tokens, output_tokens

        except KeyError:
            return "Error: GEMINI_API_KEY not found", "", 0, 0
        except Exception as e:
            print(f"Gemini STT Error: {e}")
            return f"Error: {e}", "", 0, 0

    def generate_alt_text(
        self, image_bytes: bytes, language: str = "en", **kwargs
    ) -> Tuple[str, int, int]:
        try:
            from google import genai
            from google.genai import types

            GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
            client = genai.Client(api_key=GEMINI_API_KEY)

            prompt = (
                f"Please provide a descriptive alt text for this infographic "
                f"in {language}. "
                "The alt text should describe the visual layout and key information "
                "presented, making it accessible for someone who cannot see the image. "
                "Start the response immediately with the alt text."
            )

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

            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )

            response_text = response.text or ""
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

    def generate_speech(
        self, text: str, voice: str, language_code: Optional[str] = None, **kwargs
    ) -> Tuple[bytes, int]:
        try:
            from google import genai
            from google.genai import types

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY environment variable not set.")
                return b"", 0

            client = genai.Client(api_key=api_key)

            # Gemini TTS has limits. We chunk the text and concatenate the results.
            # Using 5000 chars as a safe chunk size (similar to GCP).
            text_bytes = text.encode("utf-8")
            chunk_size = 4800

            if len(text_bytes) <= 5000:
                chunks = [text]
            else:
                from youtube_to_docs.tts import _chunk_text_by_bytes

                chunks = _chunk_text_by_bytes(text, chunk_size)

            all_audio = b""
            for i, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    rprint(f"  Synthesizing Gemini chunk {i + 1}/{len(chunks)}...")

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=chunk,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            language_code=language_code,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice,
                                )
                            ),
                        ),
                    ),
                )

                if (
                    response.candidates
                    and response.candidates[0].content
                    and response.candidates[0].content.parts
                    and response.candidates[0].content.parts[0].inline_data
                    and response.candidates[0].content.parts[0].inline_data.data
                ):
                    all_audio += (
                        response.candidates[0].content.parts[0].inline_data.data
                    )
                else:
                    print(f"Error: No audio data in response for chunk {i + 1}.")

            return all_audio, 24000

        except Exception as e:
            print(f"Error generating Gemini speech: {e}")
            return b"", 0

    def translate(self, text: str, target_lang: str, **kwargs) -> str:
        prompt = (
            f"Please translate the following text to {target_lang}. "
            "Return only the translated text without any preamble or explanation."
            "\n\n"
            f"{text}"
        )
        translated, _, _ = self.generate_content(prompt)
        return translated


class VertexProvider(BaseProvider, LLMProvider, MultimodalProvider):
    def generate_content(self, prompt: str, **kwargs) -> Tuple[str, int, int]:
        response_text: str = ""
        input_tokens: int = 0
        output_tokens: int = 0
        try:
            import subprocess

            import google.auth
            from google.auth.exceptions import RefreshError

            vertex_project_id = os.environ["PROJECT_ID"]
            actual_model_name = self.model_name.replace("vertex-", "")

            if actual_model_name.startswith("claude"):
                from anthropic import AnthropicVertex

                vertex_location = os.environ.get("VERTEX_LOCATION", "us-east5")

                def make_anthropic_call(creds):
                    client = AnthropicVertex(
                        project_id=vertex_project_id,
                        region=vertex_location,
                        credentials=creds,
                    )
                    return client.messages.create(
                        model=actual_model_name,
                        max_tokens=8192,
                        messages=[{"role": "user", "content": prompt}],
                    )

                res = get_gcp_client(google.auth.default, "Vertex AI Credentials")
                if res is None:
                    return (
                        "Error: Vertex AI Credentials could not be initialized.",
                        0,
                        0,
                    )
                vertex_credentials, _ = res

                try:
                    message = make_anthropic_call(vertex_credentials)
                except Exception as e:
                    from google.auth.exceptions import RefreshError

                    is_refresh_error = (
                        isinstance(e, RefreshError)
                        or "RefreshError" in str(e)
                        or "expired" in str(e).lower()
                    )
                    if is_refresh_error:
                        print(
                            "Vertex AI Credentials expired. Launching gcloud login..."
                        )
                        try:
                            subprocess.run(
                                ["gcloud", "auth", "application-default", "login"],
                                check=True,
                            )
                            res = get_gcp_client(
                                google.auth.default, "Vertex AI Credentials"
                            )
                            if res is None:
                                return "Error: Re-authentication failed.", 0, 0
                            vertex_credentials, _ = res
                            message = make_anthropic_call(vertex_credentials)
                        except Exception as re_err:
                            error_msg = f"Re-authentication failed: {re_err}"
                            print(error_msg)
                            return f"Error: {error_msg}", 0, 0
                    else:
                        raise e

                if message.content:
                    response_text = "".join(
                        block.text
                        for block in message.content
                        if getattr(block, "type", None) == "text"
                    )
                else:
                    response_text = f"Unexpected response format: {message}"

                if message.usage:
                    input_tokens = message.usage.input_tokens or 0
                    output_tokens = message.usage.output_tokens or 0
            elif actual_model_name.startswith("gemma"):
                raise NotImplementedError(
                    f"Vertex AI Gemma models ('{self.model_name}') require a "
                    "dedicated deployment. Use the 'gemma-' prefix "
                    "(without 'vertex-') to run via the Google GenAI "
                    "client instead."
                )
            elif actual_model_name.startswith("gemini"):
                from google import genai
                from google.genai import types

                vertex_location = os.environ.get("VERTEX_LOCATION", "us-east5")
                vertex_api_key = os.environ.get("VERTEXAI_API_KEY") or os.environ.get(
                    "VERTEX_API_KEY"
                )
                if vertex_api_key:
                    client = genai.Client(
                        vertexai=True,
                        api_key=vertex_api_key,
                    )
                else:
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
        except NotImplementedError:
            raise
        except Exception as e:
            print(f"Vertex Request Error: {e}")
            response_text = f"Error: {e}"

        return response_text, input_tokens, output_tokens

    def generate_alt_text(
        self, image_bytes: bytes, language: str = "en", **kwargs
    ) -> Tuple[str, int, int]:
        # Vertex AI multimodal usually uses the same logic as Gemini
        # But we need to ensure the client is initialized with vertexai=True
        # For now, we leverage the existing generate_alt_text function if
        # it handles vertex.
        # But wait, generate_alt_text currently only handles gemini and bedrock.
        # Let's fix that.
        actual_model_name = self.model_name.replace("vertex-", "")
        if not actual_model_name.startswith("gemini"):
            return (
                f"Error: Multimodal alt text not yet implemented for {self.model_name}",
                0,
                0,
            )

        try:
            from google import genai
            from google.genai import types

            vertex_project_id = os.environ["PROJECT_ID"]
            vertex_location = os.environ.get("VERTEX_LOCATION", "us-east5")
            vertex_api_key = os.environ.get("VERTEXAI_API_KEY") or os.environ.get(
                "VERTEX_API_KEY"
            )
            if vertex_api_key:
                client = genai.Client(
                    vertexai=True,
                    api_key=vertex_api_key,
                )
            else:
                client = genai.Client(
                    vertexai=True,
                    project=vertex_project_id,
                    location=vertex_location,
                    http_options=types.HttpOptions(api_version="v1"),
                )

            prompt = (
                f"Please provide a descriptive alt text for this infographic "
                f"in {language}. "
                "The alt text should describe the visual layout and key information "
                "presented, making it accessible for someone who cannot see the image. "
                "Start the response immediately with the alt text."
            )

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

            response = client.models.generate_content(
                model=actual_model_name,
                contents=contents,
            )

            response_text = response.text or ""
            response_text = re.sub(
                r"^(Alt text[:\-\s]+)", "", response_text, flags=re.IGNORECASE
            ).strip()

            input_tokens = 0
            output_tokens = 0
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0

            return response_text, input_tokens, output_tokens
        except Exception as e:
            print(f"Vertex Alt Text Error: {e}")
            return f"Error: {e}", 0, 0


class BedrockProvider(BaseProvider, LLMProvider, MultimodalProvider):
    def generate_content(self, prompt: str, **kwargs) -> Tuple[str, int, int]:
        response_text: str = ""
        input_tokens: int = 0
        output_tokens: int = 0
        try:
            aws_bearer_token_bedrock = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
            actual_model_name = self.model_name.replace("bedrock-", "")
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

        return response_text, input_tokens, output_tokens

    def generate_alt_text(
        self, image_bytes: bytes, language: str = "en", **kwargs
    ) -> Tuple[str, int, int]:
        try:
            import base64

            aws_bearer_token_bedrock = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
            actual_model_name = self.model_name.replace("bedrock-", "")

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

            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            prompt = (
                f"Please provide a descriptive alt text for this infographic "
                f"in {language}. "
                "The alt text should describe the visual layout and key information "
                "presented, making it accessible for someone who cannot see the image. "
                "Start the response immediately with the alt text."
            )

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


class AzureFoundryProvider(BaseProvider, LLMProvider):
    def generate_content(self, prompt: str, **kwargs) -> Tuple[str, int, int]:
        response_text: str = ""
        input_tokens: int = 0
        output_tokens: int = 0
        try:
            from openai import OpenAI

            AZURE_FOUNDRY_ENDPOINT = os.environ["AZURE_FOUNDRY_ENDPOINT"]
            AZURE_FOUNDRY_API_KEY = os.environ["AZURE_FOUNDRY_API_KEY"]
            actual_model_name = self.model_name.replace("foundry-", "")
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
            response_text = completion.choices[0].message.content or ""
            if completion.usage:
                input_tokens = completion.usage.prompt_tokens or 0
                output_tokens = completion.usage.completion_tokens or 0
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


class GCPProvider(BaseProvider, STTProvider, TTSProvider, TranslationProvider):
    def transcribe(
        self,
        audio_path: str,
        url: str,
        language: str = "en",
        duration_seconds: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, str, int, int]:
        return _transcribe_gcp(
            self.model_name, audio_path, url, language, duration_seconds
        )

    def generate_speech(
        self, text: str, voice: str, language_code: Optional[str] = None, **kwargs
    ) -> Tuple[bytes, int]:
        try:
            from google.cloud import texttospeech
        except ImportError:
            print(
                "Error: google-cloud-texttospeech is required for GCP TTS models. "
                "Install with `pip install '.[gcp]'`"
            )
            return b"", 0

        try:
            client = get_gcp_client(
                texttospeech.TextToSpeechClient, "GCP Text-to-Speech"
            )
            if client is None:
                return b"", 0

            if language_code:
                full_voice_name = f"{language_code}-Chirp3-HD-{voice}"
            else:
                full_voice_name = f"en-US-Chirp3-HD-{voice}"

            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code or "en-US",
                name=full_voice_name,
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
            )

            text_bytes = text.encode("utf-8")
            rprint(
                f"[cyan]GCP TTS: input text length: {len(text)} chars, "
                f"{len(text_bytes)} bytes[/cyan]"
            )
            if len(text_bytes) <= 5000:
                input_text = texttospeech.SynthesisInput(text=text)
                response = client.synthesize_speech(
                    input=input_text,
                    voice=voice_params,
                    audio_config=audio_config,
                )
                return response.audio_content, 24000
            else:
                rprint(
                    f"[yellow]Text is long ({len(text_bytes)} bytes). "
                    "Chunking for GCP TTS...[/yellow]"
                )
                from youtube_to_docs.tts import _chunk_text_by_bytes

                chunks = _chunk_text_by_bytes(text, 4800)
                all_audio = b""
                for i, chunk in enumerate(chunks):
                    rprint(f"  Synthesizing chunk {i + 1}/{len(chunks)}...")
                    input_text = texttospeech.SynthesisInput(text=chunk)
                    response = client.synthesize_speech(
                        input=input_text,
                        voice=voice_params,
                        audio_config=audio_config,
                    )
                    all_audio += response.audio_content

                return all_audio, 24000

        except Exception as e:
            print(f"Error generating speech with GCP TTS: {e}")
            return b"", 0

    def translate(self, text: str, target_lang: str, **kwargs) -> str:
        from youtube_to_docs.translate import _translate_gcp

        translated, _, _ = _translate_gcp(text, target_lang)
        return translated


class AWSProvider(BaseProvider, STTProvider, TTSProvider, TranslationProvider):
    def transcribe(
        self,
        audio_path: str,
        url: str,
        language: str = "en",
        duration_seconds: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, str, int, int]:
        return _transcribe_aws(
            self.model_name, audio_path, url, language, duration_seconds
        )

    def generate_speech(
        self, text: str, voice: str, language_code: Optional[str] = None, **kwargs
    ) -> Tuple[bytes, int]:
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError:
            print(
                "Error: boto3 is required for AWS Polly. "
                "Install with `pip install boto3`"
            )
            return b"", 0

        try:
            polly = boto3.client("polly")
            chunk_size = 2000
            text_bytes = text.encode("utf-8")

            rprint(
                f"[cyan]AWS Polly: input text length: {len(text)} chars, "
                f"{len(text_bytes)} bytes[/cyan]"
            )

            audio_stream = b""
            from youtube_to_docs.tts import _chunk_text_by_bytes

            if len(text_bytes) <= chunk_size:
                chunks = [text]
            else:
                rprint(
                    f"[yellow]Text is long ({len(text_bytes)} bytes). "
                    "Chunking for AWS Polly...[/yellow]"
                )
                chunks = _chunk_text_by_bytes(text, chunk_size)

            for i, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    rprint(f"  Synthesizing chunk {i + 1}/{len(chunks)}...")

                response = polly.synthesize_speech(
                    Text=chunk,
                    OutputFormat="pcm",
                    VoiceId=voice,
                    Engine=kwargs.get("engine", "long-form"),
                )
                if "AudioStream" in response:
                    with response["AudioStream"] as stream:
                        audio_stream += stream.read()
                else:
                    print("Error: No AudioStream in Polly response")

            return audio_stream, 16000

        except (BotoCoreError, ClientError) as error:
            print(f"Error generating speech with AWS Polly: {error}")
            return b"", 0
        except Exception as e:
            print(f"Error generating speech with AWS Polly: {e}")
            return b"", 0

    def translate(self, text: str, target_lang: str, **kwargs) -> str:
        from youtube_to_docs.translate import _translate_aws

        translated, _, _ = _translate_aws(text, target_lang)
        return translated


def _query_llm(model_name: str, prompt: str) -> Tuple[str, int, int]:
    """
    Generic function to query the specified LLM model.
    Returns (response_text, input_tokens, output_tokens).
    """
    from youtube_to_docs.providers import LLMProvider, get_provider

    try:
        provider = get_provider(model_name)
        if isinstance(provider, LLMProvider):
            return provider.generate_content(prompt)
        return f"Error: {model_name} does not support LLM tasks", 0, 0
    except NotImplementedError:
        raise
    except Exception as e:
        return f"Error: {e}", 0, 0


def generate_transcript_with_srt(
    model_name: str,
    audio_path: str,
    url: str,
    language: str = "en",
    duration_seconds: Optional[float] = None,
) -> Tuple[str, str, int, int]:
    """
    Generates both transcript text and SRT content from audio in a single call.
    Returns (transcript_text, srt_content, input_tokens, output_tokens).
    """
    from youtube_to_docs.providers import STTProvider, get_provider

    try:
        provider = get_provider(model_name)
        if isinstance(provider, STTProvider):
            return provider.transcribe(
                audio_path, url, language, duration_seconds, srt=True
            )
        return f"Error: STT not implemented for {model_name}", "", 0, 0
    except NotImplementedError:
        raise
    except Exception as e:
        return f"Error: {e}", "", 0, 0


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
    Returns (transcript_text, input_tokens, output_tokens).
    """
    from youtube_to_docs.providers import STTProvider, get_provider

    try:
        provider = get_provider(model_name)
        if isinstance(provider, STTProvider):
            text, srt_content, in_tok, out_tok = provider.transcribe(
                audio_path, url, language, duration_seconds, srt=srt
            )
            if srt:
                return srt_content, in_tok, out_tok
            return text, in_tok, out_tok
        return f"Error: STT not implemented for {model_name}", 0, 0
    except NotImplementedError:
        raise
    except Exception as e:
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
                pass  # Best-effort cleanup of temporary GCS blob

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

    CHUNK_SIZE_SEC = 1140  # 19 minutes
    should_chunk = duration_seconds and duration_seconds > 1140

    storage_client = get_gcp_client(storage.Client, "GCP Storage", project=project_id)
    if storage_client is None:
        return "Error: GCP Storage client could not be initialized.", "", 0, 0

    bucket = storage_client.bucket(bucket_name)

    client_options = None
    if location != "global":
        api_endpoint = f"{location}-speech.googleapis.com"
        client_options = ClientOptions(api_endpoint=api_endpoint)

    client = get_gcp_client(
        speech_v2.SpeechClient, "GCP Speech-to-Text", client_options=client_options
    )
    if client is None:
        return "Error: GCP Speech-to-Text client could not be initialized.", "", 0, 0

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
                    recognizer=(
                        f"projects/{project_id}/locations/{location}/recognizers/_"
                    ),
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
                    pass  # Best-effort cleanup of temporary GCS blob

        # Calculate duration-based cost (represented as pseudo-tokens for main.py)
        # 1,000,000 pseudo-tokens = 1 minute of audio
        # Split 50/50 between input and output
        pseudo_in_tok = 0
        pseudo_out_tok = 0
        if duration_seconds:
            total_pseudo = int(duration_seconds * (1_000_000 / 60))
            pseudo_in_tok = total_pseudo // 2
            pseudo_out_tok = total_pseudo - pseudo_in_tok

        return (
            " ".join(full_transcript_parts),
            "\n".join(full_srt_entries),
            pseudo_in_tok,
            pseudo_out_tok,
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
            pass  # Best-effort cleanup of temporary GCS blob

        # Calculate duration-based cost (represented as pseudo-tokens for main.py)
        # 1,000,000 pseudo-tokens = 1 minute of audio
        # Split 50/50 between input and output
        pseudo_in_tok = 0
        pseudo_out_tok = 0
        if duration_seconds:
            total_pseudo = int(duration_seconds * (1_000_000 / 60))
            pseudo_in_tok = total_pseudo // 2
            pseudo_out_tok = total_pseudo - pseudo_in_tok

        return t_text, t_srt, pseudo_in_tok, pseudo_out_tok


def _transcribe_aws(
    model_name: str,
    audio_path: str,
    url: str,
    language: str = "en",
    duration_seconds: Optional[float] = None,
) -> Tuple[str, str, int, int]:
    """
    Transcribes audio using AWS Transcribe (Batch).
    Returns (transcript_text, srt_content, input_tokens, output_tokens).
    Requires 'YTD_S3_BUCKET_NAME' env var for temporary storage.
    """
    if boto3 is None:
        return (
            "Error: boto3 is required for AWS Transcribe. "
            "Install with `pip install '.[aws]'`",
            "",
            0,
            0,
        )

    bucket_name = os.environ.get("YTD_S3_BUCKET_NAME")
    if not bucket_name:
        return (
            "Error: YTD_S3_BUCKET_NAME environment variable is required for "
            "AWS Transcribe.",
            "",
            0,
            0,
        )

    region = os.environ.get("AWS_REGION", "us-east-1")
    s3_client = boto3.client("s3", region_name=region)
    transcribe_client = boto3.client("transcribe", region_name=region)

    job_name = f"ytd_transcribe_{uuid.uuid4()}"
    s3_key = f"temp_audio/{job_name}.m4a"

    print(f"Uploading audio to s3://{bucket_name}/{s3_key}...")
    try:
        s3_client.upload_file(audio_path, bucket_name, s3_key)
    except ClientError as e:
        return f"Error uploading to S3: {e}", "", 0, 0

    s3_uri = f"s3://{bucket_name}/{s3_key}"

    if language == "en":
        language_code = "en-US"
    else:
        # AWS uses different codes sometimes, but en-US, es-ES etc are common
        # Simple mapping for now
        language_code = (
            language if "-" in language else f"{language}-{language.upper()}"
        )

    print(f"Starting AWS Transcribe job: {job_name}...")
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat="m4a",
            LanguageCode=language_code,
            OutputBucketName=bucket_name,
            OutputKey=f"transcripts/{job_name}.json",
            Settings={
                "ShowAlternatives": False,
            },
        )
    except ClientError as e:
        return f"Error starting Transcribe job: {e}", "", 0, 0

    # Polling
    print("Waiting for AWS Transcribe job to complete...")
    while True:
        try:
            status_resp = transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = status_resp["TranscriptionJob"]["TranscriptionJobStatus"]
            if status in ["COMPLETED", "FAILED"]:
                break
        except ClientError as e:
            print(f"Error polling job status: {e}")
        time.sleep(10)

    if status == "FAILED":
        reason = status_resp["TranscriptionJob"].get("FailureReason", "Unknown failure")
        return f"AWS Transcribe job failed: {reason}", "", 0, 0

    # Download result
    transcript_key = f"transcripts/{job_name}.json"
    print(f"Downloading transcript from s3://{bucket_name}/{transcript_key}...")
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=transcript_key)
        transcript_json = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        return f"Error downloading transcript from S3: {e}", "", 0, 0

    # Cleanup
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        s3_client.delete_object(Bucket=bucket_name, Key=transcript_key)
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")

    # Parse JSON for text and SRT
    results = transcript_json.get("results", {})
    transcript_text = ""
    transcripts = results.get("transcripts", [])
    if transcripts:
        transcript_text = transcripts[0].get("transcript", "")

    # For SRT, we need word-level timestamps
    items = results.get("items", [])
    srt_entries = []
    srt_counter = 1

    current_segment_words = []
    current_segment_len = 0

    for item in items:
        if item.get("type") == "punctuation":
            if current_segment_words:
                # Add punctuation to the last word
                word, start, end = current_segment_words.pop()
                current_segment_words.append(
                    (word + item["alternatives"][0]["content"], start, end)
                )
            continue

        word = item["alternatives"][0]["content"]
        start_sec = float(item["start_time"])
        end_sec = float(item["end_time"])

        current_segment_words.append((word, start_sec, end_sec))
        current_segment_len += len(word) + 1

        if current_segment_len > 80:
            seg_text = " ".join([w[0] for w in current_segment_words])
            seg_start = _format_srt_time(current_segment_words[0][1])
            seg_end = _format_srt_time(current_segment_words[-1][2])
            srt_entries.append(
                f"{srt_counter}\n{seg_start} --> {seg_end}\n{seg_text}\n"
            )
            srt_counter += 1
            current_segment_words = []
            current_segment_len = 0

    if current_segment_words:
        seg_text = " ".join([w[0] for w in current_segment_words])
        seg_start = _format_srt_time(current_segment_words[0][1])
        seg_end = _format_srt_time(current_segment_words[-1][2])
        srt_entries.append(f"{srt_counter}\n{seg_start} --> {seg_end}\n{seg_text}\n")

    srt_content = "\n".join(srt_entries)

    return transcript_text, srt_content, 0, 0


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
    from youtube_to_docs.providers import MultimodalProvider, get_provider

    try:
        provider = get_provider(model_name)
        if isinstance(provider, MultimodalProvider):
            return provider.generate_alt_text(image_bytes, language)
        return f"Error: Multimodal not implemented for {model_name}", 0, 0
    except NotImplementedError:
        raise
    except Exception as e:
        return f"Error: {e}", 0, 0


def suggest_corrected_captions(
    model_name: str,
    srt_content: str,
    speakers_text: str = "",
) -> Tuple[str, int, int]:
    """
    Suggests WCAG 2.1 Level AA compliant corrected captions for an SRT file,
    per Section 508 guidance (https://www.section508.gov/create/captions-transcripts/).

    Fixes punctuation, capitalization, and adds speaker identification when a
    known speaker list is provided. Returns only segments that require changes.

    If no corrections are needed, the response will be the string 'NO_CHANGES'.
    Returns (corrected_srt_or_no_changes, input_tokens, output_tokens).
    """
    speakers_section = ""
    if speakers_text and speakers_text.strip():
        speakers_section = (
            "\n\nKnown speakers (from speaker extraction):\n"
            f"{speakers_text.strip()}\n\n"
            "Speaker identification rules (Section 508 / WCAG 2.1 AA):\n"
            "- When a known speaker begins talking, prepend their name as "
            "[Name] at the start of the first segment where they speak.\n"
            "- Only add [Name] again when the speaker changes.\n"
            "- If the speaker cannot be confidently identified from context, "
            "do not add a speaker label.\n"
        )

    prompt = (
        "You are a professional caption editor producing captions compliant with "
        "WCAG 2.1 Level AA and Section 508 guidelines "
        "(https://www.section508.gov/create/captions-transcripts/).\n\n"
        "I have provided an SRT subtitle file generated by speech-to-text software "
        "that needs corrections.\n"
        f"{speakers_section}"
        "Correction rules:\n"
        "1. Fix punctuation (commas, periods, question marks, colons, em dashes, "
        "ellipses) and capitalization throughout.\n"
        "2. Retain filler words (e.g., 'uh', 'um') unless removing them does not "
        "change the accessible meaning — per Section 508 guidance, include filler "
        "words when they aid comprehension.\n"
        "3. Add bracketed descriptors (e.g., [applause], [laughter], [crosstalk]) "
        "for meaningful non-speech audio only if clearly implied by context.\n"
        "4. If a grammatically complete sentence naturally spans multiple consecutive "
        "segments, you may merge them into a single segment: use the start timestamp "
        "of the first segment and the end timestamp of the last, keeping the first "
        "segment's number.\n"
        "5. Do NOT add new segments that did not exist in the original.\n"
        "6. Output ONLY the segments that have changes. Do not output unchanged "
        "segments.\n"
        "7. Return a valid SRT file containing only the corrected or merged segments, "
        "with no additional commentary.\n"
        "8. If no corrections are needed at all, return exactly: NO_CHANGES\n\n"
        "SRT file:\n\n"
        f"{srt_content}"
    )
    return _query_llm(model_name, prompt)
