import base64
import os
from typing import Optional, Tuple

# Gemini image models that only render at 1K; the rest render at 4K.
GEMINI_1K_ONLY_MODELS = ("gemini-3.1-flash-lite-image",)


def image_file_extension(image_bytes: bytes) -> str:
    """Returns the file extension matching the image data's actual format."""
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def build_infographic_prompt(
    summary_text: str,
    video_title: str,
    language: str = "en",
) -> str:
    """Build the text prompt used to generate an infographic.

    Kept separate from generate_infographic so callers can persist the exact
    prompt that was sent to the image model.
    """
    return (
        "Create a visually appealing infographic summarizing the following "
        "video content. Do not include any people in the infographic.\n"
        f"Video Title: {video_title}\n\n"
        f"Summary:\n{summary_text}\n\n"
        "The infographic should be easy to read, professional, and capture "
        "the key points. Ensure any text in the infographic is in "
        f"{language}."
    )


def generate_infographic(
    image_model: Optional[str],
    summary_text: str,
    video_title: str,
    language: str = "en",
) -> Tuple[Optional[bytes], int, int]:
    """
    Generates an infographic image using the specified model.
    Returns (image_bytes, input_tokens, output_tokens).
    """
    if not image_model:
        return None, 0, 0

    prompt = build_infographic_prompt(summary_text, video_title, language)

    try:
        if image_model.startswith("gemini"):
            from google import genai
            from google.genai import types

            GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                print("Error: GEMINI_API_KEY not found for infographic generation")
                return None, 0, 0

            client = genai.Client(api_key=GEMINI_API_KEY)
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                response_modalities=[
                    "IMAGE",
                    "TEXT",
                ],
                # Landscape so it works as a video thumbnail and in the MP4,
                # at the highest resolution the model supports.
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size="1K" if image_model in GEMINI_1K_ONLY_MODELS else "4K",
                ),
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.HIGH
                ),
            )

            image_data = None
            input_tokens = 0
            output_tokens = 0

            for chunk in client.models.generate_content_stream(
                model=image_model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.usage_metadata:
                    input_tokens = chunk.usage_metadata.prompt_token_count or 0
                    # Thinking tokens are billed as output.
                    output_tokens = (
                        chunk.usage_metadata.candidates_token_count or 0
                    ) + (chunk.usage_metadata.thoughts_token_count or 0)

                if (
                    chunk.candidates is None
                    or not chunk.candidates
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue

                for part in chunk.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        # Found the image data
                        image_data = part.inline_data.data

            if image_data:
                if output_tokens == 0:
                    output_tokens = 1290
                return image_data, input_tokens, output_tokens

            print(f"No image data found in response from {image_model}")
            return None, 0, 0

        elif "titan-image-generator" in image_model or "nova-canvas" in image_model:
            print(
                f"{image_model} has reached end of life on Amazon Bedrock. "
                "Use a Gemini or Azure Foundry image model instead."
            )
            return None, 0, 0

        elif image_model.startswith("foundry"):
            from openai import OpenAI

            try:
                AZURE_FOUNDRY_ENDPOINT = os.environ["AZURE_FOUNDRY_ENDPOINT"]
                AZURE_FOUNDRY_API_KEY = os.environ["AZURE_FOUNDRY_API_KEY"]
            except KeyError:
                print(
                    "Error: AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY "
                    "required for Foundry models."
                )
                return None, 0, 0

            client = OpenAI(
                base_url=AZURE_FOUNDRY_ENDPOINT, api_key=AZURE_FOUNDRY_API_KEY
            )
            response = client.images.generate(
                model=image_model.replace("foundry-", "", 1),
                prompt=prompt,
                n=1,
                # 16:9 to match the Gemini infographics, at the largest size
                # (longest edge 3840) and the highest quality.
                size="3840x2160",
                # "max" isn't in the SDK's quality type yet.
                extra_body={"quality": "max"},
            )

            if not response.data or not response.data[0].b64_json:
                print(f"No image data found in response from {image_model}")
                return None, 0, 0

            input_tokens = 0
            output_tokens = 0
            if response.usage:
                input_tokens = response.usage.input_tokens or 0
                output_tokens = response.usage.output_tokens or 0
            return (
                base64.b64decode(response.data[0].b64_json),
                input_tokens,
                output_tokens,
            )

        else:
            print(f"Image model {image_model} not supported yet.")
            return None, 0, 0

    except Exception as e:
        print(f"Infographic generation error with {image_model}: {e}")
        return None, 0, 0
