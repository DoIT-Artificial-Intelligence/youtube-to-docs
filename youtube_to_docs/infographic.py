import base64
import os
from typing import Optional, Tuple

import requests
from google import genai
from google.genai import types

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


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

    prompt = (
        "Create a visually appealing infographic summarizing the following "
        "video content.\n"
        f"Video Title: {video_title}\n\n"
        f"Summary:\n{summary_text}\n\n"
        "The infographic should be easy to read, professional, and capture "
        "the key points. Ensure any text in the infographic is in "
        f"{language}."
    )

    try:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            print("Error: GEMINI_API_KEY not found for infographic generation")
            return None, 0, 0

        client = genai.Client(api_key=GEMINI_API_KEY)

        if image_model.startswith("gemini"):
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
                image_config=types.ImageConfig(),
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
                    output_tokens = chunk.usage_metadata.candidates_token_count or 0

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

        elif image_model.startswith("imagen"):
            if len(prompt) > 1000:
                print(
                    f"Warning: Prompt length ({len(prompt)}) exceeds Imagen "
                    "limit (1000). Skipping infographic generation for "
                    f"{image_model}."
                )
                return None, 0, 0

            response = client.models.generate_images(
                model=image_model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    person_generation=types.PersonGeneration.DONT_ALLOW,
                    aspect_ratio="16:9",
                ),
            )

            if response.generated_images:
                return response.generated_images[0].image.image_bytes, 0, 1000

            print(f"No image data found in response from {image_model}")
            return None, 0, 0

        elif "titan-image-generator" in image_model or "nova-canvas" in image_model:
            try:
                actual_model_id = image_model
                if actual_model_id.startswith("bedrock-"):
                    actual_model_id = actual_model_id.replace("bedrock-", "")

                # If it doesn't start with amazon. but it is one of these, add it
                if not actual_model_id.startswith("amazon."):
                    actual_model_id = f"amazon.{actual_model_id}"

                # Add :0 if missing
                if not actual_model_id.endswith(":0"):
                    actual_model_id = f"{actual_model_id}:0"

                # Bedrock image models have a 1024 character limit for the prompt
                if len(prompt) > 1024:
                    print(
                        f"Warning: Prompt length ({len(prompt)}) exceeds Bedrock "
                        "limit (1024). Skipping infographic generation for "
                        f"{image_model}."
                    )
                    return None, 0, 0

                aws_bearer_token_bedrock = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
                endpoint = (
                    f"https://bedrock-runtime.us-east-1.amazonaws.com/model/"
                    f"{actual_model_id}/invoke"
                )
                payload = {
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {"text": prompt},
                    "imageGenerationConfig": {
                        "numberOfImages": 1,
                        "quality": "standard",
                        "cfgScale": 8.0 if "titan" in image_model else 6.5,
                        "width": 1280,
                        "height": 768 if "titan" in image_model else 720,
                    },
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
                    images = response_json.get("images", [])
                    if images:
                        image_data = base64.b64decode(images[0])
                        output_tokens = 1000
                        if "nova-canvas" in image_model:
                            output_tokens = 4000
                        return image_data, 0, output_tokens
                    else:
                        print(f"No images in Bedrock response: {response_json}")
                else:
                    print(f"Bedrock API Error {response.status_code}: {response.text}")
            except KeyError:
                print("Error: AWS_BEARER_TOKEN_BEDROCK required for Bedrock models.")
            except Exception as e:
                print(f"Bedrock Infographic Error: {e}")
            return None, 0, 0

        elif image_model.startswith("foundry"):
            try:
                if OpenAI is None:
                    print("Error: openai library not installed.")
                    return None, 0, 0

                AZURE_FOUNDRY_ENDPOINT = os.environ["AZURE_FOUNDRY_ENDPOINT"]
                AZURE_FOUNDRY_API_KEY = os.environ["AZURE_FOUNDRY_API_KEY"]
                actual_model_name = image_model.replace("foundry-", "")

                openai_client = OpenAI(
                    base_url=AZURE_FOUNDRY_ENDPOINT, api_key=AZURE_FOUNDRY_API_KEY
                )

                response = openai_client.images.generate(
                    model=actual_model_name,
                    prompt=prompt,
                    n=1,
                    size="1536x1024" if "gpt-image-1.5" in image_model else "1024x1024",
                    response_format="b64_json",
                )

                image_data = base64.b64decode(response.data[0].b64_json)
                # Pricing for gpt-image-1.5 is $0.034/image -> 3400 units
                output_tokens = 3400 if "gpt-image-1.5" in image_model else 1000

                return image_data, 0, output_tokens
            except KeyError:
                print(
                    "Error: AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY "
                    "required for Foundry models."
                )
            except Exception as e:
                print(f"Foundry Infographic Error: {e}")
            return None, 0, 0

        else:
            print(f"Image model {image_model} not supported yet.")
            return None, 0, 0

    except Exception as e:
        print(f"Infographic generation error with {image_model}: {e}")
        return None, 0, 0
