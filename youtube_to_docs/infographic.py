import os
from typing import Optional, Tuple

from google import genai
from google.genai import types


def generate_infographic(
    image_model: Optional[str], summary_text: str, video_title: str
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
        "the key points."
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
            # Although the example uses streaming, for a single image return,
            # we can iterate the stream or just use generate_content if applicable.
            # Sticking to the provided example using stream.
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
                # Fallback for output tokens if 0 (e.g. older API behavior or missing)
                # "Output images up to 1024x1024px consume 1290 tokens"
                if output_tokens == 0:
                    output_tokens = 1290
                return image_data, input_tokens, output_tokens

            print(f"No image data found in response from {image_model}")
            return None, 0, 0

        elif image_model.startswith("imagen"):
            response = client.models.generate_images(
                model=image_model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    person_generation=types.PersonGeneration.ALLOW_ALL,
                    aspect_ratio="16:9",
                ),
            )

            if response.generated_images:
                # The SDK example shows .image.save(), implying .image wraps the data.
                # Assuming .image_bytes exists or similar based on previous code.
                # If .image is a wrapper with bytes, it usually has
                # image_bytes property.
                # Imagen pricing is per image. We treat 1 image as 1000 output units
                # to align with prices.py logic (0.04/image -> 40.0/1M tokens)
                return response.generated_images[0].image.image_bytes, 0, 1000

            print(f"No image data found in response from {image_model}")
            return None, 0, 0

        else:
            print(f"Image model {image_model} not supported yet.")
            return None, 0, 0

    except Exception as e:
        print(f"Infographic generation error with {image_model}: {e}")
        return None, 0, 0
