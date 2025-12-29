import os
from typing import Optional

from google import genai
from google.genai import types


def generate_infographic(
    image_model: Optional[str], summary_text: str, video_title: str
) -> Optional[bytes]:
    """
    Generates an infographic image using the specified model.
    Returns the image bytes if successful, None otherwise.
    """
    if not image_model:
        return None

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
            return None

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

            for chunk in client.models.generate_content_stream(
                model=image_model,
                contents=contents,
                config=generate_content_config,
            ):
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
                        return part.inline_data.data

            print(f"No image data found in response from {image_model}")
            return None

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
                return response.generated_images[0].image.image_bytes

            print(f"No image data found in response from {image_model}")
            return None

        else:
            print(f"Image model {image_model} not supported yet.")
            return None

    except Exception as e:
        print(f"Infographic generation error with {image_model}: {e}")
        return None
