import argparse
import io
import os
import re
import wave
from typing import List, Optional, Tuple

import polars as pl
from rich import print as rprint

from youtube_to_docs.storage import Storage
from youtube_to_docs.utils import format_clickable_path


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Writes PCM data to a WAV file (or file-like object)."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def _chunk_text_by_bytes(text: str, max_bytes: int = 5000) -> List[str]:
    """Helper to chunk text into pieces below the byte limit."""
    if not text:
        return []

    text_bytes = text.encode("utf-8")
    if len(text_bytes) <= max_bytes:
        return [text]

    chunks = []
    # Split by sentences if possible
    sentences = re.split(r"(?<=[.!?]) +", text)
    current_chunk = ""

    for s in sentences:
        # If adding this sentence exceeds limit
        if len((current_chunk + " " + s).encode("utf-8")) > max_bytes:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = s
            else:
                # Sentence itself is too long, split by words
                words = s.split(" ")
                for w in words:
                    if len((current_chunk + " " + w).encode("utf-8")) > max_bytes:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = w
                        else:
                            # Word too long (rare), split by bytes safely
                            w_bytes = w.encode("utf-8")
                            while len(w_bytes) > max_bytes:
                                split_point = max_bytes
                                while split_point > 0:
                                    try:
                                        part = w_bytes[:split_point].decode("utf-8")
                                        break
                                    except UnicodeDecodeError:
                                        split_point -= 1
                                chunks.append(part)
                                w_bytes = w_bytes[len(part.encode("utf-8")) :]
                            current_chunk = w_bytes.decode("utf-8")
                    else:
                        current_chunk = (current_chunk + " " + w).strip()
        else:
            current_chunk = (current_chunk + " " + s).strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def generate_speech_gcp(
    text: str, voice_name: str, language_code: Optional[str] = None
) -> bytes:
    """
    Generates speech from text using Google Cloud Text-to-Speech API.
    Returns raw PCM audio bytes (LINEAR16 format, 24kHz, mono).
    Handles text longer than 5000 bytes by chunking and concatenating results.
    """
    try:
        from google.cloud import texttospeech
    except ImportError:
        print(
            "Error: google-cloud-texttospeech is required for GCP TTS models. "
            "Install with `pip install '.[gcp]'`"
        )
        return b""

    try:
        client = texttospeech.TextToSpeechClient()

        # Build the voice name from language code and voice name
        # e.g., language_code="en-US", voice_name="Kore" -> "en-US-Chirp3-HD-Kore"
        if language_code:
            full_voice_name = f"{language_code}-Chirp3-HD-{voice_name}"
        else:
            full_voice_name = f"en-US-Chirp3-HD-{voice_name}"

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code or "en-US",
            name=full_voice_name,
        )

        # Use LINEAR16 (PCM) to match Gemini TTS output format
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )

        text_bytes = text.encode("utf-8")
        if len(text_bytes) <= 5000:
            input_text = texttospeech.SynthesisInput(text=text)
            response = client.synthesize_speech(
                input=input_text,
                voice=voice,
                audio_config=audio_config,
            )
            return response.audio_content
        else:
            # GCP TTS has a 5000 byte limit for synthesize_speech.
            # We chunk the text and concatenate the PCM results.
            rprint(
                f"[yellow]Text is long ({len(text_bytes)} bytes). "
                "Chunking for GCP TTS...[/yellow]"
            )
            chunks = _chunk_text_by_bytes(text, 4800)
            all_audio = b""
            for i, chunk in enumerate(chunks):
                rprint(f"  Synthesizing chunk {i + 1}/{len(chunks)}...")
                input_text = texttospeech.SynthesisInput(text=chunk)
                response = client.synthesize_speech(
                    input=input_text,
                    voice=voice,
                    audio_config=audio_config,
                )
                all_audio += response.audio_content

            return all_audio

    except Exception as e:
        print(f"Error generating speech with GCP TTS: {e}")
        return b""


def generate_speech(
    text: str, model_name: str, voice_name: str, language_code: Optional[str] = None
) -> bytes:
    """
    Generates speech from text using the specified Gemini model and voice.
    Returns the raw PCM audio bytes.
    """
    try:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            return b""

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    language_code=language_code,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    ),
                ),
            ),
        )

        # The response structure based on the docs:
        # response.candidates[0].content.parts[0].inline_data.data
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
            and response.candidates[0].content.parts[0].inline_data
            and response.candidates[0].content.parts[0].inline_data.data
        ):
            return response.candidates[0].content.parts[0].inline_data.data
        else:
            print("Error: No audio data in response.")
            return b""

    except Exception as e:
        print(f"Error generating speech: {e}")
        return b""


def is_gcp_tts_model(model_name: str) -> bool:
    """Check if the model is a GCP Cloud TTS model."""
    return model_name.startswith("gcp-")


def parse_tts_arg(tts_arg: str) -> Tuple[str, str]:
    """
    Parses the --tts argument into (model_name, voice_name).

    For Gemini models:
        gemini-2.5-flash-preview-tts-Kore -> (gemini-2.5-flash-preview-tts, Kore)
    For GCP Chirp3 models:
        gcp-chirp3-Kore -> (gcp-chirp3, Kore)
        gcp-chirp3 -> (gcp-chirp3, Kore)  # Default voice
    """
    # Handle GCP models specially
    if tts_arg.startswith("gcp-chirp3"):
        if tts_arg == "gcp-chirp3":
            return "gcp-chirp3", "Kore"  # Default voice
        # Format: gcp-chirp3-VoiceName
        parts = tts_arg.split("-", 2)  # Split into ['gcp', 'chirp3', 'VoiceName']
        if len(parts) >= 3:
            return "gcp-chirp3", parts[2]
        return "gcp-chirp3", "Kore"

    # Gemini models: voice name is the last part after the last hyphen
    if "-" in tts_arg:
        parts = tts_arg.rsplit("-", 1)
        return parts[0], parts[1]
    return (
        tts_arg,
        "Kore",
    )  # Default to Kore if no hyphen (though user format implies hyphen)


def process_tts(
    df: pl.DataFrame,
    tts_arg: str,
    storage: Storage,
    base_dir: str = ".",
    languages: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Processes the DataFrame to generate TTS for each summary file found.
    """
    model_name, voice_name = parse_tts_arg(tts_arg)
    rprint(f"Using TTS Model: {model_name}, Voice: {voice_name}")

    # Setup Audio Directory
    # Setup Audio Directory
    audio_dir = os.path.join(base_dir, "audio-files")
    storage.ensure_directory(audio_dir)

    # Find Summary File columns
    summary_file_cols = [c for c in df.columns if c.startswith("Summary File ")]

    if not summary_file_cols:
        print("No 'Summary File ...' columns found in the CSV.")
        return df

    updated_df = df

    # Map 2-letter language codes to BCP-47 codes supported by Gemini
    lang_map = {
        "en": "en-US",
        "es": "es-US",
        "fr": "fr-FR",
        "de": "de-DE",
        "hi": "hi-IN",
        "pt": "pt-BR",
        "zh": "cmn-CN",
        "ja": "ja-JP",
        "ko": "ko-KR",
    }

    for col in summary_file_cols:
        # Extract language from column name: e.g. "... (es)" -> "es"
        col_lang = "en"
        lang_code = None
        if "(" in col and col.endswith(")"):
            code = col.split("(")[-1].strip(")")
            col_lang = code
            lang_code = lang_map.get(code, "en-US")  # Default to en-US if unknown
        else:
            col_lang = "en"
            lang_code = "en-US"  # Default for columns without language suffix

        if languages and col_lang not in languages:
            continue

        new_col_name = (
            col.replace("Summary File", "Summary Audio File") + f" {tts_arg} File"
        )
        rprint(f"Processing column: {col} -> {new_col_name} (Language: {lang_code})")

        new_col_values = []

        for row in df.iter_rows(named=True):
            summary_path = row.get(col)

            if (
                not summary_path
                or not isinstance(summary_path, str)
                or not storage.exists(summary_path)
            ):
                new_col_values.append(None)
                continue

            if row.get("Title") and row.get("URL"):
                # Extract Video ID
                video_id = "unknown"
                match = re.search(r"v=([a-zA-Z0-9_-]+)", row["URL"])
                if match:
                    video_id = match.group(1)
                elif "youtu.be/" in row["URL"]:
                    video_id = row["URL"].split("youtu.be/")[1].split("?")[0]

                # Safe Title
                safe_title = (
                    re.sub(r'[\\/*?:"><>|]', "_", row["Title"])
                    .replace("\n", " ")
                    .replace("\r", "")
                )

                # Construct Name
                audio_filename = (
                    f"{video_id} - {safe_title} - {tts_arg} ({lang_code}).wav"
                )
                summary_filename = audio_filename  # For logging purposes
            else:
                summary_filename = os.path.basename(summary_path)
                base_name = os.path.splitext(summary_filename)[0]
                audio_filename = f"{base_name} - {tts_arg}.wav"

            # Use relative path for storage
            target_path = os.path.join(audio_dir, audio_filename)

            if storage.exists(target_path):
                if hasattr(storage, "get_full_path"):
                    full_path = storage.get_full_path(target_path)
                    new_col_values.append(full_path)
                    continue
                else:
                    pass

            rprint(f"Generating audio for: {summary_filename}")

            try:
                # Read summary from storage
                # summary_path comes from row, might be Link or Path.
                text = storage.read_text(summary_path)

            except Exception as e:
                print(f"Error reading summary file {summary_path}: {e}")
                new_col_values.append(None)
                continue

            if not text.strip():
                print("Empty summary text.")
                new_col_values.append(None)
                continue

            # Generate audio using the appropriate TTS engine
            if is_gcp_tts_model(model_name):
                # GCP TTS with LINEAR16 returns PCM data that needs WAV wrapping
                pcm_data = generate_speech_gcp(text, voice_name, lang_code)
                if pcm_data:
                    try:
                        wav_io = io.BytesIO()
                        wave_file(wav_io, pcm_data)
                        wav_bytes = wav_io.getvalue()

                        saved_path = storage.write_bytes(target_path, wav_bytes)
                        rprint(f"Saved audio: {format_clickable_path(saved_path)}")
                        new_col_values.append(saved_path)
                    except Exception as e:
                        print(f"Error writing audio file: {e}")
                        new_col_values.append(None)
                else:
                    new_col_values.append(None)
            else:
                # Gemini TTS returns PCM data that needs to be wrapped in WAV
                pcm_data = generate_speech(text, model_name, voice_name, lang_code)
                if pcm_data:
                    try:
                        # Write to BytesIO then storage.write_bytes
                        wav_io = io.BytesIO()
                        wave_file(wav_io, pcm_data)
                        wav_bytes = wav_io.getvalue()

                        saved_path = storage.write_bytes(target_path, wav_bytes)
                        rprint(f"Saved audio: {format_clickable_path(saved_path)}")
                        new_col_values.append(saved_path)
                    except Exception as e:
                        print(f"Error writing audio file: {e}")
                        new_col_values.append(None)
                else:
                    new_col_values.append(None)

        updated_df = updated_df.with_columns(
            pl.Series(name=new_col_name, values=new_col_values)
        )

    return updated_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outfile",
        default="youtube-to-docs-artifacts/youtube-docs.csv",
        help=("Local file path to the CSV file to process."),
    )
    parser.add_argument(
        "--tts",
        default=None,
        help=(
            "The TTS model and voice to use. "
            "Format: {model}-{voice} e.g. 'gemini-2.5-flash-preview-tts-Kore'"
        ),
    )

    args = parser.parse_args()
    outfile: str = args.outfile
    tts_arg: Optional[str] = args.tts

    if not tts_arg:
        print("No TTS model specified. Use --tts to specify a model.")
        return

    if not os.path.exists(outfile):
        print(f"Error: Output file {outfile} not found.")
        return

    try:
        df = pl.read_csv(outfile)
    except Exception as e:
        print(f"Error reading CSV {outfile}: {e}")
        return

    from youtube_to_docs.storage import LocalStorage

    print("Using Local storage for standalone TTS run.")
    storage = LocalStorage()
    output_dir = os.path.dirname(outfile)
    storage.ensure_directory(output_dir)
    base_dir = output_dir if output_dir else "."

    updated_df = process_tts(df, tts_arg, storage, base_dir)

    # Save the updated DataFrame
    # If using local storage, outfile is path
    updated_df.write_csv(outfile)
    print(f"Updated {outfile} with new audio columns.")


if __name__ == "__main__":
    main()
