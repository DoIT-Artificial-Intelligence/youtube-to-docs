import argparse
import os
import wave
from typing import List, Optional, Tuple

import polars as pl
from google import genai
from google.genai import types


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Writes PCM data to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def generate_speech(
    text: str, model_name: str, voice_name: str, language_code: Optional[str] = None
) -> bytes:
    """
    Generates speech from text using the specified Gemini model and voice.
    Returns the raw PCM audio bytes.
    """
    try:
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
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].inline_data.data
        else:
            print("Error: No audio data in response.")
            return b""

    except Exception as e:
        print(f"Error generating speech: {e}")
        return b""


def parse_tts_arg(tts_arg: str) -> Tuple[str, str]:
    """
    Parses the --tts argument into (model_name, voice_name).
    Expects format: model-name-voice
    Example: gemini-2.5-flash-preview-tts-Kore -> (gemini-2.5-flash-preview-tts, Kore)
    """
    # Assuming the voice name is the last part after the last hyphen
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
    base_dir: str = ".",
    languages: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Processes the DataFrame to generate TTS for each summary file found.
    """
    model_name, voice_name = parse_tts_arg(tts_arg)
    print(f"Using TTS Model: {model_name}, Voice: {voice_name}")

    # Setup Audio Directory
    audio_dir = os.path.join(base_dir, "audio-files")
    os.makedirs(audio_dir, exist_ok=True)

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
        print(f"Processing column: {col} -> {new_col_name} (Language: {lang_code})")

        new_col_values = []

        for row in df.iter_rows(named=True):
            summary_path = row.get(col)

            if (
                not summary_path
                or not isinstance(summary_path, str)
                or not os.path.exists(summary_path)
            ):
                new_col_values.append(None)
                continue

            summary_filename = os.path.basename(summary_path)
            base_name = os.path.splitext(summary_filename)[0]
            audio_filename = f"{base_name} - {tts_arg}.wav"
            audio_full_path = os.path.abspath(os.path.join(audio_dir, audio_filename))

            if os.path.exists(audio_full_path) and os.path.getsize(audio_full_path) > 0:
                new_col_values.append(audio_full_path)
                continue

            print(f"Generating audio for: {summary_filename}")

            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading summary file {summary_path}: {e}")
                new_col_values.append(None)
                continue

            if not text.strip():
                print("Empty summary text.")
                new_col_values.append(None)
                continue

            pcm_data = generate_speech(text, model_name, voice_name, lang_code)

            if pcm_data:
                try:
                    wave_file(audio_full_path, pcm_data)
                    print(f"Saved audio: {audio_filename}")
                    new_col_values.append(audio_full_path)
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

    output_dir = os.path.dirname(outfile)
    base_dir = output_dir if output_dir else "."

    updated_df = process_tts(df, tts_arg, base_dir)

    # Save the updated DataFrame
    updated_df.write_csv(outfile)
    print(f"Updated {outfile} with new audio columns.")


if __name__ == "__main__":
    main()
