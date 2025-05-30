"""
Detect the spoken language in an audio file
and transcribe it in one call to Whisper (whisper-1).

Usage:
    python audio_detect_and_transcribe.py path/to/audio.[wav|mp3|m4a...]

If no path is given, the script defaults to "./secret_rev_slow.wav".
"""

from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
import json
import sys
import openai
import os

load_dotenv()

# ---------------------------------------------------------------------------

def load_openai_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment")
    return openai.OpenAI(api_key=api_key)

# ---------------------------------------------------------------------------

def detect_and_transcribe(
    wav_path: Path,
    model: str = "whisper-1",
) -> tuple[str, str]:
    """
    Returns (language_code, transcript_text)
    """
    client = load_openai_client()

    with wav_path.open("rb") as audio_file:
        resp = client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format="verbose_json",   # includes detected language
        )

    # 'resp' is an OpenAI object; convert to dict for convenience
    data = json.loads(resp.model_dump_json())

    language = data.get("language", "unknown")
    transcript = data.get("text", "").strip()

    return language, transcript

# ---------------------------------------------------------------------------

def main() -> None:
    wav_arg = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("secret_rev.wav")
    if not wav_arg.exists():
        sys.exit(f"File not found: {wav_arg}")

    print(f"Processing: {wav_arg}")

    lang, text = detect_and_transcribe(wav_arg)

    print("\nDetected language :", lang)
    print("\n--- TRANSCRIPTION ---")
    print(text)
    print("----------------------")

    # save alongside the audio file
    out_txt = wav_arg.with_suffix(".txt")
    out_txt.write_text(text, encoding="utf-8")
    print(f"\nTranscription saved to: {out_txt.resolve()}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
