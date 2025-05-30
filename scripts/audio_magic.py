# scripts/audio_magic.py
#
# Reverse an audio file and optionally create a slowed-down copy.
#
# Usage:
#   python scripts/audio_magic.py data/secret.wav

from pathlib import Path
from pydub import AudioSegment
import sys


def change_speed(sound: AudioSegment, factor: float) -> AudioSegment:
    """
    Return a new AudioSegment playing `factor` times slower (<1) or faster (>1).
    """
    if factor <= 0:
        raise ValueError("factor must be > 0")

    altered = sound._spawn(
        sound.raw_data,
        overrides={"frame_rate": int(sound.frame_rate * factor)}
    )
    return altered.set_frame_rate(sound.frame_rate)


def main(src: Path) -> None:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)

    audio = AudioSegment.from_file(src)

    # 1) reverse
    rev = audio.reverse()
    rev.export("secret_rev.wav", format="wav")

    # 2) optional 0.75× speed (== 25 % slower)
    slow = change_speed(rev, 0.75)
    slow.export("secret_rev_slow.wav", format="wav")

    print("Saved secret_rev.wav  and  secret_rev_slow.wav")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audio_magic.py <audio-file>")
        sys.exit(1)
    main(sys.argv[1])
