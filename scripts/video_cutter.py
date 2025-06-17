"""video_cutter.py: Cut a segment from an MP4 video using ffmpeg, with millisecond precision."""

import argparse
import subprocess
import shutil
import sys
import os
import re


def parse_timestamp(ts: str) -> float:
    """Parse a timestamp string into seconds (float).
    Accepts formats: HH:MM:SS(.ms), MM:SS(.ms), SS(.ms), with dot or comma separator."""
    pattern = r'^(\d+:)?(\d{1,2}:)?\d{1,2}([\.,]\d{1,3})?$'
    if not re.match(pattern, ts):
        raise argparse.ArgumentTypeError(
            f"Invalid timestamp format: '{ts}'. Use HH:MM:SS(.ms), MM:SS(.ms), or SS(.ms).")
    # Normalize comma to dot
    ts = ts.replace(',', '.')
    parts = ts.split(':')
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0.0; m, s = parts
    else:
        h = 0.0; m = 0.0; s = parts[0]
    return h * 3600 + m * 60 + s


def format_timestamp(seconds: float) -> str:
    """Format seconds (float) into HH:MM:SS.mmm with millisecond precision."""
    if seconds < 0:
        raise ValueError("Negative timestamp.")
    total_sec = int(seconds)
    ms = int(round((seconds - total_sec) * 1000))
    h = total_sec // 3600
    m = (total_sec // 60) % 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def check_ffmpeg():
    if shutil.which('ffmpeg') is None:
        print("Error: ffmpeg is not installed or not found in PATH.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Cut a segment from an MP4 video using ffmpeg with millisecond precision.")
    parser.add_argument('input', help="Path to input MP4 file")
    parser.add_argument('start', type=parse_timestamp,
                        help="Start timestamp (e.g. 00:01:23.456 or 83.500)")
    parser.add_argument('end', type=parse_timestamp,
                        help="End timestamp (e.g. 00:02:34.789 or 154.250)")
    parser.add_argument('-o', '--output',
                        help="Path to output MP4 file. Defaults to input_start_end.mp4")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    start_sec = args.start
    end_sec = args.end
    if end_sec <= start_sec:
        print("Error: End time must be greater than start time.", file=sys.stderr)
        sys.exit(1)
    duration_sec = end_sec - start_sec

    start_ts = format_timestamp(start_sec)
    duration_ts = format_timestamp(duration_sec)

    base, _ = os.path.splitext(os.path.basename(input_path))
    end_ts = format_timestamp(end_sec)
    default_output = f"{base}_{start_ts.replace(':','-')}_{end_ts.replace(':','-')}.mp4"
    output_path = os.path.abspath(args.output or default_output)

    check_ffmpeg()

    cmd = [
        'ffmpeg',
        '-y',
        '-ss', start_ts,
        '-i', input_path,
        '-t', duration_ts,
        '-c', 'copy',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Created output file: {output_path}")
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed with error:", e, file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == '__main__':
    main()
