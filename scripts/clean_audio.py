"""Make cleaned versions of an audio file so they can be compared by ear.

Usage:
    python scripts/clean_audio.py [AUDIO_FILE]

Defaults to the 30-second clip used in the first Monlam test.
Writes one file per cleaning level to output/cleaning/<audio name>/:

    0_original.mp3  untouched copy, for comparison
    1_light.wav     remove low rumble + even out the volume
    2_medium.wav    light + gentle background-noise reduction
    3_strong.wav    light + strong noise reduction + cut very high frequencies

Cleaned files are mono 16 kHz WAV, the format speech models work in.
Nothing is sent to Monlam.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "output" / "test" / "260813_0631_300s_30s.mp3"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "cleaning"

# highpass: drops rumble/hum below 80 Hz (fans, traffic, mic handling); speech is above this.
# afftdn:   FFT noise reduction; nr = how many dB of noise to remove, tn=1 keeps tracking the noise level.
# lowpass:  drops hiss above 7 kHz; speech intelligibility sits well below that.
# loudnorm: evens out the volume to a standard speech loudness (-16 LUFS).
HIGHPASS = "highpass=f=80"
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
LEVELS = {
    "1_light": [HIGHPASS, LOUDNORM],
    "2_medium": [HIGHPASS, "afftdn=nr=12:nf=-35:tn=1", LOUDNORM],
    "3_strong": [HIGHPASS, "lowpass=f=7000", "afftdn=nr=24:nf=-30:tn=1", LOUDNORM],
}


def clean(source: Path, filters: list, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-af", ",".join(filters), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
        check=True,
    )


def volume_stats(path: Path) -> str:
    """Mean and peak volume in dB, as measured by ffmpeg's volumedetect."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    mean = re.search(r"mean_volume: (-?[\d.]+) dB", result.stderr)
    peak = re.search(r"max_volume: (-?[\d.]+) dB", result.stderr)
    return f"average {mean.group(1) if mean else '?'} dB, peak {peak.group(1) if peak else '?'} dB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="?", default=DEFAULT_INPUT)
    args = parser.parse_args()

    if not args.audio.exists():
        sys.exit(f"Audio file not found: {args.audio}")

    out_dir = OUTPUT_ROOT / args.audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    original = out_dir / f"0_original{args.audio.suffix}"
    shutil.copyfile(args.audio, original)
    print(f"Input: {args.audio}")
    print(f"  {original.name:16} {volume_stats(original)}")

    for name, filters in LEVELS.items():
        dest = out_dir / f"{name}.wav"
        clean(args.audio, filters, dest)
        print(f"  {dest.name:16} {volume_stats(dest)}")

    print(f"\nSaved to {out_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
