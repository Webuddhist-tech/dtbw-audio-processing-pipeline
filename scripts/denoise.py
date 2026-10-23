"""Remove the background static from a recording, using the recording's own static as the noise sample.

Usage:
    .venv/bin/python scripts/denoise.py AUDIO_FILE --test            # short before/after clips to compare by ear
    .venv/bin/python scripts/denoise.py AUDIO_FILE --strength medium  # clean the whole recording

How it works:
    1. Take the stretches that detect_speakers.py labelled "no speech" (pure static) as the noise sample.
    2. Cut out the low rumble (below LOW_CUT Hz) - it is below the voices.
    3. Learn the noise "fingerprint": how loud the static is at every pitch.
    4. Everywhere in the recording, turn down each pitch that is no louder than the static
       (by the strength's dB), and leave the pitches where a voice is clearly above it.
       The decision is smoothed over time and pitch, so the voices don't sound "watery".

Writes to output/Recordings/<recording>/4 - Clean audio/:
    test/<clip>_0_original.mp3, _1_light.mp3, _2_medium.mp3, _3_strong.mp3    (--test)
    <name>_clean_<strength>.wav                                               (whole recording)
Nothing is sent anywhere.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter
from scipy.signal import butter, sosfiltfilt, stft, istft

from paths import recording_dir, show, speakers_dir

SR = 44100
LOW_CUT = 80            # Hz; the rumble sits at 40-80 Hz
NFFT = 2048
HOP = NFFT // 4
N_STD = 1.5             # a pitch counts as "voice" when it is this many std above the static's level
STRENGTHS = {"light": 8, "medium": 14, "strong": 22}   # dB the static is turned down
BLOCK = 60.0            # process long recordings in blocks of this many seconds
PAD = 2.0
TEST_CLIPS = [("teacher", 2158.5, 27.0), ("chanting", 1500.0, 30.0), ("translator", 2190.75, 21.0)]


def decode(path: Path, start: float = 0.0, duration: float = 0.0) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-af", "pan=mono|c0=c0", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).copy()


def noise_sample(path: Path, segments: list, max_seconds: float = 120.0) -> np.ndarray:
    """Up to max_seconds of the recording's pure static, from the longest "no speech" stretches."""
    quiet = sorted((t for t in segments if t["name"] == "no speech"), key=lambda t: t["start"] - t["end"])
    parts, total = [], 0.0
    for t in quiet:
        s, e = t["start"] + 0.5, t["end"] - 0.5      # stay clear of the edges
        if e - s < 3 or total >= max_seconds:
            continue
        d = min(e - s, max_seconds - total)
        parts.append(decode(path, s, d))
        total += d
    if not parts:
        sys.exit("No 'no speech' stretches to learn the static from.")
    return np.concatenate(parts)


def highpass(x: np.ndarray) -> np.ndarray:
    return sosfiltfilt(butter(4, LOW_CUT, "highpass", fs=SR, output="sos"), x).astype(np.float32)


def fingerprint(noise: np.ndarray) -> np.ndarray:
    """Per-pitch threshold (dB): the static's mean level plus N_STD standard deviations."""
    _, _, z = stft(highpass(noise), SR, nperseg=NFFT, noverlap=NFFT - HOP)
    db = 20 * np.log10(np.abs(z) + 1e-10)
    return (db.mean(axis=1) + N_STD * db.std(axis=1))[:, None]


def clean_block(x: np.ndarray, threshold: np.ndarray, reduce_db: float) -> np.ndarray:
    _, _, z = stft(highpass(x), SR, nperseg=NFFT, noverlap=NFFT - HOP)
    db = 20 * np.log10(np.abs(z) + 1e-10)
    voice = (db > threshold).astype(np.float32)
    voice = uniform_filter(voice, size=(5, 7))          # smooth across ~5 pitches and ~7 frames
    floor = 10 ** (-reduce_db / 20)
    gain = floor + (1 - floor) * voice
    _, y = istft(z * gain, SR, nperseg=NFFT, noverlap=NFFT - HOP)
    return y[: len(x)].astype(np.float32)


def clean(x: np.ndarray, threshold: np.ndarray, reduce_db: float) -> np.ndarray:
    """Clean a long signal block by block (with padding, so block edges don't click)."""
    n, pad, out = int(BLOCK * SR), int(PAD * SR), np.zeros_like(x)
    for a in range(0, len(x), n):
        lo, hi = max(0, a - pad), min(len(x), a + n + pad)
        y = clean_block(x[lo:hi], threshold, reduce_db)
        out[a: min(len(x), a + n)] = y[a - lo: a - lo + min(n, len(x) - a)]
    return out


def write_mp3(x: np.ndarray, mp3: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, np.clip(x, -1, 1), SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp.name,
                        "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--test", action="store_true", help="make short before/after clips at every strength")
    parser.add_argument("--strength", choices=STRENGTHS, default="medium")
    args = parser.parse_args()

    rec = recording_dir(args.audio)
    segments = json.loads((speakers_dir(rec) / "segments.json").read_text())
    out_dir = rec / "4 - Clean audio"
    print("Learning the static from the 'no speech' stretches ...")
    threshold = fingerprint(noise_sample(args.audio, segments))

    if args.test:
        test_dir = out_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        clips = TEST_CLIPS + [("static only", max((t for t in segments if t["name"] == "no speech"),
                                                  key=lambda t: t["end"] - t["start"])["start"] + 1, 15.0)]
        for name, start, dur in clips:
            x = decode(args.audio, start, dur)
            stem = f"{name.replace(' ', '_')}_{int(start // 60)}-{int(start % 60):02d}"
            write_mp3(x, test_dir / f"{stem}_0_original.mp3")
            for k, (level, db) in enumerate(STRENGTHS.items(), 1):
                write_mp3(clean(x, threshold, db), test_dir / f"{stem}_{k}_{level}.mp3")
            print(f"  {stem}: original + {', '.join(STRENGTHS)}")
        print(f"\nListen in {show(test_dir)}/")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cleaning the whole recording ({args.strength}, static turned down {STRENGTHS[args.strength]} dB) ...")
    y = clean(decode(args.audio), threshold, STRENGTHS[args.strength])
    wav = out_dir / f"{args.audio.stem}_clean_{args.strength}.wav"
    sf.write(wav, np.clip(y, -1, 1), SR, subtype="PCM_16")
    print(f"Saved {show(wav)}")


if __name__ == "__main__":
    main()
