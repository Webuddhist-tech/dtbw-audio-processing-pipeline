"""Remove the background static from a recording, using the recording's own static as the noise sample.

Usage:
    .venv/bin/python scripts/denoise.py AUDIO_FILE --test            # short before/after clips to compare by ear
    .venv/bin/python scripts/denoise.py AUDIO_FILE --strength medium  # clean the whole recording

How it works:
    1. Take the stretches that detect_speakers.py labelled "no speech" (pure static) as the noise sample.
    2. Cut out the low rumble (below LOW_CUT Hz) - it is below the voices.
    3. Learn the noise "fingerprint": how loud the static is at every pitch.
    4. Everywhere in the recording, turn down the pitches where there is only static:
       --method smooth (default): gradually, by how far each pitch is above the static
       --method gate: on/off - each pitch no louder than the static is turned down by the
                      strength's dB (smoothed over time and pitch)

Writes to output/Recordings/<recording>/4 - Clean audio/:
    test/<clip>_0_original.mp3, _1_light[_smooth].mp3, _2_medium…, _3_strong…    (--test)
    <name>_clean_<method>_<strength>.wav                                       (whole recording)
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
SMOOTH = 0.98           # smooth method: how much each frame's estimate leans on the previous one
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


def fingerprint(noise: np.ndarray) -> dict:
    """The static per pitch: a level threshold (dB, gate method) and its average power (smooth method)."""
    _, _, z = stft(highpass(noise), SR, nperseg=NFFT, noverlap=NFFT - HOP)
    db = 20 * np.log10(np.abs(z) + 1e-10)
    return {"threshold": (db.mean(axis=1) + N_STD * db.std(axis=1))[:, None],
            "power": (np.abs(z) ** 2).mean(axis=1)}


def clean_block(x: np.ndarray, noise: dict, reduce_db: float, method: str = "gate") -> np.ndarray:
    _, _, z = stft(highpass(x), SR, nperseg=NFFT, noverlap=NFFT - HOP)
    if method == "smooth":
        return smooth_block(z, noise["power"], reduce_db, len(x))
    threshold = noise["threshold"]
    db = 20 * np.log10(np.abs(z) + 1e-10)
    voice = (db > threshold).astype(np.float32)
    voice = uniform_filter(voice, size=(5, 7))          # smooth across ~5 pitches and ~7 frames
    floor = 10 ** (-reduce_db / 20)
    gain = floor + (1 - floor) * voice
    _, y = istft(z * gain, SR, nperseg=NFFT, noverlap=NFFT - HOP)
    return y[: len(x)].astype(np.float32)


def smooth_block(z: np.ndarray, noise_power: np.ndarray, reduce_db: float, length: int) -> np.ndarray:
    """Turn each pitch down gradually by how far it is above the static (a Wiener-style filter with
    "decision-directed" smoothing), instead of an on/off gate - fewer watery or muffled side effects."""
    power = np.abs(z) ** 2
    floor = 10 ** (-reduce_db / 20)
    gain = np.empty(power.shape, np.float32)
    prev = np.zeros(power.shape[0])
    for t in range(power.shape[1]):
        post = power[:, t] / noise_power
        prior = SMOOTH * prev + (1 - SMOOTH) * np.maximum(post - 1, 0)
        g = np.maximum(prior / (1 + prior), floor)
        gain[:, t] = g
        prev = g ** 2 * post
    _, y = istft(z * gain, SR, nperseg=NFFT, noverlap=NFFT - HOP)
    return y[:length].astype(np.float32)


def clean(x: np.ndarray, noise: dict, reduce_db: float, method: str = "gate") -> np.ndarray:
    """Clean a long signal block by block (with padding, so block edges don't click)."""
    n, pad, out = int(BLOCK * SR), int(PAD * SR), np.zeros_like(x)
    for a in range(0, len(x), n):
        lo, hi = max(0, a - pad), min(len(x), a + n + pad)
        y = clean_block(x[lo:hi], noise, reduce_db, method)
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
    parser.add_argument("--method", choices=["gate", "smooth"], default="smooth",
                        help="gate = on/off per pitch; smooth = gradual (keeps more of the voice)")
    args = parser.parse_args()

    rec = recording_dir(args.audio)
    segments = json.loads((speakers_dir(rec) / "segments.json").read_text())
    out_dir = rec / "4 - Clean audio"
    print("Learning the static from the 'no speech' stretches ...")
    noise = fingerprint(noise_sample(args.audio, segments))

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
                tag = "" if args.method == "gate" else f"_{args.method}"
                write_mp3(clean(x, noise, db, args.method), test_dir / f"{stem}_{k}_{level}{tag}.mp3")
            print(f"  {stem}: original + {', '.join(STRENGTHS)} ({args.method})")
        print(f"\nListen in {show(test_dir)}/")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cleaning the whole recording ({args.strength}, static turned down {STRENGTHS[args.strength]} dB) ...")
    y = clean(decode(args.audio), noise, STRENGTHS[args.strength], args.method)
    wav = out_dir / f"{args.audio.stem}_clean_{args.method}_{args.strength}.wav"
    sf.write(wav, np.clip(y, -1, 1), SR, subtype="PCM_16")
    print(f"Saved {show(wav)}")


if __name__ == "__main__":
    main()
