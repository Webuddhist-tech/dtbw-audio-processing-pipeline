"""Cut separate Tibetan (teacher) and Ladakhi (translator) audio for each course unit.

Usage:
    .venv/bin/python scripts/cut_unit_audio.py AUDIO_FILE [--units 1.2.2 1.3.2] [--chanting tibetan|both|none]

Reads 3 - Course draft/curriculum.json (unit time ranges) and 1 - Speakers/segments.json
(who speaks when) in the recording's folder (see paths.py). For every unit:
    - teacher turns            -> Tibetan file   (unit_<n>_bo.mp3)
    - translator turns         -> Ladakhi file   (unit_<n>_lad.mp3)
    - chanting                 -> where --chanting says (default: Tibetan)
    - silence / static         -> left out
Each switch between speakers is moved to the nearest real pause (within SNAP s), so no word is cut
and no word of the other language leaks in. Pieces are joined with a short pause and tiny fades.
Files go to 3 - Course draft/unit audio/. A joins list (unit_<n>_joins.txt) tells the reviewer where each join is, to check by ear, and
tracks.json records each unit's files and lengths (used by the course outline).
A unit whose curriculum entry has "tracks": ["bo"] gets no Ladakhi file.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import course_draft_dir, recording_dir, speakers_dir  # noqa: E402
from transcribe import find_pauses  # noqa: E402  (same pause detection as transcription)

OUT_SR = 44100
ANALYSIS_SR = 16000
SNAP = 2.0         # move a speaker switch to the longest pause within this many seconds
GAP = 0.6          # silence inserted between joined pieces (s)
FADE = 0.02        # fade in/out at each piece edge, avoids clicks (s)
MIN_PIECE = 1.5    # ignore pieces shorter than this (s) - usually misdetections

TRACKS = {"bo": "Tibetan (teacher)", "lad": "Ladakhi (translator)"}


def decode(path: Path, start: float, end: float, sr: int, dtype: str) -> np.ndarray:
    fmt = "s16le" if dtype == "int16" else "f32le"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
           "-i", str(path), "-af", "pan=mono|c0=c0", "-ar", str(sr), "-f", fmt, "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype)


def fmt(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def track_of(label: str, chanting: str) -> list:
    if label == "teacher":
        return ["bo"]
    if label == "translator":
        return ["lad"]
    if label == "chanting":
        return {"tibetan": ["bo"], "both": ["bo", "lad"], "none": []}[chanting]
    return []  # no speech


def unit_pieces(unit: dict, segments: list, pauses: np.ndarray, offset: float, chanting: str) -> list:
    """(track, start, end) pieces for one unit, with speaker switches snapped to pauses."""
    a, b = unit["audio"]["start"], unit["audio"]["end"]
    # Unit times are rounded to whole seconds, so ignore slivers of a neighbouring unit's turn.
    turns = [dict(t, start=max(t["start"], a), end=min(t["end"], b)) for t in segments
             if min(t["end"], b) - max(t["start"], a) >= 1.0 and t["name"] != "no speech"]

    def snap(t: float) -> float:
        near = pauses[(pauses.mean(axis=1) + offset >= t - SNAP) & (pauses.mean(axis=1) + offset <= t + SNAP)]
        if not len(near):
            return t
        longest = near[np.argmax(near[:, 1] - near[:, 0])]
        return float(longest.mean() + offset)

    pieces = []
    for t in turns:
        for track in track_of(t["name"], chanting):
            pieces.append([track, snap(t["start"]), snap(t["end"]), t["name"]])
    # join touching pieces of the same track (e.g. teacher then chanting, both Tibetan)
    merged = {}
    for track in TRACKS:
        mine = sorted([p for p in pieces if p[0] == track], key=lambda p: p[1])
        out = []
        for p in mine:
            if out and p[1] - out[-1][2] < 0.5:
                out[-1][2] = max(out[-1][2], p[2])
                out[-1][3] += "+" + p[3]
            else:
                out.append(p)
        merged[track] = [p for p in out if p[2] - p[1] >= MIN_PIECE]
    return merged


def join(audio: np.ndarray, offset: float, pieces: list) -> np.ndarray:
    fade = int(FADE * OUT_SR)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    gap = np.zeros(int(GAP * OUT_SR), np.float32)
    parts = []
    for _, s, e, _ in pieces:
        clip = audio[int((s - offset) * OUT_SR): int((e - offset) * OUT_SR)].copy()
        if len(clip) > 2 * fade:
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        if parts:
            parts.append(gap)
        parts.append(clip)
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--units", nargs="*", help="unit numbers (default: all)")
    parser.add_argument("--chanting", choices=["tibetan", "both", "none"], default="tibetan")
    args = parser.parse_args()

    rec = recording_dir(args.audio)
    course = json.loads((course_draft_dir(rec) / "curriculum.json").read_text())
    segments = json.loads((speakers_dir(rec) / "segments.json").read_text())
    out_dir = course_draft_dir(rec) / "unit audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    units = [u for s in course["sections"] for ss in s["subsections"] for u in ss["units"]]
    if args.units:
        units = [u for u in units if u["number"] in args.units]
    summary_file = out_dir / "tracks.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    for u in units:
        a, b = u["audio"]["start"], u["audio"]["end"]
        offset = max(0.0, a - SNAP - 1)
        end = b + SNAP + 1
        pauses = find_pauses(decode(args.audio, offset, end, ANALYSIS_SR, "int16"))
        audio = decode(args.audio, offset, end, OUT_SR, "float32")
        tracks = unit_pieces(u, segments, pauses, offset, args.chanting)

        lines = [f"Unit {u['number']}  {u['title_bo']}  ({u['title_en']})",
                 f"Original recording {fmt(a)} - {fmt(b)}; chanting goes to: {args.chanting}", ""]
        summary[u["number"]] = {}
        for track, label in TRACKS.items():
            if track not in u["audio"].get("tracks", TRACKS):
                lines.append(f"{label}: none for this unit")
                continue
            pieces = tracks[track]
            mp3 = out_dir / f"unit_{u['number']}_{track}.mp3"
            joined = join(audio, offset, pieces)
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, joined, OUT_SR, subtype="PCM_16")
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp.name,
                                "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)], check=True)
            lines.append(f"{label}: {mp3.name}  {len(joined) / OUT_SR / 60:.1f} min, {len(pieces)} piece(s)")
            summary[u["number"]][track] = {"file": mp3.name, "minutes": round(len(joined) / OUT_SR / 60, 1), "pieces": len(pieces)}
            pos = 0.0
            for n, (_, s, e, what) in enumerate(pieces, 1):
                lines.append(f"  piece {n}: at {fmt(pos)} in the file  =  original {fmt(s)} - {fmt(e)}  ({e - s:.0f}s, {what})")
                pos += (e - s) + GAP
            lines.append("")
        (out_dir / f"unit_{u['number']}_joins.txt").write_text("\n".join(lines))
        summary_file.write_text(json.dumps(summary, indent=2))
        print("\n".join(lines))


if __name__ == "__main__":
    main()
