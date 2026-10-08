"""Blind spot-check of the speaker labels from detect_speakers.py.

Step 1 - make clips:
    python scripts/spot_check.py make AUDIO_FILE [--count 20]
    Picks one random moment in each of COUNT equal slices of the recording (avoiding the
    places used as --example), and cuts a 10 s clip around it. The clips have no labels;
    the answer key is saved separately so the listener isn't influenced.

Step 2 - listen and fill in output/speakers/<name>/spot_check/answers.txt
    with one letter per clip: T = teacher, L = translator (Ladakhi), C = chanting,
    S = static / no speech, M = mixed (more than one in the clip).

Step 3 - score:
    python scripts/spot_check.py score AUDIO_FILE
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output" / "speakers"
CLIP_LEN = 10.0
EXAMPLE_MARGIN = 30.0  # stay this far (s) away from the examples the script learned from
SEED = 2026
CODES = {"T": "teacher", "L": "translator", "C": "chanting", "S": "no speech", "M": "mixed"}


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def labels_in(turns: list, start: float, end: float) -> dict:
    """Seconds of each label inside [start, end]."""
    share = {}
    for t in turns:
        overlap = min(end, t["end"]) - max(start, t["start"])
        if overlap > 0:
            share[t["name"]] = share.get(t["name"], 0) + overlap
    return share


def make(audio: Path, out_dir: Path, count: int) -> None:
    turns = json.loads((out_dir / "segments.json").read_text())
    examples_file = out_dir / "examples.json"
    examples = json.loads(examples_file.read_text()) if examples_file.exists() else []
    total = turns[-1]["end"]

    def near_example(t: float) -> bool:
        return any(e["start"] - EXAMPLE_MARGIN <= t <= e["end"] + EXAMPLE_MARGIN for e in examples)

    rng = random.Random(SEED)
    slice_len = total / count
    picks = []
    for i in range(count):
        for _ in range(200):
            centre = rng.uniform(i * slice_len + CLIP_LEN / 2, (i + 1) * slice_len - CLIP_LEN / 2)
            if not near_example(centre):
                break
        picks.append(centre)

    check_dir = out_dir / "spot_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    for old in check_dir.glob("check_*.mp3"):
        old.unlink()
    key = []
    for n, centre in enumerate(picks, 1):
        start = centre - CLIP_LEN / 2
        clip = check_dir / f"check_{n:02d}.mp3"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t", str(CLIP_LEN),
             "-i", str(audio), "-af", "pan=mono|c0=c0", "-c:a", "libmp3lame", "-q:a", "4", str(clip)],
            check=True,
        )
        share = labels_in(turns, start, start + CLIP_LEN)
        key.append({"clip": clip.name, "start": round(start, 2), "at": fmt(start),
                    "script_says": max(share, key=share.get), "seconds_per_label": share})

    (out_dir / "spot_check_key.json").write_text(json.dumps(key, indent=2))
    sheet = check_dir / "answers.txt"
    sheet.write_text(
        "# Listen to each clip and write one letter after the colon:\n"
        "# T = teacher, L = translator (Ladakhi), C = chanting, S = static / no speech, M = mixed\n"
        + "".join(f"check_{n:02d}: \n" for n in range(1, count + 1))
    )
    print(f"Made {count} clips of {CLIP_LEN:.0f} s in {check_dir.relative_to(PROJECT_ROOT)}/")
    print(f"Answer sheet: {sheet.relative_to(PROJECT_ROOT)}  (answer key saved separately - don't peek)")


def score(out_dir: Path) -> None:
    key = json.loads((out_dir / "spot_check_key.json").read_text())
    answers = {}
    for line in (out_dir / "spot_check" / "answers.txt").read_text().splitlines():
        if line.startswith("check_") and ":" in line:
            name, _, code = line.partition(":")
            answers[name.strip() + ".mp3"] = code.strip().upper()[:1]

    right = checked = 0
    for k in key:
        code = answers.get(k["clip"], "")
        if code not in CODES:
            print(f"  {k['clip']}  {k['at']}  (no answer)")
            continue
        heard = CODES[code]
        mixed_by_script = len(k["seconds_per_label"]) > 1
        ok = heard == k["script_says"] or (heard == "mixed" and mixed_by_script)
        checked += 1
        right += ok
        detail = ", ".join(f"{name} {sec:.0f}s" for name, sec in k["seconds_per_label"].items())
        print(f"  {'OK   ' if ok else 'WRONG'} {k['clip']}  {k['at']}  you heard: {heard:10}  script: {detail}")
    if checked:
        print(f"\nScore: {right} of {checked} correct ({right / checked:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["make", "score"])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    out_dir = OUTPUT_ROOT / args.audio.stem
    if not (out_dir / "segments.json").exists():
        sys.exit(f"Run detect_speakers.py on {args.audio.name} first.")
    if args.step == "make":
        make(args.audio, out_dir, args.count)
    else:
        score(out_dir)


if __name__ == "__main__":
    main()
