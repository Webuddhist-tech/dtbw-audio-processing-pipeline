"""Find who is speaking when (teacher / translator / chanting ...) in a recording.

Usage:
    .venv/bin/python scripts/detect_speakers.py AUDIO_FILE \\
        --example teacher=58:46 --example teacher=40:58 \\
        --example translator=30:38 \\
        --example chanting=1:07:17 --example chanting=24:05

    Each --example is LABEL=TIME (h:mm:ss, mm:ss or seconds), optionally +SECONDS
    for its length (default 20 s). Give at least one example per label; more is better.
    Without any --example the voices are sorted blindly into --groups groups.

How it works:
    1. Cut the audio into short overlapping windows (1.5 s) and skip pauses.
    2. Turn every window into a "voice fingerprint" (SpeechBrain ECAPA speaker embedding).
       Fingerprints are saved, so re-running with other examples takes seconds.
    3. Compare every window with the examples and give it the closest label,
       then let each label's "average voice" adapt a little to the whole recording.
    4. Smooth out flickers: turns last minutes, so a 1-second switch is a mistake.
       Short pauses stay with the speaker; long ones (static, breaks) are labelled "no speech".

Writes to output/speakers/<audio name>/:
    labels_audacity.txt   open the audio in Audacity, then File > Import > Labels
    segments.json         the same timeline, for the next pipeline step
    samples/              a few 20 s clips of each label, to check by ear
Times are always relative to the start of the original file. Nothing is sent to Monlam.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from speechbrain.inference.speaker import EncoderClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output" / "speakers"
MODEL_DIR = PROJECT_ROOT / ".cache" / "spkrec-ecapa-voxceleb"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SR = 16000
WIN = 1.5            # seconds per fingerprint window
HOP = 0.75           # step between windows
SMOOTH = 6           # majority vote over +/- this many windows (~9 s)
MIN_TURN = 4.0       # turns shorter than this (s) are merged into a neighbour
LONG_GAP = 5.0       # stretches without sound this long (s) become their own "no speech" turn
NO_SPEECH = "no speech"
VOICE_DB = 5         # a 50 ms frame counts as sound when this many dB above the noise floor
                     # (8 dropped the teacher's quiet passages; 0 lets pauses count as speech)
EXAMPLE_LEN = 20.0   # default length of an --example
ANCHOR = 0.5         # how strongly labels stay tied to the examples while adapting
REFINE_ROUNDS = 5
SAMPLE_LEN = 20.0
SAMPLES_PER_LABEL = 3


def parse_time(text: str) -> float:
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def parse_example(text: str) -> tuple:
    label, _, when = text.partition("=")
    if not label or not when:
        raise argparse.ArgumentTypeError(f"expected LABEL=TIME, got {text!r}")
    when, _, length = when.partition("+")
    return label.strip(), parse_time(when), float(length) if length else EXAMPLE_LEN


def load_audio(path: Path) -> np.ndarray:
    """Decode the left channel (the right one is silent in these recordings) to 16 kHz mono."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-af", "pan=mono|c0=c0", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)


def speech_mask(audio: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """True for windows where at least 40% of 50 ms frames are clearly above the noise floor."""
    frame = int(0.05 * SR)
    frames = audio[: len(audio) // frame * frame].reshape(-1, frame)
    frame_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    voiced = frame_db > np.percentile(frame_db, 10) + VOICE_DB
    per_win = int(WIN / 0.05)
    return np.array([voiced[f:f + per_win].mean() >= 0.4 for f in (starts // frame)])


def fingerprints(audio_path: Path, cache: Path) -> tuple:
    """Return (window start times in s, speech mask, normalised embeddings); embeddings are cached on disk."""
    audio = load_audio(audio_path)
    print(f"  {len(audio) / SR / 60:.1f} minutes")
    win, hop = int(WIN * SR), int(HOP * SR)
    starts = np.arange(0, len(audio) - win + 1, hop)
    speech = speech_mask(audio, starts)

    if cache.exists():
        data = np.load(cache)
        if float(data["win"]) == WIN and float(data["hop"]) == HOP and len(data["emb"]) == len(starts):
            print(f"  using saved fingerprints ({cache.name})")
            return starts / SR, speech, data["emb"]

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODEL_DIR), run_opts={"device": DEVICE}
    )
    out = []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            batch = np.stack([audio[s:s + win] for s in starts[i:i + 64]])
            out.append(model.encode_batch(torch.from_numpy(batch)).squeeze(1).cpu().numpy())
            print(f"\r  fingerprints: {min(i + 64, len(starts))}/{len(starts)}", end="", flush=True)
    print()
    emb = np.concatenate(out)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    times = starts / SR
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb, win=WIN, hop=HOP)
    return times, speech, emb


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def classify_with_examples(times, speech, emb, examples) -> tuple:
    """Label every speech window with the closest example label, adapting labels to the recording."""
    names = sorted({label for label, _, _ in examples})
    anchors = []
    for name in names:
        picked = np.zeros(len(times), bool)
        for label, start, length in examples:
            if label == name:
                picked |= (times >= start) & (times + WIN <= start + length)
        picked &= speech
        if not picked.any():
            sys.exit(f"Example(s) for {name!r} contain no speech windows; pick another time.")
        anchors.append(unit(emb[picked].mean(axis=0)))
    anchors = np.array(anchors)

    centroids = anchors.copy()
    speech_emb = emb[speech]
    for _ in range(REFINE_ROUNDS):
        assigned = np.argmax(speech_emb @ centroids.T, axis=1)
        for k in range(len(names)):
            mine = speech_emb[assigned == k]
            if len(mine):
                centroids[k] = unit(ANCHOR * anchors[k] + (1 - ANCHOR) * unit(mine.mean(axis=0)))
    return names, centroids


def classify_blind(speech, emb, groups) -> tuple:
    km = KMeans(n_clusters=groups, n_init=10, random_state=0).fit(emb[speech])
    order = np.argsort(-np.bincount(km.labels_, minlength=groups))
    names = [f"Group {chr(65 + i)}" for i in range(groups)]
    return names, unit(km.cluster_centers_[order])


def smooth(labels: np.ndarray) -> np.ndarray:
    """Majority vote over neighbouring windows; -1 (pause) does not vote."""
    out = labels.copy()
    for i in np.flatnonzero(labels >= 0):
        near = labels[max(0, i - SMOOTH): i + SMOOTH + 1]
        out[i] = np.bincount(near[near >= 0]).argmax()
    return out


def to_turns(labels: np.ndarray, margins: np.ndarray, times: np.ndarray, gap_label: int) -> list:
    """Collapse per-window labels into turns: long pauses get gap_label, short pauses
    and very short turns are absorbed by a neighbouring speaker."""
    filled = labels.copy()
    i = 0
    while i < len(filled):
        if filled[i] < 0:
            j = i
            while j < len(filled) and filled[j] < 0:
                j += 1
            if (j - i - 1) * HOP + WIN >= LONG_GAP:
                filled[i:j] = gap_label
            i = j
        else:
            i += 1
    for i in range(1, len(filled)):
        if filled[i] < 0:
            filled[i] = filled[i - 1]
    first_real = filled[filled >= 0][0] if (filled >= 0).any() else 0
    filled[filled < 0] = first_real

    turns = []
    for i, lab in enumerate(filled):
        if turns and turns[-1]["label"] == lab:
            turns[-1]["last"] = i
        else:
            turns.append({"label": int(lab), "first": i, "last": i})

    def length(t):
        return (t["last"] - t["first"]) * HOP + WIN

    # Merge turns shorter than MIN_TURN into the longer neighbour, shortest first.
    while len(turns) > 1:
        idx = min(range(len(turns)), key=lambda k: length(turns[k]))
        if length(turns[idx]) >= MIN_TURN:
            break
        neighbours = [t for t in (turns[idx - 1] if idx else None, turns[idx + 1] if idx + 1 < len(turns) else None) if t]
        speakers = [t for t in neighbours if t["label"] != gap_label]
        target = max(speakers or neighbours, key=length)
        target["first"] = min(target["first"], turns[idx]["first"])
        target["last"] = max(target["last"], turns[idx]["last"])
        del turns[idx]
        merged = [turns[0]]
        for t in turns[1:]:
            if t["label"] == merged[-1]["label"]:
                merged[-1]["last"] = t["last"]
            else:
                merged.append(t)
        turns = merged

    result = []
    for t in turns:
        start = result[-1]["end"] if result else float(times[t["first"]])
        end = float(times[t["last"]]) + WIN
        conf = float(np.mean(margins[t["first"]: t["last"] + 1]))
        result.append({"label": t["label"], "start": round(start, 2), "end": round(end, 2), "confidence": conf})
    return result


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def cut_sample(src: Path, start: float, length: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-t", str(length),
         "-i", str(src), "-af", "pan=mono|c0=c0", "-c:a", "libmp3lame", "-q:a", "4", str(dest)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--example", type=parse_example, action="append", default=[],
                        help="LABEL=TIME[+SECONDS], e.g. teacher=58:46 (repeatable)")
    parser.add_argument("--groups", type=int, default=2, help="number of groups when no examples are given")
    args = parser.parse_args()
    if not args.audio.exists():
        sys.exit(f"Audio file not found: {args.audio}")

    out_dir = OUTPUT_ROOT / args.audio.stem
    print(f"Fingerprinting {args.audio.name} on {DEVICE} ...")
    times, speech, emb = fingerprints(args.audio, out_dir / "fingerprints.npz")
    print(f"  {len(times)} windows, {speech.mean():.0%} contain sound")

    if args.example:
        print(f"Learning from {len(args.example)} examples ...")
        names, centroids = classify_with_examples(times, speech, emb, args.example)
    else:
        print(f"No examples: sorting blindly into {args.groups} groups ...")
        names, centroids = classify_blind(speech, emb, args.groups)

    sims = emb @ centroids.T
    raw = np.where(speech, np.argmax(sims, axis=1), -1)
    top2 = np.sort(sims, axis=1)[:, -2:]
    margins = np.where(speech, top2[:, 1] - top2[:, 0], 0.0)

    if args.example:
        print("  check - how the script labels your own examples (before smoothing):")
        for label, start, length in args.example:
            inside = speech & (times >= start) & (times + WIN <= start + length)
            share = np.mean(raw[inside] == names.index(label)) if inside.any() else 0
            print(f"    {label:12} at {fmt(start)}: {share:.0%} of its windows labelled {label}")

    names = names + [NO_SPEECH]
    turns = to_turns(smooth(raw), margins, times, gap_label=len(names) - 1)
    voiced = [t["confidence"] for t in turns if names[t["label"]] != NO_SPEECH]
    unsure_cut = np.percentile(voiced, 15) if len(voiced) > 5 else -1
    for t in turns:
        t["name"] = names[t["label"]]
        t["unsure"] = bool(t["name"] != NO_SPEECH and t["confidence"] < unsure_cut)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for old in samples_dir.glob("*.mp3"):
        old.unlink()
    with open(out_dir / "labels_audacity.txt", "w") as f:
        for t in turns:
            f.write(f"{t['start']:.2f}\t{t['end']:.2f}\t{t['name']}{' (unsure)' if t['unsure'] else ''}\n")
    (out_dir / "segments.json").write_text(json.dumps(turns, indent=2))
    examples = [{"label": label, "start": start, "end": start + length} for label, start, length in args.example]
    (out_dir / "examples.json").write_text(json.dumps(examples, indent=2))

    print("\nSummary")
    for i, name in enumerate(names):
        mine = [t for t in turns if t["label"] == i]
        lengths = [t["end"] - t["start"] for t in mine]
        print(f"  {name:12} {sum(lengths) / 60:5.1f} min in {len(mine):3d} turns "
              f"(typical {np.median(lengths) if lengths else 0:4.0f}s, longest {max(lengths, default=0):4.0f}s)")
        for n, t in enumerate(sorted(mine, key=lambda t: t["start"] - t["end"])[:SAMPLES_PER_LABEL], 1):
            s = max(t["start"], (t["start"] + t["end"]) / 2 - SAMPLE_LEN / 2)
            dest = samples_dir / f"{name.replace(' ', '_')}_{n}_at_{fmt(s).replace(':', '-')}.mp3"
            cut_sample(args.audio, s, min(SAMPLE_LEN, t["end"] - s), dest)
    print(f"  turns marked unsure: {sum(t['unsure'] for t in turns)} of {len(turns)}")

    print("\nTimeline")
    for t in turns:
        print(f"  {fmt(t['start'])} - {fmt(t['end'])}  {t['name']}{'  (unsure)' if t['unsure'] else ''}")
    print(f"\nSaved to {out_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
