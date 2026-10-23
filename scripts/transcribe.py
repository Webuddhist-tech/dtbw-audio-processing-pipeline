"""Transcribe the teacher's parts of a recording with Monlam AI, with real timestamps.

Usage:
    .venv/bin/python scripts/transcribe.py AUDIO_FILE [--labels teacher] [--limit-minutes 10] [--dry-run] [--yes]

Needs the speaker timeline from detect_speakers.py (1 - Speakers/segments.json).

Steps:
    1. Keep only the segments with the chosen labels (default: teacher).
    2. Widen each segment to the nearest pause on both sides (up to MAX_PAD s), so the
       first/last words survive a speaker switch that is a few seconds off.
    3. Split long segments into 10-30 s pieces, cutting only in pauses. With no pause,
       a piece may grow to MAX_PIECE s; only then is it cut at the quietest moment (flagged).
    4. Show pieces, minutes and estimated cost; ask before sending (skip with --yes).
    5. Send pieces as Monlam jobs, PARALLEL at a time. Every reply is saved at once, so a
       re-run skips finished pieces and nothing is paid twice.
    6. Check each reply for warning signs. A garbled piece (far too much text, or a phrase
       stuck repeating - a known Monlam glitch) is cut in half at a pause and both halves
       resent, up to RESPLIT_ROUNDS times.
    7. Shift Monlam's in-piece times to times in the original file and write the transcript.

Writes to output/Recordings/<recording>/2 - Transcript/ (see paths.py):
    transcript_raw.txt     readable transcript, with markers for skipped parts and warnings
    transcript_raw.json    the same, for the correction step
    working files/
        progress.log       the live progress log (also printed)
        plan.json          the pieces and where they were cut
        pieces/ results/   the audio pieces sent and Monlam's raw replies
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from paths import recording_dir, show, speakers_dir, transcript_dir, working_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SR = 16000
FRAME = 0.02            # seconds per loudness frame
VOICE_DB = 5            # same "is there sound" rule as detect_speakers.py
MIN_PAUSE = 0.3         # a pause must be at least this quiet stretch (s)
MAX_PAD = 3.0           # widen segments to the nearest pause, at most this far (s)
MIN_PIECE, TARGET_PIECE, MAX_PIECE = 10.0, 30.0, 45.0
PARALLEL = 3
POLL_SECONDS = 3
JOB_TIMEOUT = 10 * 60
RETRIES = 3
RESPLIT_ROUNDS = 2       # how many times a garbled piece may be halved and resent
MIN_HALF = 4.0           # never make halves shorter than this (s)
COST_PER_MINUTE = 0.032  # from the first test: $0.016 per 30 s

# Warning thresholds, in Tibetan syllables per second of audio (normal speech ~3-6).
TOO_FEW_SYL_PER_S = 1.0
TOO_MANY_SYL_PER_S = 9.0

_log_lock = threading.Lock()
_log_file = None


def log(message: str) -> None:
    line = f"[{datetime.datetime.now():%H:%M:%S}] {message}"
    with _log_lock:
        print(line, flush=True)
        if _log_file:
            _log_file.write(line + "\n")
            _log_file.flush()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ---------- audio and pauses ----------

def load_audio(path: Path) -> np.ndarray:
    """Left channel (the right one is silent in these recordings), 16 kHz, int16."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-af", "pan=mono|c0=c0", "-ar", str(SR), "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.int16)


def find_pauses(audio: np.ndarray) -> np.ndarray:
    """Return an array of (start, end) seconds for every quiet stretch of at least MIN_PAUSE."""
    n = int(FRAME * SR)
    frames = audio[: len(audio) // n * n].reshape(-1, n).astype(np.float32) / 32768
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    quiet = db <= np.percentile(db, 10) + VOICE_DB
    edges = np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = (ends - starts) * FRAME >= MIN_PAUSE
    return np.stack([starts[keep] * FRAME, ends[keep] * FRAME], axis=1)


def pauses_between(pauses: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Pauses whose middle lies in [lo, hi]."""
    mids = pauses.mean(axis=1)
    return pauses[(mids >= lo) & (mids <= hi)]


def widen(start: float, end: float, pauses: np.ndarray, total: float) -> tuple:
    """Move start back / end forward to the nearest pause within MAX_PAD seconds."""
    before = pauses_between(pauses, start - MAX_PAD, start)
    new_start = before[-1].mean() if len(before) else start - MAX_PAD
    after = pauses_between(pauses, end, end + MAX_PAD)
    new_end = after[0].mean() if len(after) else end + MAX_PAD
    return max(0.0, new_start), min(total, new_end)


def split(start: float, end: float, pauses: np.ndarray, audio: np.ndarray) -> list:
    """Split [start, end] into pieces cut in pauses. Returns (start, end, forced_cut) tuples."""
    pieces, cur = [], start
    while end - cur > TARGET_PIECE:
        cut, forced = None, False
        for lo, hi in ((MIN_PIECE, TARGET_PIECE), (TARGET_PIECE, MAX_PIECE)):
            options = pauses_between(pauses, cur + lo, min(cur + hi, end - 2))
            if len(options):
                longest = options[np.argmax(options[:, 1] - options[:, 0])]
                cut = float(longest.mean())
                break
        if cut is None:
            if end - cur <= MAX_PIECE:
                break
            # No pause at all: cut at the quietest 20 ms between TARGET_PIECE and MAX_PIECE.
            n = int(FRAME * SR)
            a, b = int((cur + TARGET_PIECE) * SR), int((cur + MAX_PIECE) * SR)
            frames = audio[a: a + (b - a) // n * n].reshape(-1, n).astype(np.float32)
            cut, forced = cur + TARGET_PIECE + np.argmin(np.mean(frames**2, axis=1)) * FRAME, True
        pieces.append((cur, cut, forced))
        cur = cut
    pieces.append((cur, end, False))
    return pieces


def plan_pieces(audio: np.ndarray, pauses: np.ndarray, segments: list, labels: set, limit_minutes: float) -> list:
    total = len(audio) / SR
    plan, used = [], 0.0
    for idx, seg in enumerate(segments):
        if seg["name"] not in labels:
            continue
        start, end = widen(seg["start"], seg["end"], pauses, total)
        for n, (a, b, forced) in enumerate(split(start, end, pauses, audio)):
            if limit_minutes and used >= limit_minutes * 60:
                return plan
            plan.append({
                "id": f"{len(plan) + 1:04d}",
                "start": round(float(a), 2), "end": round(float(b), 2),
                "label": seg["name"], "segment": idx,
                "segment_edge": bool(n == 0 or b >= end),
                "forced_cut": bool(forced),
            })
            used += b - a
    return plan


# ---------- Monlam ----------

class Monlam:
    def __init__(self, base_url: str, api_key: str):
        self.base_url, self.headers = base_url.rstrip("/"), {"X-API-Key": api_key}

    def submit(self, wav: Path, idempotency_key: str) -> str:
        with wav.open("rb") as f:
            resp = requests.post(
                f"{self.base_url}/api/v1/speech-to-text/jobs",
                headers={**self.headers, "x-idempotency-key": idempotency_key},
                files={"file": (wav.name, f, "audio/wav")},
                data={"language": "bo", "task": "transcribe", "return_timestamps": "true"},
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()["job_id"]

    def wait(self, job_id: str) -> dict:
        started = time.monotonic()
        while time.monotonic() - started < JOB_TIMEOUT:
            resp = requests.get(f"{self.base_url}/api/v1/speech-to-text/jobs/{job_id}",
                                headers=self.headers, timeout=60)
            resp.raise_for_status()
            job = resp.json()
            if job["status"] in ("COMPLETED", "FAILED"):
                return job
            time.sleep(POLL_SECONDS)
        raise TimeoutError(f"job {job_id} not finished after {JOB_TIMEOUT}s")


def load_result(piece: dict, out_dir: Path):
    """A saved reply for this piece, or None. Ignored if it was made for different audio."""
    result_file = out_dir / "results" / f"piece_{piece['id']}.json"
    if not result_file.exists():
        return None
    job = json.loads(result_file.read_text())
    made_for = job.get("piece")
    if made_for and (made_for["start"], made_for["end"]) != (piece["start"], piece["end"]):
        return None
    return job


def transcribe_piece(api: Monlam, audio: np.ndarray, piece: dict, stem: str, out_dir: Path) -> dict:
    saved = load_result(piece, out_dir)
    if saved:
        return saved
    result_file = out_dir / "results" / f"piece_{piece['id']}.json"

    wav = out_dir / "pieces" / f"piece_{piece['id']}_{piece['start']:.2f}.wav"
    sf.write(wav, audio[int(piece["start"] * SR): int(piece["end"] * SR)], SR, subtype="PCM_16")

    last_error = None
    for attempt in range(1, RETRIES + 1):
        # Same key while retrying a submit, so Monlam does not bill a duplicate;
        # a new key after a FAILED job, since that request has to run again.
        key = f"{stem}-{piece['id']}-{piece['start']:.2f}-try{attempt}"
        try:
            started = time.monotonic()
            for submit_try in range(3):
                try:
                    job_id = api.submit(wav, key)
                    break
                except requests.RequestException:
                    if submit_try == 2:
                        raise
                    time.sleep(5 * (submit_try + 1))
            job = api.wait(job_id)
            if job["status"] == "COMPLETED":
                job["seconds_taken"] = round(time.monotonic() - started, 1)
                job["piece"] = {"start": piece["start"], "end": piece["end"]}
                result_file.write_text(json.dumps(job, ensure_ascii=False, indent=2))
                return job
            last_error = job.get("error") or "FAILED"
        except Exception as exc:  # network trouble, timeouts, HTTP errors
            last_error = str(exc)
        log(f"  piece {piece['id']}: attempt {attempt} failed ({last_error}); retrying")
        time.sleep(5 * attempt)
    raise RuntimeError(f"piece {piece['id']} failed after {RETRIES} attempts: {last_error}")


# ---------- checks and output ----------

def syllables(text: str) -> list:
    return [s for s in re.split(r"[་།༎\s]+", text) if s]


def warnings_for(piece: dict, result: dict) -> list:
    text = (result.get("text") or "").strip()
    seconds = piece["end"] - piece["start"]
    syl = syllables(text)
    rate = len(syl) / seconds if seconds else 0
    found = []
    if not text:
        found.append("empty result")
    elif rate < TOO_FEW_SYL_PER_S:
        found.append(f"too little text ({rate:.1f} syllables/s) - words may be missing")
    elif rate > TOO_MANY_SYL_PER_S:
        found.append(f"too much text ({rate:.1f} syllables/s) - text may be invented")
    for n in (3, 4, 5):  # the same phrase three or more times in a row
        grams = [tuple(syl[i:i + n]) for i in range(len(syl) - n + 1)]
        if any(grams[i] == grams[i + n] == grams[i + 2 * n] for i in range(len(grams) - 2 * n)):
            found.append("a phrase repeats several times")
            break
    if piece["forced_cut"]:
        found.append("cut without a pause - a word may be split")
    return found


def is_garbled(piece: dict, result: dict) -> bool:
    return any(w.startswith(("too much text", "a phrase repeats")) for w in warnings_for(piece, result))


def halve(piece: dict, pauses: np.ndarray, audio: np.ndarray) -> list:
    """Cut a piece in two at the longest pause near its middle (or the quietest moment)."""
    length = piece["end"] - piece["start"]
    lo, hi = piece["start"] + 0.35 * length, piece["start"] + 0.65 * length
    options = pauses_between(pauses, lo, hi)
    forced = not len(options)
    if not forced:
        cut = float(options[np.argmax(options[:, 1] - options[:, 0])].mean())
    else:
        n = int(FRAME * SR)
        a, b = int(lo * SR), int(hi * SR)
        frames = audio[a: a + (b - a) // n * n].reshape(-1, n).astype(np.float32)
        cut = lo + float(np.argmin(np.mean(frames**2, axis=1))) * FRAME
    cut = round(cut, 2)
    return [
        {**piece, "id": piece["id"] + "a", "end": cut, "forced_cut": piece["forced_cut"] or forced},
        {**piece, "id": piece["id"] + "b", "start": cut, "forced_cut": piece["forced_cut"] or forced},
    ]


def send_all(api: Monlam, audio: np.ndarray, pieces: list, stem: str, out_dir: Path, results: dict) -> list:
    """Transcribe pieces PARALLEL at a time, logging progress. Returns ids that failed."""
    todo = [p for p in pieces if not load_result(p, out_dir)]
    for p in pieces:
        if p not in todo:
            results[p["id"]] = load_result(p, out_dir)
    if not todo:
        return []
    started, sent_seconds, finished, failed = time.monotonic(), 0.0, 0, []
    log(f"Sending {len(todo)} pieces, {PARALLEL} at a time ...")
    with ThreadPoolExecutor(PARALLEL) as pool:
        futures = {pool.submit(transcribe_piece, api, audio, p, stem, out_dir): p for p in todo}
        for future in as_completed(futures):
            p = futures[future]
            finished += 1
            try:
                results[p["id"]] = job = future.result()
            except Exception as exc:
                failed.append(p["id"])
                log(f"FAILED   piece {p['id']} [{fmt(p['start'])}]: {exc}")
                continue
            sent_seconds += p["end"] - p["start"]
            text = (job.get("result") or {}).get("text") or ""
            warns = warnings_for(p, job.get("result") or {})
            eta = (time.monotonic() - started) / finished * (len(todo) - finished)
            log(f"done {finished:3d}/{len(todo)}  piece {p['id']} [{fmt(p['start'])}, {p['end'] - p['start']:4.1f}s] "
                f"in {job.get('seconds_taken', 0):4.1f}s  {len(syllables(text)):3d} syllables"
                f"{'  ⚠ ' + '; '.join(warns) if warns else ''}  | ~${sent_seconds / 60 * COST_PER_MINUTE:.2f} "
                f"so far, ~{eta / 60:.0f} min left")
    return failed


def build_transcript(plan: list, results: dict, segments: list, out_dir: Path) -> tuple:
    lines = []
    covered_until = max(p["end"] for p in plan)
    for piece in plan:
        result = results[piece["id"]].get("result") or {}
        warns = warnings_for(piece, result)
        segs = result.get("segments") or [{"start": 0, "end": piece["end"] - piece["start"],
                                           "text": result.get("text") or ""}]
        for k, s in enumerate(segs):
            lines.append({
                "start": round(piece["start"] + s["start"], 2),
                "end": round(min(piece["start"] + s["end"], piece["end"]), 2),
                "speaker": piece["label"], "text": s["text"].strip(), "piece": piece["id"],
                "warnings": warns if k == 0 else [],
                "near_switch": piece["segment_edge"] and (k == 0 or k == len(segs) - 1),
            })
    sent = {p["segment"] for p in plan}
    for idx, seg in enumerate(segments):
        if idx not in sent and seg["start"] < covered_until:
            lines.append({"start": seg["start"], "end": seg["end"], "speaker": seg["name"],
                          "text": None, "piece": None, "warnings": [], "near_switch": False})
    lines.sort(key=lambda l: l["start"])

    (out_dir / "transcript_raw.json").write_text(json.dumps(lines, ensure_ascii=False, indent=2))
    with open(out_dir / "transcript_raw.txt", "w") as f:
        f.write("# ⚠ = please listen and check   ⓘ = at a speaker switch\n\n")
        for l in lines:
            when = f"[{fmt(l['start'])} - {fmt(l['end'])}]"
            if l["text"] is None:
                f.write(f"{when}  ({l['speaker']} - not transcribed)\n")
                continue
            mark = "⚠" if l["warnings"] else ("ⓘ" if l["near_switch"] else " ")
            f.write(f"{when} {mark} {l['text']}\n")
            for w in l["warnings"]:
                f.write(f"{' ' * 30}⚠ {w}\n")
    return lines, sum(1 for l in lines if l["warnings"])


def main() -> None:
    global _log_file
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--labels", default="teacher", help="comma-separated labels to send (default: teacher)")
    parser.add_argument("--limit-minutes", type=float, default=0, help="only the first N minutes of selected audio")
    parser.add_argument("--dry-run", action="store_true", help="show the plan and cost, send nothing")
    parser.add_argument("--yes", action="store_true", help="don't ask before sending")
    args = parser.parse_args()

    rec = recording_dir(args.audio)
    segments_file = speakers_dir(rec) / "segments.json"
    if not segments_file.exists():
        sys.exit(f"No speaker timeline: run detect_speakers.py on {args.audio.name} first.")
    out_dir, work = transcript_dir(rec), working_dir(rec)
    for sub in ("pieces", "results"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    _log_file = open(work / "progress.log", "a")
    log(f"===== {args.audio.name}  labels={args.labels}  limit={args.limit_minutes or 'none'} min =====")

    segments = json.loads(segments_file.read_text())
    log("Reading audio and finding pauses ...")
    audio = load_audio(args.audio)
    pauses = find_pauses(audio)
    plan = plan_pieces(audio, pauses, segments, set(args.labels.split(",")), args.limit_minutes)
    if not plan:
        sys.exit("Nothing to send.")
    (work / "plan.json").write_text(json.dumps(plan, indent=2))

    lengths = [p["end"] - p["start"] for p in plan]
    done = {p["id"] for p in plan if load_result(p, work)}
    todo = [p for p in plan if p["id"] not in done]
    todo_minutes = sum(p["end"] - p["start"] for p in todo) / 60
    log(f"Plan: {len(plan)} pieces covering {fmt(plan[0]['start'])} - {fmt(plan[-1]['end'])}")
    log(f"      piece length {min(lengths):.0f}-{max(lengths):.0f}s (typical {np.median(lengths):.0f}s), "
        f"{sum(p['forced_cut'] for p in plan)} cut without a pause")
    log(f"      already done: {len(done)}   to send: {len(todo)} pieces, {todo_minutes:.1f} min, "
        f"estimated cost ${todo_minutes * COST_PER_MINUTE:.2f}")
    if args.dry_run:
        return
    if todo and not args.yes and input("Send to Monlam? (yes/no) ").strip().lower() not in ("y", "yes"):
        sys.exit("Cancelled - nothing sent.")

    load_env(PROJECT_ROOT / ".env")
    api_key = os.environ.get("MONLAM_API_KEY")
    if not api_key:
        sys.exit("MONLAM_API_KEY is not set (put it in .env)")
    api = Monlam(os.environ.get("MONLAM_BASE_URL", "https://api-v1.monlamai.studio"), api_key)

    started, results = time.monotonic(), {}
    failed = send_all(api, audio, plan, rec.name, work, results)
    if failed:
        log(f"{len(failed)} pieces failed: {', '.join(failed)}. Run the same command again to retry only those.")
        sys.exit(1)

    for round_no in range(1, RESPLIT_ROUNDS + 1):
        garbled = [p for p in plan if is_garbled(p, results[p["id"]]["result"] or {})
                   and p["end"] - p["start"] >= 2 * MIN_HALF]
        if not garbled:
            break
        log(f"Re-split round {round_no}: {len(garbled)} garbled piece(s) "
            f"({', '.join(p['id'] for p in garbled)}) - cutting in half and resending")
        new_plan, halves = [], []
        for p in plan:
            if p in garbled:
                parts = halve(p, pauses, audio)
                new_plan += parts
                halves += parts
            else:
                new_plan.append(p)
        plan = new_plan
        failed = send_all(api, audio, halves, rec.name, work, results)
        if failed:
            log(f"{len(failed)} pieces failed: {', '.join(failed)}. Run the same command again to retry only those.")
            sys.exit(1)
    (work / "plan.json").write_text(json.dumps(plan, indent=2))

    lines, warned = build_transcript(plan, results, segments, out_dir)
    log(f"Finished in {(time.monotonic() - started) / 60:.1f} min. {len(lines)} lines, {warned} with ⚠ warnings.")
    log(f"Transcript: {show(out_dir / 'transcript_raw.txt')}")


if __name__ == "__main__":
    main()
