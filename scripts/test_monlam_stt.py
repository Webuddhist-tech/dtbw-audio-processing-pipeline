"""Send a short audio clip to Monlam AI speech-to-text and print the timestamped result.

Usage:
    python scripts/test_monlam_stt.py AUDIO_FILE [--start SECONDS] [--duration SECONDS]
    python scripts/test_monlam_stt.py AUDIO_FILE --whole [--start OFFSET_SECONDS]

By default a clip is cut with ffmpeg and saved to output/test/.
With --whole the file is sent as-is (e.g. an already-cut or cleaned clip) and the
results are saved next to it; --start then only shifts the printed timestamps.

The job is sent async and polled until done. Saved next to the audio:
    <name>.json  raw reply from Monlam
    <name>.txt   timestamped transcript, times shifted to the full recording
"""
import argparse
import json
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "test"
POLL_SECONDS = 5
TIMEOUT_SECONDS = 15 * 60


def load_env(path: Path) -> None:
    """Read KEY=VALUE lines from .env into os.environ."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def cut_clip(source: Path, start: float, duration: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(start), "-t", str(duration), "-i", str(source),
         "-c", "copy", str(dest)],
        check=True,
    )


def submit_job(base_url: str, api_key: str, clip: Path) -> str:
    with clip.open("rb") as f:
        resp = requests.post(
            f"{base_url}/api/v1/speech-to-text/jobs",
            headers={"X-API-Key": api_key},
            files={"file": (clip.name, f, mimetypes.guess_type(clip.name)[0] or "application/octet-stream")},
            data={"language": "bo", "task": "transcribe", "return_timestamps": "true"},
            timeout=120,
        )
    if not resp.ok:
        sys.exit(f"Submit failed: HTTP {resp.status_code} {resp.text}")
    job = resp.json()
    print(f"Job submitted: {job['job_id']} ({job.get('status')})")
    return job["job_id"]


def wait_for_job(base_url: str, api_key: str, job_id: str) -> dict:
    started = time.monotonic()
    last_status = None
    while time.monotonic() - started < TIMEOUT_SECONDS:
        resp = requests.get(
            f"{base_url}/api/v1/speech-to-text/jobs/{job_id}",
            headers={"X-API-Key": api_key},
            timeout=60,
        )
        if not resp.ok:
            sys.exit(f"Status check failed: HTTP {resp.status_code} {resp.text}")
        job = resp.json()
        if job["status"] != last_status:
            print(f"  [{time.monotonic() - started:5.0f}s] status: {job['status']}")
            last_status = job["status"]
        if job["status"] in ("COMPLETED", "FAILED"):
            return job
        time.sleep(POLL_SECONDS)
    sys.exit(f"Gave up after {TIMEOUT_SECONDS}s; job {job_id} still {last_status}")


def fmt_time(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def transcript_lines(result: dict, offset: float) -> list:
    lines = []
    for seg in result.get("segments") or []:
        lines.append(f"[{fmt_time(offset + seg['start'])} -> {fmt_time(offset + seg['end'])}] {seg['text']}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--start", type=float, default=300, help="clip start in seconds (default 300)")
    parser.add_argument("--duration", type=float, default=30, help="clip length in seconds (default 30)")
    parser.add_argument("--whole", action="store_true", help="send the file as-is instead of cutting a clip")
    args = parser.parse_args()

    load_env(PROJECT_ROOT / ".env")
    api_key = os.environ.get("MONLAM_API_KEY")
    base_url = os.environ.get("MONLAM_BASE_URL", "https://api.monlamai.studio").rstrip("/")
    if not api_key:
        sys.exit("MONLAM_API_KEY is not set (put it in .env)")
    if not args.audio.exists():
        sys.exit(f"Audio file not found: {args.audio}")

    if args.whole:
        clip = args.audio
        print(f"Sending as-is: {clip}")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        clip = OUTPUT_DIR / f"{args.audio.stem}_{int(args.start)}s_{int(args.duration)}s.mp3"
        cut_clip(args.audio, args.start, args.duration, clip)
        print(f"Clip: {args.duration:.0f}s from {fmt_time(args.start)} of {args.audio.name}")
        print(f"Saved clip to {clip}")
    job_id = submit_job(base_url, api_key, clip)

    job = wait_for_job(base_url, api_key, job_id)

    json_file = clip.with_suffix(".json")
    json_file.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    print(f"Saved raw reply to {json_file}")

    if job["status"] == "FAILED":
        sys.exit(f"Job failed: {job.get('error')}")

    result = job.get("result") or {}
    lines = transcript_lines(result, args.start)
    txt_file = clip.with_suffix(".txt")
    txt_file.write_text("\n".join(lines) + "\n")
    print(f"Saved transcript to {txt_file}")

    print(f"\nCost: {result.get('cost')}   Audio duration: {result.get('duration')}")
    print("\nFull text:\n" + (result.get("text") or "(empty)"))
    print(f"\nSegments ({len(lines)}), times shifted to the full recording:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
