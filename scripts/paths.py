"""Where every script reads and writes its files, so the output folder stays easy to browse.

output/
    README.txt                              what is where
    Course - <course title>/                the finished course: Section / Subsection / Unit folders
    Recordings/
        Day 1 - Morning - Teachings/        one folder per recording, named after where the audio lives
            1 - Speakers/                   who speaks when (detect_speakers.py, spot_check.py)
            2 - Transcript/                 transcript_raw.txt, transcript_corrected.txt, review notes
                working files/              pieces sent to Monlam, raw replies, log, hand corrections
            3 - Course draft/               curriculum, unit content, unit audio, Word outline
    Early tests/                            the first Monlam test clips and audio cleaning tests

A recording's folder name comes from its audio path:
    .../Day 1/Audio/Morning /teachings.mp3  ->  "Day 1 - Morning - Teachings"
so Day 2's teachings.mp3 gets its own folder instead of overwriting Day 1's.
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "output"
RECORDINGS = OUTPUT / "Recordings"
EARLY_TESTS = OUTPUT / "Early tests"

SPEAKERS = "1 - Speakers"
TRANSCRIPT = "2 - Transcript"
WORKING = "working files"
COURSE_DRAFT = "3 - Course draft"


def recording_name(audio: Path) -> str:
    """'Day 1 - Morning - Teachings' for .../Day 1/Audio/Morning /teachings.mp3."""
    parts = []
    day = next((p.name.strip() for p in audio.resolve().parents if re.fullmatch(r"Day\s*\d+", p.name.strip())), None)
    if day:
        parts.append(day)
        session = audio.resolve().parent.name.strip()
        if session not in (day, "Audio"):
            parts.append(session)
    parts.append(audio.stem[:1].upper() + audio.stem[1:])
    return " - ".join(parts)


def recording_dir(audio: Path) -> Path:
    return RECORDINGS / recording_name(audio)


def find_recording(name: str) -> Path:
    """A recording folder from an audio path, its full folder name, or a unique part of it ('teachings')."""
    if Path(name).suffix.lower() in (".mp3", ".wav", ".m4a"):
        return recording_dir(Path(name))
    folders = sorted(p for p in RECORDINGS.iterdir() if p.is_dir()) if RECORDINGS.exists() else []
    exact = [p for p in folders if p.name.casefold() == name.casefold()]
    matches = exact or [p for p in folders if name.casefold() in p.name.casefold()]
    if len(matches) == 1:
        return matches[0]
    found = "\n    ".join(p.name for p in (matches or folders)) or "(none yet)"
    sys.exit(f"{'Several' if matches else 'No'} recordings match {name!r}. Use one of:\n    {found}")


def speakers_dir(rec: Path) -> Path:
    return rec / SPEAKERS


def transcript_dir(rec: Path) -> Path:
    return rec / TRANSCRIPT


def working_dir(rec: Path) -> Path:
    return rec / TRANSCRIPT / WORKING


def course_draft_dir(rec: Path) -> Path:
    return rec / COURSE_DRAFT


def show(path: Path) -> str:
    """A path relative to the project, for messages."""
    return str(path.relative_to(PROJECT_ROOT))
