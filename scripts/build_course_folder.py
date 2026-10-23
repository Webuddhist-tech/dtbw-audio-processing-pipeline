"""Build the course folder: Course / Section / Subsection / Unit, each unit with its text and audio.

Usage:
    python3 scripts/build_course_folder.py teachings
    (the recording's folder name, or a unique part of it - see paths.py)

Reads (all in output/Recordings/<recording>/3 - Course draft/):
    curriculum.json                 structure, titles, audio ranges
    content/unit_text_draft.json    each unit's teacher lines (from the corrected transcript)
    content/units_content.json      paragraph breaks, dropped fragments, key points, MCQs
    unit audio/                     unit_<n>_bo.mp3 / unit_<n>_lad.mp3 from cut_unit_audio.py

Writes output/Course - <course title>/ :
    Section 1 - <title>/Subsection 1.1 - <title>/Unit 1.1.1 - <title>/
        content.txt          text, key points, MCQs (with the correct answer marked)
        audio_tibetan.mp3    teacher
        audio_ladakhi.mp3    translator (not for units marked Tibetan only)
    course_outline.txt       the whole structure at a glance

Folder names use " - " instead of ":" because ":" is not allowed in Windows file names and
shows as "/" on a Mac.
"""
import json
import re
import shutil
import sys
from paths import OUTPUT, course_draft_dir, find_recording, show

TIB_DIGITS = str.maketrans("0123456789", "༠༡༢༣༤༥༦༧༨༩")
OPTION_LETTERS = ["ཀ", "ཁ", "ག", "ང", "ཅ"]
RULE = "━" * 12


def safe(name: str) -> str:
    """A folder name that works on Mac, Windows and Linux."""
    name = re.sub(r'[\\/:*?"<>|]', "-", name)
    return re.sub(r"\s+", " ", name).strip().rstrip(".")


def tib(n: int) -> str:
    return str(n).translate(TIB_DIGITS)


def unit_text(lines: list, content: dict) -> list:
    """Paragraphs of the teacher's words for one unit."""
    drop, breaks = set(content.get("drop", [])), set(content.get("breaks", [0]))
    paras, cur = [], []
    for i, line in enumerate(lines):
        if i in breaks and cur:
            paras.append(" ".join(cur))
            cur = []
        if i not in drop:
            cur.append(line["text"].replace("(English:) ", ""))
    if cur:
        paras.append(" ".join(cur))
    return [re.sub(r"\s{2,}", " ", p).strip() for p in paras if p.strip()]


def write_content(path, source: str, section: dict, sub: dict, unit: dict, paras: list, content: dict, tracks: dict) -> None:
    a = unit["audio"]
    audio_files = []
    if "bo" in tracks:
        audio_files.append(f"audio_tibetan.mp3 (བོད་སྐད། {tracks['bo']['minutes']} min)")
    if "lad" in tracks:
        audio_files.append(f"audio_ladakhi.mp3 (ལ་དྭགས་སྐད། {tracks['lad']['minutes']} min)")
    out = [
        f"Unit {unit['number']} - {unit['title_bo']}",
        f"Section {section['number']} - {section['title_bo']}  ›  Subsection {sub['number']} - {sub['title_bo']}",
        f"Audio: {', '.join(audio_files)}",
        f"Source: {source}, {a['start_hms']} - {a['end_hms']}",
        "",
    ]
    if paras:
        out += [f"{RULE} ཡི་གེ། (Text) {RULE}", ""]
        for p in paras:
            out += [p, ""]
    else:
        out += ["(No text for this unit - audio only.)", ""]
    if content.get("key_points"):
        out += [f"{RULE} གནད་བསྡུས། (Key points) {RULE}", ""]
        out += [f"{tib(n)}། {kp}" for n, kp in enumerate(content["key_points"], 1)] + [""]
    if content.get("mcq"):
        out += [f"{RULE} འདྲི་ཚིག (Multiple-choice questions) {RULE}", ""]
        for n, q in enumerate(content["mcq"], 1):
            out.append(f"{tib(n)}། {q['q']}")
            for k, opt in enumerate(q["options"]):
                mark = "   ✓" if k == q["answer"] else ""
                out.append(f"    {OPTION_LETTERS[k]}། {opt}{mark}")
            out += [f"    ལན་ཡང་དག {OPTION_LETTERS[q['answer']]}།", ""]
    if content.get("reflection"):
        out += [f"{RULE} བསམ་ཞིབ། (Reflection question) {RULE}", "", content["reflection"], ""]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    rec = find_recording(sys.argv[1] if len(sys.argv) > 1 else "teachings")
    src = course_draft_dir(rec)
    course = json.loads((src / "curriculum.json").read_text())
    drafts = json.loads((src / "content" / "unit_text_draft.json").read_text())
    contents = json.loads((src / "content" / "units_content.json").read_text())
    tracks = json.loads((src / "unit audio" / "tracks.json").read_text())

    root = OUTPUT / safe(f"Course - {course['course_title_bo']}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    outline = [course["course_title_bo"], course.get("course_title_en", ""), ""]
    for section in course["sections"]:
        s_dir = root / safe(f"Section {section['number']} - {section['title_bo']}")
        outline.append(s_dir.name)
        for sub in section["subsections"]:
            ss_dir = s_dir / safe(f"Subsection {sub['number']} - {sub['title_bo']}")
            outline.append(f"    {ss_dir.name}")
            for unit in sub["units"]:
                n = unit["number"]
                u_dir = ss_dir / safe(f"Unit {n} - {unit['title_bo']}")
                u_dir.mkdir(parents=True)
                content = contents[n]
                paras = unit_text(drafts[n]["lines"], content) if content.get("text", True) else []
                unit_tracks = tracks.get(n, {})
                write_content(u_dir / "content.txt", rec.name, section, sub, unit, paras, content, unit_tracks)
                for t, fname in (("bo", "audio_tibetan.mp3"), ("lad", "audio_ladakhi.mp3")):
                    if t in unit_tracks:
                        shutil.copyfile(src / "unit audio" / unit_tracks[t]["file"], u_dir / fname)
                files = ", ".join(sorted(p.name for p in u_dir.iterdir()))
                outline.append(f"        {u_dir.name}    [{files}]")
                print(f"{n}: {len(paras)} paragraphs, {len(content.get('key_points', []))} key points, "
                      f"{len(content.get('mcq', []))} MCQs, audio {'+'.join(unit_tracks)}")
    (root / "course_outline.txt").write_text("\n".join(outline) + "\n", encoding="utf-8")
    print(f"\nCourse folder: {show(root)}")


if __name__ == "__main__":
    main()
