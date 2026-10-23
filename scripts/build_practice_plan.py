"""Build the practice plan folder: one folder per step, each with its text and audio.

Usage:
    python3 scripts/build_practice_plan.py practice
    (the recording's folder name, or a unique part of it - see paths.py)

Reads (in output/Recordings/<recording>/):
    3 - Practice plan draft/steps.json      steps, titles, audio ranges, transcript lines, prayers
    3 - Practice plan draft/step audio/     step_<n>_bo.mp3 / step_<n>_lad.mp3 from cut_unit_audio.py
    2 - Transcript/transcript_corrected.json  the teacher's words for each step's instructions

Writes output/Practice plan - <plan title>/ :
    Step 1 - <title>/
        content.txt          how to do it, the teacher's instructions, the prayer
        audio_tibetan.mp3    teacher and group chanting
        audio_ladakhi.mp3    translator (and group chanting, if cut with --chanting both)
    plan_outline.txt         the whole routine at a glance
    prayers_review.txt       the prayer drafts next to what was heard, for the reviewer
"""
import json
import re
import shutil
import sys

from build_course_folder import RULE, safe
from paths import OUTPUT, find_recording, practice_draft_dir, show, transcript_dir

# Fillers that read badly in a written instruction; the audio keeps everything.
FILLER_LINE = re.compile(r"^(ཨ|ཨོ|ཨེ|ཧེ|འོ|ཨ་འོ|དེ་ནས|ད)[།\s]*$")
FILLER_START = re.compile(r"^((ཨོ་ལགས་སོ|ཨོ|ཨ|ཨེ་ནས)[།་]?\s*)+")


def instructions(step: dict, rows: list) -> str:
    """The teacher's own words for one step, as one paragraph."""
    drop = set(step.get("drop", []))
    words = []
    for first, last in step["lines"]:
        for i in range(first, last + 1):
            r = rows[i]
            if i in drop or r["corrected"] is None or r["tag"] != "teacher":
                continue
            text = FILLER_START.sub("", r["corrected"].replace("(English:) ", "")).strip()
            text = re.sub(r"\s*ཨ།$", "", text)
            if text and not FILLER_LINE.match(text):
                words.append(text)
    # a line that ends mid-phrase (on a tsheg) runs straight on into the next one
    joined = "".join(w + ("" if w.endswith("་") else " ") for w in words)
    return re.sub(r"\s{2,}", " ", joined).strip()


def write_content(path, plan: dict, step: dict, text: str, tracks: dict) -> None:
    a = step["audio"]
    audio = []
    if "bo" in tracks:
        audio.append(f"audio_tibetan.mp3 (བོད་སྐད། {tracks['bo']['minutes']} min)")
    if "lad" in tracks:
        audio.append(f"audio_ladakhi.mp3 (ལ་དྭགས་སྐད། {tracks['lad']['minutes']} min)")
    left_out = "".join(f"; left out {s}" for s in a.get("skip_hms", []))
    out = [f"Step {step['number']} - {step['title_bo']}", plan["plan_title_bo"]]
    if step["how_bo"]:
        out.append(f"ཇི་ལྟར་བྱ། {step['how_bo']}")
    out += [f"Audio: {', '.join(audio)}",
            f"Source: {plan['source'].split(' (')[0]}, {a['start_hms']} - {a['end_hms']}{left_out}", ""]
    if text:
        out += [f"{RULE} ལམ་སྟོན། (Instructions) {RULE}", "", text, ""]
    prayer = step.get("prayer")
    if prayer:
        out += [f"{RULE} འདོན་ཆ། (Prayer) {RULE}", "", prayer["title_bo"], ""]
        out += [prayer["text"], ""] if prayer["text"] else ["[ དཔེ་ཆ་ནས་འདྲེན་དགོས། - copy from the prayer booklet ]", ""]
        if prayer.get("draft"):
            out += ["[ Draft from the recording; [?…] = not sure. Check against the prayer booklet - see prayers_review.txt ]", ""]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    rec = find_recording(sys.argv[1] if len(sys.argv) > 1 else "practice")
    src = practice_draft_dir(rec)
    plan = json.loads((src / "steps.json").read_text())
    tracks = json.loads((src / "step audio" / "tracks.json").read_text())
    rows = json.loads((transcript_dir(rec) / "transcript_corrected.json").read_text())

    root = OUTPUT / safe(f"Practice plan - {plan['plan_title_bo']}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    outline = [plan["plan_title_bo"], plan.get("plan_title_en", ""), f"For: {plan.get('platform', '')}", ""]
    for step in plan["steps"]:
        n = step["number"]
        s_dir = root / safe(f"Step {n} - {step['title_bo']}")
        s_dir.mkdir()
        text = instructions(step, rows)
        step_tracks = tracks.get(n, {})
        write_content(s_dir / "content.txt", plan, step, text, step_tracks)
        for t, fname in (("bo", "audio_tibetan.mp3"), ("lad", "audio_ladakhi.mp3")):
            if t in step_tracks:
                shutil.copyfile(src / "step audio" / step_tracks[t]["file"], s_dir / fname)
        mins = " / ".join(f"{step_tracks[t]['minutes']}" for t in ("bo", "lad") if t in step_tracks)
        outline.append(f"{s_dir.name}    ({step['how_en'] or step['title_en']}; {mins} min)")
        prayer = step.get("prayer") or {}
        print(f"Step {n}: {len(text.split())} words of instructions, "
              f"prayer: {'text' if prayer.get('text') else ('from booklet' if prayer else 'none')}, audio {'+'.join(step_tracks)}")
    (root / "plan_outline.txt").write_text("\n".join(outline) + "\n", encoding="utf-8")
    if (src / "prayers_review.txt").exists():  # for the reviewer: the prayers heard in the recording, side by side
        shutil.copyfile(src / "prayers_review.txt", root / "prayers_review.txt")
    print(f"\nPractice plan folder: {show(root)}")


if __name__ == "__main__":
    main()
