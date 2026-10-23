"""Correct a Monlam transcript: rule-based spelling fixes plus hand corrections, with a review log.

Usage:
    python scripts/correct_transcript.py prepare RECORDING   # rule-based pass + numbered chunks for hand review
    python scripts/correct_transcript.py build RECORDING     # merge hand corrections, write outputs
    RECORDING = the recording's folder name or a unique part of it, e.g. "Day 1 - Morning - Teachings" or teachings

Rule-based pass (only clear, mechanical spelling errors):
    - the gi/kyi/gyi particle (Monlam writes ཀི, གི, ཀྱི, གྱི almost at random, +ས and +ན) is
      rewritten as ཀྱི/གི/གྱི chosen by the previous syllable's final letter
    - "མྱི" -> "མི"; spoken forms in SPOKEN_FORMS (ཏོག་ཙི → ཏོག་ཙམ, ཡོད་རཱ → ཡོད་རེད, ...)
    - tidy stray punctuation left by Monlam ("། །།། །", "།་", double spaces)

Reads 2 - Transcript/transcript_raw.json. Hand corrections live in 2 - Transcript/working files/corrections/*.json as
    {"<line index>": {"text": "...corrected...", "note": "why", "tag": "teacher|prayer|participant|translator|unclear|announcement"}}
A line not listed keeps its rule-corrected text and the tag "teacher".

Outputs in output/Recordings/<recording>/2 - Transcript/ (review chunks go to working files/review_chunks/):
    transcript_corrected.txt    readable corrected transcript (tags shown for non-teaching lines)
    transcript_corrected.json   every line with original, corrected text, tag and note
    corrections_review.txt      only the changed lines: before -> after, with the reason
"""
import json
import re
import sys
from pathlib import Path

from paths import find_recording, transcript_dir, working_dir

CHUNK = 45

BASE = set(chr(c) for c in range(0x0F40, 0x0F6D))   # Tibetan base consonants
VOWEL_SIGNS = set(chr(c) for c in range(0x0F71, 0x0F85))


def particle_for(prev: str) -> str:
    """Correct form of the gi/kyi/gyi particle after syllable `prev`."""
    marks = [k for k, ch in enumerate(prev) if ch in VOWEL_SIGNS or "\u0f90" <= ch <= "\u0fbc"]
    if marks:  # letters after the vowel sign / subscript are the suffixes
        suffixes = [ch for ch in prev[marks[-1] + 1:] if ch in BASE]
    else:      # no vowel sign: work out root + suffixes from the letter count
        letters = [ch for ch in prev if ch in BASE]
        suffixes = letters[1:] if len(letters) <= 2 else letters[-2:] if letters[-1] == "ས" and letters[-2] in "གངབམ" else letters[-1:]
    final = suffixes[-1] if suffixes else ""
    if final in ("", "འ"):
        return "གི"                          # open syllable
    if final in "དབས":
        return "ཀྱི"
    if final in "ནམརལ":
        return "གྱི"
    return "གི"                              # ག, ང


# Spoken forms Monlam spells inconsistently -> one written form.
SPOKEN_FORMS = [
    (r"ཏ[ོི]ག་ཙི", "ཏོག་ཙམ"),
    (r"(?<=་)རཱ(?=[་།\s]|$)", "རེད"),         # ཡོད་རཱ → ཡོད་རེད
    (r"ཟེར་གོ་རེ(ད)?(?=[་།\s]|$)", "ཟེར་དགོས་རེད"),
]

PARTICLES = {p + end: end for p in ("ཀི", "གི", "ཀྱི", "གྱི") for end in ("", "ས", "ན")}


def rule_fix(text: str, tidy: bool = True) -> str:
    sylls = re.split(r"(་)", text)           # keep the tshegs so we can rebuild exactly
    words = [s for s in sylls]
    for i, w in enumerate(words):
        core = re.match(r"\s*([^\s།]*)", w).group(1)   # the syllable, without a following ། or space
        if core in PARTICLES:
            # previous real syllable
            j = i - 1
            while j >= 0 and words[j] in ("་", ""):
                j -= 1
            prev = re.split(r"[\s།]", words[j])[-1] if j >= 0 else ""
            new = particle_for(prev) + PARTICLES[core]
            words[i] = w.replace(core, new, 1)
        elif core == "མྱི":
            words[i] = w.replace(core, "མི", 1)
    out = "".join(words)
    for wrong, right in SPOKEN_FORMS:
        out = re.sub(wrong, right, out)
    if not tidy:  # hand-corrected text: keep its punctuation (e.g. verse marks "། །")
        return out
    out = re.sub(r"།\s*་", "། ", out)                  # "།་" -> "། "
    out = re.sub(r"(།\s*){2,}", "། ", out)             # runs of shads -> one
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+།", "།", out)
    return out


def load(name: str) -> tuple:
    rec = find_recording(name)
    d = transcript_dir(rec)
    return d, working_dir(rec), json.loads((d / "transcript_raw.json").read_text())


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def prepare(name: str) -> None:
    d, work, lines = load(name)
    chunks = work / "review_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    (work / "corrections").mkdir(exist_ok=True)
    rows = [(i, l) for i, l in enumerate(lines) if l["text"]]
    for n in range(0, len(rows), CHUNK):
        part = rows[n:n + CHUNK]
        with open(chunks / f"chunk_{n // CHUNK + 1:02d}.txt", "w") as f:
            for i, l in part:
                f.write(f"#{i} {fmt(l['start'])}{' ⓘ' if l['near_switch'] else ''}\n{rule_fix(l['text'])}\n")
    changed = sum(rule_fix(l["text"]) != l["text"] for _, l in rows)
    print(f"{len(rows)} text lines in {(len(rows) + CHUNK - 1) // CHUNK} chunks; rule-based pass changed {changed} lines")


def build(name: str) -> None:
    d, work, lines = load(name)
    hand = {}
    for f in sorted((work / "corrections").glob("*.json")):
        hand.update(json.loads(f.read_text()))

    out, review = [], []
    for i, l in enumerate(lines):
        row = {"start": l["start"], "end": l["end"], "speaker": l["speaker"], "original": l["text"]}
        if l["text"] is None:
            row.update(corrected=None, tag=l["speaker"], note="not transcribed")
        else:
            auto = rule_fix(l["text"])
            h = hand.get(str(i), {})
            corrected = rule_fix(h["text"], tidy=False) if "text" in h else auto
            row.update(corrected=corrected, tag=h.get("tag", "teacher"), note=h.get("note", ""))
            if row["corrected"] != l["text"] or row["tag"] != "teacher":
                review.append((i, l, row, auto))
        out.append(row)

    (d / "transcript_corrected.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    with open(d / "transcript_corrected.txt", "w") as f:
        f.write("# Corrected transcript. Lines not by the teacher, or not teaching, are tagged in [brackets].\n\n")
        for r in out:
            when = f"[{fmt(r['start'])} - {fmt(r['end'])}]"
            if r["corrected"] is None:
                f.write(f"{when}  ({r['tag']} - not transcribed)\n")
            elif r["tag"] != "teacher":
                f.write(f"{when}  [{r['tag']}] {r['corrected']}\n")
            else:
                f.write(f"{when}  {r['corrected']}\n")
    with open(d / "corrections_review.txt", "w") as f:
        f.write("# Every changed line: ORIGINAL (Monlam) -> CORRECTED, with the reason.\n")
        f.write("# 'spelling (rules)' = only the automatic particle/punctuation fixes.\n\n")
        for i, l, r, auto in review:
            reason = r["note"] or ("spelling (rules)" if r["corrected"] == auto else "")
            f.write(f"#{i} {fmt(l['start'])}  [{r['tag']}]  {reason}\n")
            f.write(f"  was: {l['text']}\n  now: {r['corrected']}\n\n")
    hand_count = sum(1 for _, _, r, auto in review if r["corrected"] != auto or r["tag"] != "teacher")
    print(f"{len(out)} lines; {len(review)} changed or tagged ({hand_count} by hand). "
          f"Wrote transcript_corrected.txt/.json and corrections_review.txt")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("prepare", "build"):
        sys.exit(__doc__)
    {"prepare": prepare, "build": build}[sys.argv[1]](sys.argv[2])
