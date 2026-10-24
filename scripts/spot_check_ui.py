"""A small web page for the blind speaker spot-check: play each clip, pick what you hear, see the score.

Usage:
    .venv/bin/python scripts/spot_check_ui.py AUDIO_FILE [--port 8765]
    then open http://localhost:8765

Uses the clips made by `spot_check.py make`. Answers are saved to the same answers.txt
(so `spot_check.py score` still works), and the page shows the score right after Submit.
The clip times stay hidden until then, so the check stays blind. Runs only on this computer.
"""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from paths import recording_dir, speakers_dir
from spot_check import CODES, compare, read_answers, write_answers

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speaker check</title>
<style>
:root { --bg:#f7f5f1; --card:#fff; --ink:#1f1d1a; --muted:#6b665e; --line:#e4dfd6; --accent:#8a4b12;
        --accent-soft:#f3e6d8; --ok:#2f7d4f; --ok-soft:#e3f1e8; --bad:#b3261e; --bad-soft:#f8e1df; }
@media (prefers-color-scheme: dark) { :root { --bg:#181715; --card:#22201d; --ink:#ece8e1; --muted:#a39d93;
        --line:#36332e; --accent:#e0a060; --accent-soft:#3a2c1f; --ok:#6fc28f; --ok-soft:#1f3328; --bad:#f08a82; --bad-soft:#3d2321; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width:860px; margin:0 auto; padding:24px 16px 120px; }
h1 { font-size:22px; margin:0 0 4px; }
.sub { color:var(--muted); margin:0 0 18px; }
.legend { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 18px; }
.legend span { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:4px 10px; font-size:13px; }
.legend b { color:var(--accent); }
.row { display:grid; grid-template-columns:44px 52px 1fr; gap:12px; align-items:center; background:var(--card);
       border:1px solid var(--line); border-radius:12px; padding:10px 12px; margin-bottom:8px; }
.row.focus { outline:2px solid var(--accent); outline-offset:-1px; }
.row.ok { background:var(--ok-soft); border-color:var(--ok); }
.row.bad { background:var(--bad-soft); border-color:var(--bad); }
.num { font-weight:600; color:var(--muted); text-align:center; }
.play { width:44px; height:44px; border-radius:50%; border:1px solid var(--line); background:var(--accent-soft);
        color:var(--accent); font-size:16px; cursor:pointer; position:relative; }
.play.on { background:var(--accent); color:var(--card); }
.choices { display:flex; flex-wrap:wrap; gap:6px; }
.choices button { border:1px solid var(--line); background:transparent; color:var(--ink); border-radius:8px;
                  padding:7px 10px; cursor:pointer; font-size:14px; min-width:44px; }
.choices button b { color:var(--accent); margin-right:4px; }
.choices button.sel { background:var(--accent); border-color:var(--accent); color:var(--card); }
.choices button.sel b { color:var(--card); }
.result { grid-column:3; font-size:13px; color:var(--muted); }
.result strong { color:var(--ink); }
.bar { position:fixed; left:0; right:0; bottom:0; background:var(--card); border-top:1px solid var(--line); }
.bar div { max-width:860px; margin:0 auto; padding:12px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
.bar .count { flex:1; color:var(--muted); }
.bar button { background:var(--accent); color:var(--card); border:0; border-radius:10px; padding:10px 18px; font-size:15px; cursor:pointer; }
.bar button:disabled { opacity:.45; cursor:default; }
.score { font-size:18px; font-weight:600; color:var(--ink); }
.keys { color:var(--muted); font-size:13px; }
@media (max-width:560px) { .row { grid-template-columns:32px 48px 1fr; } .choices button span { display:none; } }
</style></head>
<body><main>
<h1>Speaker check</h1>
<p class="sub">__REC__ &middot; Listen to each 10-second clip and pick what you hear. The clips come from random places;
their times stay hidden until you submit, so the check is blind.</p>
<div class="legend">
  <span><b>T</b> Teacher (Tibetan)</span><span><b>L</b> Translator (Ladakhi)</span><span><b>C</b> Chanting</span>
  <span><b>S</b> Static / silence</span><span><b>M</b> Mixed (more than one)</span>
</div>
<p class="sub">Hear two things in one clip? Pick both (click, or hold <b>Shift</b> with the letter) - e.g. Translator + Static.</p>
<div id="list"></div>
<p class="keys">Keyboard: <b>Space</b> play / pause &middot; <b>T L C S M</b> answer (then the next clip plays) &middot; <b>Shift</b> + letter: add a second answer &middot; <b>&uarr; &darr;</b> move</p>
</main>
<div class="bar"><div><span class="count" id="count"></span><button id="submit" disabled>Submit and score</button></div></div>
<audio id="player" preload="none"></audio>
<script>
const CLIPS = __CLIPS__;
const NAMES = {T:"Teacher", L:"Translator", C:"Chanting", S:"Static", M:"Mixed"};
const answers = __ANSWERS__;
let focus = 0, playing = null, scored = false;
const player = document.getElementById("player"), list = document.getElementById("list");

function render() {
  list.innerHTML = "";
  CLIPS.forEach((c, i) => {
    const row = document.createElement("div");
    row.className = "row" + (i === focus ? " focus" : "");
    row.id = "row" + i;
    row.innerHTML = `<div class="num">${i + 1}</div>
      <button class="play${playing === i ? " on" : ""}" aria-label="Play clip ${i + 1}">${playing === i ? "&#10074;&#10074;" : "&#9654;"}</button>
      <div class="choices">${Object.keys(NAMES).map(k =>
        `<button data-k="${k}" class="${(answers[c] || "").includes(k) ? "sel" : ""}"><b>${k}</b><span>${NAMES[k]}</span></button>`).join("")}</div>`;
    row.querySelector(".play").onclick = () => { focus = i; toggle(i); };
    row.querySelectorAll(".choices button").forEach(b => b.onclick = () => { focus = i; choose(b.dataset.k, false, true); });
    list.appendChild(row);
  });
  const n = CLIPS.filter(c => answers[c]).length;
  document.getElementById("count").textContent = `${n} of ${CLIPS.length} answered`;
  document.getElementById("submit").disabled = n === 0;
}
function toggle(i) {
  if (playing === i && !player.paused) { player.pause(); playing = null; render(); return; }
  player.src = "clip/" + CLIPS[i] + ".mp3"; player.play(); playing = i; render();
}
player.onended = () => { playing = null; render(); };
function choose(k, advance, add) {
  const c = CLIPS[focus], cur = answers[c] || "";
  if (!add || k === "M") answers[c] = (add && cur === k) ? "" : k;            // one answer (or clear it)
  else answers[c] = (cur.includes(k) ? cur.replace(k, "") : cur.replace("M", "") + k);  // toggle one of several
  if (!answers[c]) delete answers[c];
  if (advance && focus < CLIPS.length - 1) { focus++; toggle(focus); } else render();
  document.getElementById("row" + focus).scrollIntoView({block: "nearest"});
}
document.addEventListener("keydown", e => {
  if (scored) return;
  const k = e.key.toUpperCase();
  if (e.key === " ") { e.preventDefault(); toggle(focus); }
  else if (NAMES[k]) choose(k, !e.shiftKey, e.shiftKey);
  else if (e.key === "ArrowDown") { e.preventDefault(); focus = Math.min(focus + 1, CLIPS.length - 1); render(); }
  else if (e.key === "ArrowUp") { e.preventDefault(); focus = Math.max(focus - 1, 0); render(); }
  else return;
  document.getElementById("row" + focus).scrollIntoView({block: "nearest"});
});
document.getElementById("submit").onclick = async () => {
  const missing = CLIPS.length - CLIPS.filter(c => answers[c]).length;
  if (missing && !confirm(`${missing} clip(s) have no answer yet. Submit anyway?`)) return;
  const res = await fetch("answers", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(answers)});
  const data = await res.json();
  scored = true; player.pause(); playing = null; render();
  data.rows.forEach((r, i) => {
    const row = document.getElementById("row" + i);
    if (r.ok !== null) row.classList.add(r.ok ? "ok" : "bad");
    const detail = Object.entries(r.seconds_per_label).map(([n, s]) => `${n} ${Math.round(s)}s`).join(", ");
    const res = document.createElement("div"); res.className = "result";
    res.innerHTML = `${r.ok === null ? "no answer" : (r.ok ? "&#10003; match" : "&#10007; <strong>does not match</strong>")}
      &middot; at ${r.at} &middot; you: ${r.heard || "-"} &middot; script says: ${detail}`;
    row.appendChild(res);
  });
  document.getElementById("count").innerHTML = `<span class="score">${data.right} of ${data.checked} match (${data.checked ? Math.round(100 * data.right / data.checked) : 0}%)</span> &middot; saved to answers.txt`;
  document.getElementById("submit").textContent = "Saved";
  document.getElementById("submit").disabled = true;
};
render();
</script></body></html>
"""


def make_handler(out_dir: Path, title: str):
    check_dir = out_dir / "spot_check"
    clips = sorted(p.stem for p in check_dir.glob("check_*.mp3"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def send(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                saved = {k[:-4]: v for k, v in read_answers(out_dir).items() if v}
                page = (PAGE.replace("__REC__", title).replace("__CLIPS__", json.dumps(clips))
                        .replace("__ANSWERS__", json.dumps(saved)))
                self.send(page.encode(), "text/html; charset=utf-8")
            elif re.fullmatch(r"/clip/check_\d+\.mp3", self.path) and (check_dir / self.path[6:]).exists():
                self.send((check_dir / self.path[6:]).read_bytes(), "audio/mpeg")
            else:
                self.send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:
            if self.path != "/answers":
                return self.send(b"not found", "text/plain", 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            write_answers(out_dir, {c: v for c, v in body.items() if c in clips and v and all(x in CODES for x in v)})
            rows = compare(out_dir)
            checked = [r for r in rows if r["heard"] is not None]
            self.send(json.dumps({"rows": rows, "checked": len(checked), "right": sum(r["ok"] for r in checked)}).encode(),
                      "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    rec = recording_dir(args.audio)
    out_dir = speakers_dir(rec)
    if not (out_dir / "spot_check_key.json").exists():
        raise SystemExit(f"No clips yet: run  spot_check.py make {args.audio}  first.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(out_dir, rec.name))
    print(f"Speaker check for {rec.name}: http://localhost:{args.port}  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
