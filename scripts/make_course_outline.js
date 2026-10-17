// Build the course outline Word file from output/course/<name>/curriculum.json.
// Usage: node scripts/make_course_outline.js teachings
// (needs the npm package "docx"; set NODE_PATH if it is installed elsewhere)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, LevelFormat, BorderStyle,
} = require("docx");

const name = process.argv[2] || "teachings";
const dir = path.join(__dirname, "..", "output", "course", name);
const course = JSON.parse(fs.readFileSync(path.join(dir, "curriculum.json"), "utf8"));

const TIB = "Monlam Uni OuChan2";   // Tibetan Unicode font; Word falls back if it is missing
const LAT = "Arial";
const GREY = "666666";
const ACCENT = "7A3E00";
const CONTENT_W = 9638;              // A4 (11906) minus 2 cm margins

const allFonts = (f) => ({ ascii: f, hAnsi: f, eastAsia: f, cs: f });
const bo = (text, o = {}) => new TextRun({ text, font: allFonts(TIB), size: o.size || 28, sizeComplexScript: o.size || 28, bold: o.bold, color: o.color });
const en = (text, o = {}) => new TextRun({ text, font: allFonts(LAT), size: o.size || 20, sizeComplexScript: o.size || 20, bold: o.bold, italics: o.italics, color: o.color });
const para = (children, o = {}) => new Paragraph({ children, spacing: { after: o.after ?? 80, before: o.before ?? 0 }, alignment: o.align, numbering: o.bullet ? { reference: "bullets", level: 0 } : undefined, heading: o.heading, border: o.border });

const cell = (children, width, fill) => new TableCell({
  children: Array.isArray(children) ? children : [children],
  width: { size: width, type: WidthType.DXA },
  shading: fill ? { type: ShadingType.CLEAR, color: "auto", fill } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
});

const units = [];
course.sections.forEach((s) => s.subsections.forEach((ss) => ss.units.forEach((u) => units.push(u))));
const tracksFile = path.join(dir, "audio", "tracks.json");
const tracks = fs.existsSync(tracksFile) ? JSON.parse(fs.readFileSync(tracksFile, "utf8")) : {};
const mins = (u, t) => (tracks[u.number] && tracks[u.number][t] ? tracks[u.number][t].minutes : null);
const totalBo = units.reduce((a, u) => a + (mins(u, "bo") || 0), 0);
const totalLad = units.reduce((a, u) => a + (mins(u, "lad") || 0), 0);
const minsLabel = (u) => `${mins(u, "bo") ?? "–"} / ${mins(u, "lad") ?? "–"}`;

// ---------- overview table ----------
const W = [1000, 3400, 2838, 1400, 1000];
const head = new TableRow({ tableHeader: true, children: ["#", "ཁ་བྱང་། (Title)", "English (for reviewers only)", "Original recording", "Min (བོད / ལ་དྭགས)"].map((h, i) =>
  cell(para([i === 1 ? bo(h, { size: 22, bold: true }) : en(h, { bold: true, size: 18 })], { after: 0 }), W[i], "E8DCCB")) });
const rows = [head];
course.sections.forEach((s) => {
  rows.push(new TableRow({ children: [
    cell(para([en(`Section ${s.number}`, { bold: true, size: 18 })], { after: 0 }), W[0], "F3EBDF"),
    cell(para([bo(s.title_bo, { bold: true, size: 24 })], { after: 0 }), W[1], "F3EBDF"),
    cell(para([en(s.title_en, { bold: true, size: 18 })], { after: 0 }), W[2], "F3EBDF"),
    cell(para([en("", { size: 18 })], { after: 0 }), W[3], "F3EBDF"),
    cell(para([en("", { size: 18 })], { after: 0 }), W[4], "F3EBDF"),
  ] }));
  s.subsections.forEach((ss) => {
    rows.push(new TableRow({ children: [
      cell(para([en(ss.number, { bold: true, size: 18 })], { after: 0 }), W[0], "FAF6F0"),
      cell(para([bo(ss.title_bo, { bold: true, size: 22 })], { after: 0 }), W[1], "FAF6F0"),
      cell(para([en(ss.title_en, { bold: true, size: 18 })], { after: 0 }), W[2], "FAF6F0"),
      cell(para([en("")], { after: 0 }), W[3], "FAF6F0"),
      cell(para([en("")], { after: 0 }), W[4], "FAF6F0"),
    ] }));
    ss.units.forEach((u) => rows.push(new TableRow({ children: [
      cell(para([en(u.number, { size: 18 })], { after: 0 }), W[0]),
      cell(para([bo(u.title_bo, { size: 22 })], { after: 0 }), W[1]),
      cell(para([en(u.title_en, { size: 18 })], { after: 0 }), W[2]),
      cell(para([en(`${u.audio.start_hms} – ${u.audio.end_hms}`, { size: 16 })], { after: 0 }), W[3]),
      cell(para([en(minsLabel(u), { size: 18 })], { after: 0, align: AlignmentType.RIGHT }), W[4]),
    ] })));
  });
});
const overview = new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: W, rows });

// ---------- unit detail ----------
function componentsLine(c) {
  const parts = [];
  parts.push(c.text ? "Text ✓" : "Text –");
  parts.push(c.audio ? (c.audio === true ? "Audio: Tibetan + Ladakhi ✓" : `Audio: ${c.audio} ✓`) : "Audio –");
  parts.push(c.key_points ? `Key points (${c.key_points})` : "Key points –");
  parts.push(c.mcq ? `MCQ (${c.mcq} questions)` : "MCQ –");
  return parts.join("     ");
}
function unitBlock(u) {
  const out = [];
  out.push(new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 240, after: 40 },
    children: [en(`Unit ${u.number}   `, { bold: true, color: ACCENT, size: 20 }), bo(u.title_bo, { bold: true, size: 28 })] }));
  out.push(para([en(u.title_en, { italics: true, color: GREY })], { after: 100 }));
  const a = u.audio;
  const silence = a.remove_silences_hms.length ? `  ·  silence removed: ${a.remove_silences_hms.join(", ")}` : "";
  const DW = [1700, CONTENT_W - 1700];
  const r = (label, children) => new TableRow({ children: [cell(para([en(label, { bold: true, size: 18 })], { after: 0 }), DW[0], "F7F3EE"), cell(children, DW[1])] });
  out.push(new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: DW, rows: [
    r("Audio", [
      para([en(`Original recording ${a.start_hms} – ${a.end_hms}${silence}`, { size: 18 })], { after: 20 }),
      ...["bo", "lad"].map((t) => {
        const tr = tracks[u.number] && tracks[u.number][t];
        const label = t === "bo" ? "Tibetan (teacher)" : "Ladakhi (translator)";
        return para([en(tr ? `${label}: ${tr.file}  ·  ${tr.minutes} min  ·  ${tr.pieces} piece(s)` : `${label}: none`, { size: 18 })], { after: 0 });
      }),
    ]),
    r("ནང་དོན།", para([bo(u.summary_bo, { size: 24 })], { after: 0 })),
    r("Content (EN)", para([en(u.summary_en, { size: 18 })], { after: 0 })),
    r("Components", para([en(componentsLine(u.components), { size: 18 })], { after: 0 })),
    ...(u.notes ? [r("Note", para([en(u.notes, { size: 18, italics: true })], { after: 0 }))] : []),
  ] }));
  return out;
}

const body = [];
body.push(para([bo(course.course_title_bo, { size: 44, bold: true })], { after: 40 }));
body.push(para([en(course.course_title_en, { size: 24, color: GREY })], { after: 40 }));
body.push(para([en("Course outline – draft 2 for review", { size: 24, bold: true, color: ACCENT })], { after: 40 }));
body.push(para([en(`Source: ${course.source}  ·  ${units.length} units  ·  audio: ${Math.round(totalBo)} min Tibetan + ${Math.round(totalLad)} min Ladakhi  ·  For WeBuddhist Academy (Open edX)`, { size: 18, color: GREY })],
  { after: 240, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "C9B79C", space: 6 } } }));

body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [en("How the course is organised", { bold: true, size: 28 })], spacing: { after: 100 } }));
body.push(para([en("Section → Subsection → Unit, as in Open edX. Each unit has these components:", { size: 20 })]));
[["Text", "the teacher's own words in standard Tibetan (from the corrected transcript)"],
 ["Audio", "two clips from the original recording: Tibetan (the teacher) and Ladakhi (the translator)"],
 ["Key points", "3–5 short points summarising the unit"],
 ["MCQ", "multiple-choice questions in Tibetan, with the correct answer marked for the uploader"]].forEach(([k, v]) =>
  body.push(para([en(`${k}: `, { bold: true }), en(v)], { bullet: true, after: 40 })));
body.push(para([en("Audio rules:", { bold: true })], { before: 120 }));
course.rules.forEach((rule) => body.push(para([en(rule)], { bullet: true, after: 40 })));
body.push(para([en("Titles are in Tibetan only. The English lines in this document are only to help reviewers and are not uploaded.", { italics: true, color: GREY })], { before: 120, after: 200 }));

body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [en("Overview", { bold: true, size: 28 })], spacing: { after: 100 } }));
body.push(overview);

course.sections.forEach((s) => {
  body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, spacing: { after: 60 },
    children: [en(`Section ${s.number}   `, { bold: true, color: ACCENT, size: 22 }), bo(s.title_bo, { bold: true, size: 36 })] }));
  body.push(para([en(s.title_en, { italics: true, color: GREY, size: 22 })], { after: 160 }));
  s.subsections.forEach((ss) => {
    body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 320, after: 60 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9B79C", space: 4 } },
      children: [en(`Subsection ${ss.number}   `, { bold: true, color: ACCENT, size: 20 }), bo(ss.title_bo, { bold: true, size: 32 }), en(`   ${ss.title_en}`, { italics: true, color: GREY, size: 20 })] }));
    ss.units.forEach((u) => body.push(...unitBlock(u)));
  });
});

body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [en("Questions for the reviewer", { bold: true, size: 28 })], spacing: { after: 100 } }));
[
  "Course title: is 'སད་པའི་སེམས། ལོ་རིམ་གཉིས་པ།' right, or should the course carry the programme's name?",
  "Section title: one section per day/session (Day 1 here). Should the afternoon session be part of Section 1 or its own section?",
  "Unit 1.2.1 is the longest teaching unit (12 min). It is one translator turn, so we suggest keeping it whole.",
  "Please check every title for wording and spelling.",
].forEach((q) => body.push(para([en(q)], { bullet: true, after: 60 })));

const doc = new Document({
  creator: "dtbw-audio-processing-pipeline",
  title: `${course.course_title_bo} – course outline`,
  styles: { default: { document: { run: { font: LAT, size: 20 } } } },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 260 } } } }] }] },
  sections: [{ properties: { page: { margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } }, children: body }],
});
const out = path.join(dir, "course_outline_v2.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out); });
