// Steps 4 and 5: write the answer from the evidence, then check every claim
// against the exact lines it cites.
import { tokenize, stem, longDate } from "./text.js";

// ── Context shown with every evidence line (and used when verifying) ─────────
export function sourceHeader(engine, d) {
  const who = engine.residentName(d);
  return `${engine.label(d.type)} ${d.key}${who ? `, ${who}` : ""}${d.date ? `, ${d.date} (${longDate(d.date)})` : ""}${d.section ? `, ${d.section}` : ""}`;
}

const cite = (e) => ({ id: e.id, key: e.doc.key, line: e.line });

// ── Answer without a language model: quote the evidence ──────────────────────
export function quotedAnswer(engine, result) {
  return {
    mode: "quoted",
    lead: result.evidence.length
      ? `The most relevant lines from ${new Set(result.evidence.map((e) => e.doc.key)).size} records, quoted exactly.`
      : "No record matched this question. Try naming a resident, an event such as a fall, or a time period.",
    claims: result.evidence.map((e) => ({ text: e.text, cites: [cite(e)], kind: "quoted", doc: e.doc })),
    facts: [],
  };
}

// ── Count questions: computed from every matching record ─────────────────────
export function countAnswer(engine, p, counted) {
  const n = counted.docs.length;
  const what = p.incidentType ? `${p.incidentType.toLowerCase()} incident` : engine.label(p.hardTypes[0] || "").toLowerCase();
  const who = p.residents.length ? ` for ${p.residents.map((i) => engine.residents[i][1]).join(" and ")}` : "";
  const when = p.dates ? ` in ${p.dates.label} (${longDate(p.dates.from)} to ${longDate(p.dates.to)})` : ` across the whole dataset (${longDate(engine.meta.date_from)} to ${longDate(engine.meta.date_to)})`;
  const listed = counted.docs.slice(-12).reverse();
  const facts = [];
  const claims = [];
  const f1 = { id: "F1", text: `${n} ${what} record${n === 1 ? "" : "s"}${who}${when}.`,
    keys: counted.docs.map((d) => d.key) };
  facts.push(f1);
  claims.push({ text: `There ${n === 1 ? "is" : "are"} ${f1.text}`, kind: "computed", fact: f1,
    cites: counted.docs.slice(0, 40).map((d) => ({ id: "F1", key: d.key, line: 1 })),
    check: { status: "computed", note: `Counted every record that passes the filters, not only the top results (${n} records).` } });
  if (!p.residents.length && counted.byResident.length) {
    const top = counted.byResident.slice(0, 3).filter(([r]) => r >= 0);
    if (top.length) {
      const f2 = { id: "F2", text: `Most ${what} records: ${top.map(([r, c]) => `${engine.residents[r][1]} (${c})`).join(", ")}.`,
        keys: counted.docs.filter((d) => d.res === top[0][0]).map((d) => d.key) };
      facts.push(f2);
      claims.push({ text: f2.text, kind: "computed", fact: f2,
        cites: f2.keys.slice(0, 20).map((k) => ({ id: "F2", key: k, line: 1 })),
        check: { status: "computed", note: "Counted per resident from the same set of records." } });
    }
  }
  // Quote each record's first two lines: what, where and when, then what happened.
  const evidence = [];
  for (const d of listed) {
    const pair = d.lines.slice(0, 2).map((text, j) => ({ id: `E${evidence.length + j + 1}`, doc: d, line: j + 1, text }));
    evidence.push(...pair);
    claims.push({ text: pair.map((e) => e.text).join(" "), cites: pair.map(cite), kind: "quoted", doc: d });
  }
  return {
    mode: "counted",
    lead: n ? (n > listed.length ? `Showing the ${listed.length} most recent of ${n}.` : "") : "",
    claims, facts, evidence,
  };
}

// ── Trend questions: one value per record, read from a fixed line format ─────
export function trendAnswer(engine, p, points) {
  const spec = p.measureSpec;
  const name = engine.residents[p.residents[0]][1];
  const when = p.dates ? ` in ${p.dates.label}` : "";
  if (!points.length) {
    const f = { id: "F1", text: `No ${spec.label} readings are recorded for ${name}${when}.`, keys: [] };
    return { mode: "trend", lead: "", facts: [f], evidence: [], points,
      claims: [{ text: f.text, kind: "computed", fact: f, cites: [],
        check: { status: "computed", note: `Searched every ${spec.types.map((t) => engine.label(t)).join(" and ")} for ${name} for a line such as "${spec.label} …". None exist, so there is nothing to report.` } }] };
  }
  const fmt = (v) => `${v}${spec.unit === "/10" ? "/10" : " " + spec.unit}`;
  const first = points[0], last = points[points.length - 1];
  let min = points[0], max = points[0];
  for (const pt of points) { if (pt.value < min.value) min = pt; if (pt.value > max.value) max = pt; }
  const window = (from, to) => points.filter((pt) => pt.date >= from && pt.date <= to);
  const shift = (iso, n) => { const d = new Date(iso + "T00:00:00Z"); d.setUTCDate(d.getUTCDate() + n); return d.toISOString().slice(0, 10); };
  const early = window(first.date, shift(first.date, 27));
  const late = window(shift(last.date, -27), last.date);
  const mean = (a) => a.reduce((s, x) => s + x.value, 0) / a.length;
  const round = (v) => Math.round(v * 10) / 10;

  const ev = [first, last, min, max].map((pt, i) => ({ id: `E${i + 1}`, doc: pt.doc, line: pt.line, text: pt.text }));
  const facts = [
    { id: "F1", text: `${points.length} ${spec.label} readings for ${name} between ${longDate(first.date)} and ${longDate(last.date)}.`, keys: points.map((x) => x.doc.key) },
    { id: "F2", text: `Average of the first four weeks of readings: ${round(mean(early))}; average of the last four weeks: ${round(mean(late))}; change ${round(mean(late) - mean(early)) >= 0 ? "+" : ""}${round(mean(late) - mean(early))}.`, keys: [...early, ...late].map((x) => x.doc.key) },
  ];
  const claims = [
    { text: `The records hold ${facts[0].text}`, kind: "computed", fact: facts[0],
      cites: points.slice(0, 30).map((x) => ({ id: "F1", key: x.doc.key, line: x.line })),
      check: { status: "computed", note: `Read one value from each record, from the line "${spec.label} …".` } },
    { text: `${facts[1].text.replace(/\.$/, "")} (${early.length} and ${late.length} readings).`, kind: "computed", fact: facts[1],
      cites: [...early.slice(0, 10), ...late.slice(-10)].map((x) => ({ id: "F2", key: x.doc.key, line: x.line })),
      check: { status: "computed", note: "Mean of the readings dated within 28 days of the first and of the last reading." } },
    { text: `First reading: ${fmt(first.value)} on ${longDate(first.date)}; most recent: ${fmt(last.value)} on ${longDate(last.date)}.`, kind: "derived", cites: [cite(ev[0]), cite(ev[1])] },
    { text: `Lowest ${fmt(min.value)} on ${longDate(min.date)}; highest ${fmt(max.value)} on ${longDate(max.date)}.`, kind: "derived", cites: [cite(ev[2]), cite(ev[3])] },
  ];
  return { mode: "trend", lead: "", claims, facts, evidence: ev, points };
}

// ── Structured questions: a table read straight from the profiles ───────────
export function tableAnswer(engine, p, table, facts) {
  const n = table.rows.length;
  const total = engine.residents.length;
  const filterNote = table.filterTerms.length ? ` whose records mention ${table.filterTerms.join(" and ")}` : "";
  const f1 = { id: "F1", text: `${n} of ${total} residents${filterNote}.`, keys: table.rows.map((r) => r.cells[0].cites[0].key) };
  const allFacts = [f1, ...facts.map((f, i) => ({ id: `F${i + 2}`, text: f.text,
    keys: (f.rows || table.rows).map((r) => r.cells[0].cites[0].key) }))];
  const claims = allFacts.map((f) => ({ text: f.id === "F1" ? `The table lists ${f.text}` : f.text, kind: "computed", fact: f,
    cites: f.keys.slice(0, 30).map((k) => ({ id: f.id, key: k, line: 1 })),
    check: { status: "computed", note: f.id === "F1"
      ? "Read from every resident profile, not just the top search results. Select any cell to open the line it came from."
      : "Counted over the rows of this table." } }));
  // Evidence for the model: the record line behind every cell.
  const evidence = [];
  const seen = new Set();
  for (const r of table.rows) for (const c of r.cells.slice(1)) for (const ci of c.cites) {
    const id = `${ci.key}#${ci.line}`;
    if (seen.has(id) || evidence.length >= 160) continue;
    seen.add(id);
    const doc = engine.byKey.get(ci.key);
    evidence.push({ id: `E${evidence.length + 1}`, doc, line: ci.line, text: doc.lines[ci.line - 1] });
  }
  return { mode: "table", lead: "", table, claims, facts: allFacts, evidence,
    promptNote: `The user is already shown a table of all ${n} matching residents, built directly from these records. Do not repeat the table or list every resident. In 1 to 3 sentences, answer the question or point out the main patterns, citing the calculated facts (F…) or evidence (E…).` };
}

// ── Answer with Gemini ───────────────────────────────────────────────────────
export const SYSTEM_PROMPT = `You answer questions about a care home's records.
Use ONLY the numbered evidence provided. Never use outside knowledge and never guess.
Every sentence must cite at least one evidence ID it is based on, e.g. ["E2","E5"] or ["F1"].
Copy names, dates and numbers exactly as they appear in the evidence.
If the evidence does not answer the question, return one sentence saying what is missing, with "insufficient": true.
Write in plain British English, at most 6 sentences.
If the user asks for a table, or the answer is several items sharing the same attributes, also return a table;
every table row must cite the evidence its cells come from. Keep the sentences short when a table is given.
Reply with JSON only:
{"answer":[{"sentence":"...","cites":["E1"]}],
 "table":{"columns":["..."],"rows":[{"cells":["..."],"cites":["E1"]}]},
 "insufficient":false}
Omit "table" when it is not needed.`;

export function buildPrompt(engine, question, evidence, facts, note = "") {
  const lines = [];
  if (note) lines.push(note, "");
  if (facts.length) {
    lines.push("Calculated facts (computed by the system from the full set of matching records):");
    for (const f of facts) lines.push(`[${f.id}] ${f.text}`);
    lines.push("");
  }
  lines.push("Evidence lines from the records:");
  for (const e of evidence) lines.push(`[${e.id}] (${sourceHeader(engine, e.doc)}, line ${e.line}) ${e.text}`);
  lines.push("", `Question: ${question}`);
  return lines.join("\n");
}

export async function callGemini({ prompt, key, model, proxyUrl, signal }) {
  const body = {
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: { temperature: 0.1, maxOutputTokens: 8192, responseMimeType: "application/json",
      ...(/^gemini-2\.5-flash/.test(model) ? { thinkingConfig: { thinkingBudget: 0 } } : {}) },
  };
  const url = proxyUrl || `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  const res = await fetch(url, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json", ...(proxyUrl ? {} : { "x-goog-api-key": key }) },
    body: JSON.stringify(proxyUrl ? { model, body } : body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.error?.message || `HTTP ${res.status}`;
    const err = new Error(res.status === 429 ? "the free Gemini limit was reached; try again in a minute"
      : `Gemini returned ${res.status}: ${msg}`);
    err.status = res.status;
    throw err;
  }
  const text = (data.candidates?.[0]?.content?.parts || []).map((x) => x.text || "").join("");
  return { text, usage: data.usageMetadata || null, model: data.modelVersion || model };
}

export function parseModelAnswer(text) {
  const clean = text.replace(/```json|```/g, "").trim();
  const start = clean.indexOf("{"), end = clean.lastIndexOf("}");
  const obj = JSON.parse(start >= 0 ? clean.slice(start, end + 1) : clean);
  const answer = Array.isArray(obj.answer) ? obj.answer : [];
  const t = obj.table && Array.isArray(obj.table.rows) && Array.isArray(obj.table.columns) ? obj.table : null;
  return { insufficient: !!obj.insufficient,
    sentences: answer.map((a) => ({ text: String(a.sentence || "").trim(), cites: (a.cites || []).map(String) })).filter((a) => a.text),
    table: t ? { columns: t.columns.map(String), rows: t.rows.map((r) => ({ cells: (r.cells || []).map((c) => String(c ?? "")), cites: (r.cites || []).map(String) })) } : null };
}

export function generatedAnswer(engine, parsed, evidence, facts) {
  const byId = new Map(evidence.map((e) => [e.id, e]));
  const factById = new Map(facts.map((f) => [f.id, f]));
  const claims = parsed.sentences.map((s) => {
    const cites = [], unknown = [];
    for (const id of s.cites) {
      const e = byId.get(id);
      if (e) cites.push(cite(e));
      else if (factById.has(id)) cites.push({ id, key: factById.get(id).keys[0], line: 1, fact: true });
      else unknown.push(id);
    }
    const claim = { text: s.text, cites, kind: "generated", unknownCites: unknown };
    claim.check = verifyClaim(engine, claim, byId, factById, parsed.insufficient);
    return claim;
  });
  let table = null;
  if (parsed.table) {
    table = { columns: parsed.table.columns, generated: true, rows: parsed.table.rows.map((r) => {
      const cites = [], unknown = [];
      for (const id of r.cites) {
        const e = byId.get(id);
        if (e) cites.push(cite(e));
        else if (factById.has(id)) cites.push({ id, key: factById.get(id).keys[0], line: 1, fact: true });
        else unknown.push(id);
      }
      const row = { text: r.cells.join(" "), cites, unknownCites: unknown };
      return { cells: r.cells.map((text) => ({ text, cites })), cites, check: verifyClaim(engine, row, byId, factById, false) };
    }) };
  }
  return { mode: "generated", lead: "", claims, facts, evidence, table, insufficient: parsed.insufficient };
}

// ── Verification ─────────────────────────────────────────────────────────────
// Words that describe the act of recording rather than the content.
const META_WORDS = new Set(["record", "records", "recorded", "resident", "according", "shows",
  "show", "noted", "note", "notes", "documented", "evidence", "states", "stated", "also",
  "however", "which", "while", "after", "before", "during", "been", "being", "would", "could",
  "there", "these", "those", "any", "all", "one", "time", "times", "between"]);

const numbersIn = (s) => (s.match(/\d+(?:\.\d+)?/g) || []).map(Number);

export function verifyClaim(engine, claim, byId, factById, insufficient) {
  if (claim.unknownCites?.length && !claim.cites.length) {
    return { status: "uncited", note: `Cites ${claim.unknownCites.join(", ")}, which ${claim.unknownCites.length === 1 ? "is" : "are"} not in the evidence.` };
  }
  if (!claim.cites.length) {
    return insufficient ? { status: "abstained", note: "The model said the evidence does not answer the question." }
      : { status: "uncited", note: "This sentence cites no evidence." };
  }
  const sources = claim.cites.map((c) => {
    if (c.fact) return factById.get(c.id).text;
    const e = byId.get(c.id);
    return `${sourceHeader(engine, e.doc)} ${e.text}`;
  }).join(" ");
  const srcTokens = new Set(tokenize(sources).map(stem));
  const claimTokens = [...new Set(tokenize(claim.text).filter((t) => !META_WORDS.has(t)))];
  const matched = claimTokens.filter((t) => srcTokens.has(stem(t)));
  const missing = claimTokens.filter((t) => !srcTokens.has(stem(t)));
  const coverage = claimTokens.length ? matched.length / claimTokens.length : 1;
  const srcNums = new Set(numbersIn(sources));
  const badNums = numbersIn(claim.text).filter((n) => !srcNums.has(n));
  let status;
  if (badNums.length) status = "number";
  else if (coverage >= 0.75) status = "supported";
  else if (coverage >= 0.45) status = "partial";
  else status = "weak";
  return { status, coverage, matched, missing, badNums,
    note: badNums.length ? `${badNums.join(", ")} does not appear in the cited ${claim.cites.length === 1 ? "line" : "lines"}.`
      : `${matched.length} of ${claimTokens.length} content words found in the cited ${claim.cites.length === 1 ? "line" : "lines"}.` };
}
