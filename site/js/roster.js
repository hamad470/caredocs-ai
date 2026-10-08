// Structured questions about residents ("list every resident with their
// diagnoses", "which residents have dementia", "what are Margaret's allergies").
// Keyword search returns the few best-matching records, so it cannot build a
// 50-row table. Every resident profile has the same labelled lines, so these
// questions are answered by reading those lines directly, the way the Flask
// app's SQL tools do, and every cell cites the line it came from.
import { tokenize, stem } from "./text.js";

export const FIELDS = [
  { id: "primary", label: "Primary diagnosis", line: "Primary diagnosis",
    re: /\bprimary\b|\bmain (diagnos\w*|conditions?|diseases?|illness\w*)/ },
  { id: "secondary", label: "Other diagnoses", line: "Secondary diagnoses", list: true,
    re: /\b(secondary|other|additional|further|more)\s+(diagnos\w*|diseases?|conditions?|illness\w*|problems?|health)|\bcomorbid\w*/ },
  { id: "diagnosis", expands: ["primary", "secondary"],
    re: /\b(diagnos\w*|diseases?|conditions?|illness\w*|medical history|health problems?)\b/ },
  { id: "allergies", label: "Allergies", line: "Allergies", re: /\ballerg\w*/ },
  { id: "meds", label: "Current medications", meds: true,
    re: /\bmedications?\b|\bmedicines?\b|\bmeds\b|\btablets\b|\bprescri\w*|\bdrugs?\b/ },
  { id: "mobility", label: "Mobility", line: "Mobility", re: /\bmobility\b|\bwalk\w*|\bwheelchair\w*|\bhoist\w*/ },
  { id: "diet", label: "Diet texture", line: "Diet texture", re: /\bdiet\w*|\bfood\b|\btexture\b|\bswallow\w*|\bdysphagia\b/ },
  { id: "restrictions", label: "Dietary restrictions", line: "Dietary restrictions", re: /\bdiet\w* restrictions?\b|\bdiabetic diet\b/ },
  { id: "falls", label: "Falls risk", line: "Falls risk", re: /\bfalls? risk\b|\brisk of fall\w*/ },
  { id: "pressure", label: "Pressure sore risk", line: "Pressure sore risk", re: /\bpressure (sore|ulcer|area)s? risk\b|\bpressure risk\b/ },
  { id: "dnacpr", label: "DNACPR status", line: "DNACPR status", re: /\bdnacpr\b|\bresus\w*|\bcpr\b/ },
  { id: "capacity", label: "Mental capacity", line: "Mental capacity", re: /\b(mental )?capacity\b/ },
  { id: "continence", label: "Continence", line: "Continence", re: /\b(in)?continen\w*/ },
  { id: "caretype", label: "Care type", line: "Care type", re: /\bcare types?\b|\btype of care\b|\bnursing or residential\b/ },
  { id: "gp", label: "GP", line: "GP", re: /\bgps?\b|\bdoctors?\b/ },
  { id: "nok", label: "Next of kin", line: "Next of kin", re: /\bnext of kin\b|\bnok\b|\brelatives?\b|\bfamily contacts?\b/ },
  { id: "keyworker", label: "Key worker", line: "Key worker", re: /\bkey ?workers?\b/ },
  { id: "age", label: "Age", head: /\bage (\d+)/, re: /\bages?\b|\bhow old\b|\boldest\b|\byoungest\b/ },
  { id: "room", label: "Room", head: /\broom (Room [^,]+|[^,]+),/, re: /\brooms?\b/ },
  { id: "admitted", label: "Admitted", head: /\badmitted (\d{4}-\d{2}-\d{2})/, re: /\badmi(tted|ssion)\b|\bmoved in\b/ },
];
// Fields whose values are categories worth counting in a summary.
const CATEGORICAL = new Set(["primary", "secondary", "mobility", "diet", "falls", "pressure",
  "dnacpr", "capacity", "continence", "caretype", "allergies"]);

// Phrases that ask for a table or about residents as a group.
const ROSTER_CUE = /\b(list|table|tabular|columns?|all|each|every|everyone|names?|which residents?|who (has|have|is|are)|residents? (with|who|whose|that|on|in|needing|requiring)|anyone (with|on)|what (\w+ )?(is|are)|how many residents)\b|'s\b/;
// Words that point at events rather than standing facts: leave those to search.
const EVENT_WORDS = /\b(called|phoned|happened|incident|fell|fall|falls|visit\w*|refus\w*|missed|given|yesterday|today|last|since|during|night|shift|handover)\b/;

const ROSTER_FILLER = new Set(["name", "names", "table", "tabular", "form", "list", "column", "columns",
  "resident", "residents", "everyone", "each", "every", "all", "their", "known", "give", "show",
  "tell", "what", "which", "who", "has", "have", "with", "whose", "any", "other", "also",
  "primary", "secondary", "additional", "further", "main", "disease", "diseases", "diagnosis",
  "diagnoses", "condition", "conditions", "illness", "illnesses", "status", "details", "info",
  "information", "current", "currently", "please", "format", "many", "how", "number", "count",
  "their", "them", "along", "plus", "including", "include", "home", "care", "people", "patients",
  "me", "tell", "taking", "take", "takes", "need", "needs", "needing", "requiring", "oldest", "youngest"]);

export function parseRoster(q, nameTokens) {
  const s = q.toLowerCase();
  const ids = [];
  const used = [];
  for (const f of FIELDS) {
    const m = s.match(f.re);
    if (!m) continue;
    used.push(m[0]);
    for (const id of f.expands || [f.id]) if (!ids.includes(id)) ids.push(id);
  }
  const wantsCount = /\bhow many residents\b|\bnumber of residents\b/.test(s);
  const groupCue = /\b(which residents?|who (has|have|is|are)|residents? (with|who|whose|that|on|in|needing|requiring)|anyone (with|on)|how many residents)\b/.test(s);
  const usedTokens = new Set(tokenize(used.join(" ")));
  const terms = tokenize(q).filter((t) => !ROSTER_FILLER.has(t) && !usedTokens.has(t) && !nameTokens.has(t));
  const fields = ids.map((id) => FIELDS.find((f) => f.id === id));
  // Ignore the field phrases themselves when looking for event words
  // ("falls risk" is a field, "falls" alone is an event).
  const rest = used.reduce((acc, u) => acc.replace(u, " "), s);
  const event = EVENT_WORDS.test(rest);
  const isRoster = (fields.length > 0 && ROSTER_CUE.test(s) && !event)
    || (groupCue && terms.length > 0 && !event);
  return isRoster ? { fields, terms, wantsCount } : null;
}

// Build the table. `residents` limits the rows; `terms` filters them.
export function buildRoster(engine, roster, residents) {
  const profiles = engine.docs.filter((d) => d.type === "profile");
  const medsByRes = new Map();
  for (const d of engine.docs) {
    if (d.type !== "medication" || d.extra.status === "stopped") continue;
    if (!medsByRes.has(d.res)) medsByRes.set(d.res, []);
    medsByRes.get(d.res).push(d);
  }
  const findLine = (d, prefix) => d.lines.findIndex((l) => l.startsWith(prefix + ":"));
  const value = (line, prefix) => line.slice(prefix.length + 1).trim().replace(/\.$/, "");
  // Only words that occur somewhere in the profiles or prescriptions can
  // filter rows; typos and stray words are reported and ignored.
  if (!engine._rosterVocab) {
    engine._rosterVocab = new Set();
    for (const d of engine.docs) if (d.type === "profile" || d.type === "medication")
      for (const l of d.lines) for (const t of tokenize(l)) engine._rosterVocab.add(stem(t));
  }
  const known = roster.terms.filter((t) => engine._rosterVocab.has(stem(t)));
  const ignored = roster.terms.filter((t) => !engine._rosterVocab.has(stem(t)));
  const noMatch = !known.length && ignored.length && !roster.fields.length;
  const termStems = known.map(stem);

  const rows = [];
  for (const d of profiles) {
    if (noMatch) break;
    if (residents.length && !residents.includes(d.res)) continue;
    const cells = [{ text: engine.residents[d.res][1], cites: [{ key: d.key, line: 1 }] }];
    const searchLines = [];
    for (const f of roster.fields) {
      if (f.meds) {
        const meds = medsByRes.get(d.res) || [];
        cells.push(meds.length
          ? { text: meds.map((m) => m.lines[0].replace(/\.$/, "")).join("; "), cites: meds.map((m) => ({ key: m.key, line: 1 })), list: true }
          : { text: "None recorded", cites: [] });
        meds.forEach((m) => searchLines.push({ key: m.key, line: 1, text: m.lines[0] }));
      } else if (f.head) {
        const m = d.lines[0].match(f.head);
        cells.push({ text: m ? m[1] : "", cites: [{ key: d.key, line: 1 }] });
      } else {
        const i = findLine(d, f.line);
        cells.push(i >= 0 ? { text: value(d.lines[i], f.line), cites: [{ key: d.key, line: i + 1 }] } : { text: "", cites: [] });
        if (i >= 0) searchLines.push({ key: d.key, line: i + 1, text: d.lines[i] });
      }
    }
    // Filter terms: look in the requested fields, or anywhere in the profile.
    let matchCell = null;
    if (termStems.length) {
      const pool = searchLines.length ? searchLines
        : d.lines.slice(1).map((text, i) => ({ key: d.key, line: i + 2, text }));
      const hits = pool.filter((l) => {
        const st = new Set(tokenize(l.text).map(stem));
        return termStems.some((t) => st.has(t));
      });
      const covered = new Set(hits.flatMap((l) => tokenize(l.text).map(stem)));
      if (!termStems.every((t) => covered.has(t))) continue;
      if (!searchLines.length) matchCell = { text: hits.map((h) => h.text.replace(/\.$/, "")).join("; "), cites: hits.map((h) => ({ key: h.key, line: h.line })) };
    }
    if (matchCell) cells.push(matchCell);
    rows.push({ res: d.res, cells });
  }
  const columns = ["Resident", ...roster.fields.map((f) => f.label)];
  if (termStems.length && !roster.fields.length) columns.push("Matching record line");
  return { columns, rows, fields: roster.fields, filterTerms: known, ignored };
}

// Frequencies and ranges to summarise the table (shown as calculated facts).
export function summariseRoster(table) {
  const facts = [];
  table.fields.forEach((f, ci) => {
    const col = ci + 1;
    const cells = table.rows.map((r) => r.cells[col]);
    if (f.id === "age") {
      const ages = cells.map((c) => Number(c.text)).filter((n) => n > 0);
      if (ages.length > 1) {
        const mean = ages.reduce((a, b) => a + b, 0) / ages.length;
        const lo = Math.min(...ages), hi = Math.max(...ages);
        const who = (v) => table.rows.filter((r) => Number(r.cells[col].text) === v).map((r) => r.cells[0].text).join(" and ");
        facts.push({ text: `Age: mean ${mean.toFixed(1)}; oldest ${who(hi)} (${hi}); youngest ${who(lo)} (${lo}); ${ages.length} residents.`, col,
          rows: table.rows.filter((r) => [lo, hi].includes(Number(r.cells[col].text))) });
      }
      return;
    }
    if (!CATEGORICAL.has(f.id) || table.rows.length < 3) return;
    const counts = new Map();
    for (const c of cells) {
      const parts = f.list ? c.text.split(/,\s*/) : [c.text];
      for (const v of parts.map((x) => x.trim()).filter(Boolean)) counts.set(v, (counts.get(v) || 0) + 1);
    }
    const top = [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 6);
    if (top.length) facts.push({ text: `${f.label}, most common: ${top.map(([v, n]) => `${v} (${n})`).join(", ")}.`, col });
  });
  return facts;
}
