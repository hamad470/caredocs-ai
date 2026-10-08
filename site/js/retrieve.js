// Steps 2 and 3 of the pipeline: filter by metadata, rank with BM25, keep a
// diverse top-k, then pick the individual lines that carry the evidence.
import { BM25 } from "./bm25.js";
import { tokenize } from "./text.js";

// Records that describe a standing state rather than an event on a day;
// a date window does not exclude them.
const UNDATED = new Set(["profile", "care_plan", "medication"]);

export class Engine {
  constructor(corpus) {
    this.meta = corpus.meta;
    this.residents = corpus.residents;
    this.docs = corpus.docs.map(([key, type, res, date, section, lines, table, row, extra], idx) => ({
      idx, key, type, res, date, section, lines, table, row, extra: extra || {},
    }));
    this.docTokens = this.docs.map((d) => tokenize(d.lines.join(" ")));
    this.bm25 = new BM25(this.docTokens);
    this.byKey = new Map(this.docs.map((d) => [d.key, d]));
  }

  label(type) { return this.meta.labels[type] || type; }
  residentName(d) { return d.res >= 0 ? this.residents[d.res][1] : ""; }

  // Apply the metadata filters from the parsed question.
  filter(p) {
    const res = new Set(p.residents);
    const types = new Set(p.hardTypes);
    const allowed = new Set();
    for (const d of this.docs) {
      if (res.size && !res.has(d.res)) continue;
      if (types.size && !types.has(d.type)) continue;
      if (p.intent === "count" && p.incidentType && d.type === "incident" && d.section !== p.incidentType) continue;
      if (p.dates && !UNDATED.has(d.type) && (d.date < p.dates.from || d.date > p.dates.to)) continue;
      allowed.add(d.idx);
    }
    return allowed;
  }

  search(p, { k = 8, poolSize = 60, lambda = 0.75 } = {}) {
    const t0 = performance.now();
    const allowed = this.filter(p);
    const filtered = allowed.size < this.docs.length;
    const terms = [...p.terms, ...p.expanded];
    const soft = new Set(p.softTypes);
    const raw = terms.length ? this.bm25.scores(terms, filtered ? allowed : null) : new Map();

    let pool;
    if (raw.size) {
      pool = [...raw].map(([idx, bm]) => {
        const d = this.docs[idx];
        let boost = soft.has(d.type) ? 1.5 : 1;
        if (p.incidentType && d.type === "incident" && d.section === p.incidentType) boost *= 1.3;
        return { idx, bm25: bm, boost, score: bm * boost };
      });
    } else {
      // Nothing to match on (e.g. "Tell me about Ethel"): newest records first,
      // with the resident's profile at the top.
      pool = [...allowed].map((idx) => {
        const d = this.docs[idx];
        const score = d.type === "profile" ? 2 : soft.has(d.type) ? 1.5 : 1;
        return { idx, bm25: 0, boost: 1, score, date: d.date };
      });
      pool.sort((a, b) => b.score - a.score || (b.date || "").localeCompare(a.date || ""));
    }
    pool.sort((a, b) => b.score - a.score || (this.docs[b.idx].date || "").localeCompare(this.docs[a.idx].date || ""));
    pool = pool.slice(0, poolSize);

    // Maximal marginal relevance: care notes repeat a lot, so prefer records
    // that add something the earlier picks do not already say.
    const sets = new Map(pool.map((c) => [c.idx, new Set(this.docTokens[c.idx])]));
    const top = pool[0]?.score || 1;
    const picked = [];
    const rest = [...pool];
    while (picked.length < k && rest.length) {
      let best = 0, bestVal = -Infinity;
      rest.forEach((c, i) => {
        let sim = 0;
        for (const pk of picked) sim = Math.max(sim, jaccard(sets.get(c.idx), sets.get(pk.idx)));
        const val = lambda * (c.score / top) - (1 - lambda) * sim;
        if (val > bestVal) { bestVal = val; best = i; }
      });
      const [c] = rest.splice(best, 1);
      c.redundancy = picked.length ? Math.max(...picked.map((pk) => jaccard(sets.get(c.idx), sets.get(pk.idx)))) : 0;
      picked.push(c);
    }

    const evidence = this.selectLines(picked, terms);
    return {
      allowedCount: allowed.size, filtered, matchedCount: raw.size, terms, ranked: picked,
      evidence, ms: performance.now() - t0,
    };
  }

  // Choose up to two lines per record that best match the query terms.
  selectLines(picked, terms, { perDoc = 2, maxLines = 14 } = {}) {
    const out = [];
    const termSet = new Set(terms);
    for (const c of picked) {
      const d = this.docs[c.idx];
      const scored = d.lines.map((text, i) => {
        const toks = tokenize(text);
        return { line: i + 1, text, score: terms.length ? this.bm25.scoreText(terms, toks) : 0,
          matched: [...new Set(toks.filter((t) => termSet.has(t)))] };
      });
      let chosen = scored.filter((l) => l.score > 0).sort((a, b) => b.score - a.score).slice(0, perDoc);
      if (!chosen.length) chosen = scored.slice(0, 1);
      chosen.sort((a, b) => a.line - b.line);
      for (const l of chosen) out.push({ doc: d, ...l });
      if (out.length >= maxLines) break;
    }
    return out.slice(0, maxLines).map((e, i) => ({ id: `E${i + 1}`, ...e }));
  }

  // Count intent: every record that passes the filters, not just the top-k.
  count(p) {
    const allowed = this.filter(p);
    const docs = [...allowed].map((i) => this.docs[i])
      .filter((d) => !UNDATED.has(d.type) || p.hardTypes.includes(d.type))
      .sort((a, b) => (a.date || "").localeCompare(b.date || ""));
    const byResident = new Map();
    for (const d of docs) byResident.set(d.res, (byResident.get(d.res) || 0) + 1);
    return { docs, byResident: [...byResident].sort((a, b) => b[1] - a[1]) };
  }

  // Trend intent: read one measurement from every matching line.
  series(p) {
    const spec = p.measureSpec;
    const allowed = this.filter({ ...p, hardTypes: spec.types });
    const points = [];
    for (const i of allowed) {
      const d = this.docs[i];
      for (let li = 0; li < d.lines.length; li++) {
        const m = d.lines[li].match(spec.line);
        if (m) { points.push({ doc: d, line: li + 1, text: d.lines[li], value: Number(m[1]), date: d.date }); break; }
      }
    }
    points.sort((a, b) => a.date.localeCompare(b.date) || a.doc.idx - b.doc.idx);
    return points;
  }
}

function jaccard(a, b) {
  if (!a || !b || !a.size || !b.size) return 0;
  let inter = 0;
  const [small, big] = a.size < b.size ? [a, b] : [b, a];
  for (const t of small) if (big.has(t)) inter++;
  return inter / (a.size + b.size - inter);
}
