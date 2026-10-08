// Run with:  node --test test/
// Needs data/corpus.json and test/reference.json from
//   python build_rag_demo.py --reference
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Engine } from "../js/retrieve.js";
import { tokenize } from "../js/text.js";
import { parseQuestion } from "../js/query.js";
import { quotedAnswer, countAnswer, trendAnswer, generatedAnswer, verifyClaim } from "../js/answer.js";

const corpus = JSON.parse(readFileSync(new URL("../data/corpus.json", import.meta.url)));
const reference = JSON.parse(readFileSync(new URL("./reference.json", import.meta.url)));
const engine = new Engine(corpus);

test("JavaScript BM25 ranks exactly like the Python BM25 the app uses", () => {
  let slots = 0;
  for (const { query, top } of reference.queries) {
    const scores = [...engine.bm25.scores(tokenize(query))]
      .sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, top.length);
    top.forEach(([key, s], i) => {
      assert.equal(engine.docs[scores[i][0]].key, key, `${query}: rank ${i + 1}`);
      assert.ok(Math.abs(scores[i][1] - s) < 1e-4 * Math.max(1, s), `${query}: score ${i + 1}`);
      slots++;
    });
  }
  assert.ok(slots >= 300);
});

test("every citation points at a real record and line", () => {
  for (const q of ["Did Margaret fall in the lounge?", "Who refused medication?", "pressure area redness"]) {
    const p = parseQuestion(q, corpus);
    const r = engine.search(p);
    assert.ok(r.evidence.length > 0, q);
    for (const e of r.evidence) {
      const d = engine.byKey.get(e.doc.key);
      assert.equal(d.lines[e.line - 1], e.text);
    }
  }
});

test("question understanding: resident, date window and intent", () => {
  const p = parseQuestion("How many falls did Ethel have in the last 6 months?", corpus);
  assert.deepEqual(p.residents, [2]);
  assert.equal(p.intent, "count");
  assert.equal(p.incidentType, "Fall");
  assert.equal(p.dates.to, corpus.meta.date_to);
  assert.equal(p.dates.from, "2026-02-21");
  assert.equal(parseQuestion("What happened in March 2026?", corpus).dates.to, "2026-03-31");
});

test("count answers match a direct count of the records", () => {
  const p = parseQuestion("How many falls did Arthur have?", corpus);
  const c = engine.count(p);
  const direct = corpus.docs.filter((d) => d[1] === "incident" && d[4] === "Fall" && d[2] === 3).length;
  assert.equal(c.docs.length, direct);
  const a = countAnswer(engine, p, c);
  assert.match(a.claims[0].text, new RegExp(`\\b${direct} fall incident`));
});

test("trend answers cite the lines their numbers come from", () => {
  const p = parseQuestion("Is Ethel's fluid intake dropping?", corpus);
  assert.equal(p.intent, "trend");
  const a = trendAnswer(engine, p, engine.series(p));
  const firstLast = a.claims[2];
  for (const c of firstLast.cites) {
    const line = engine.byKey.get(c.key).lines[c.line - 1];
    assert.match(line, /^Fluid intake \d+ ml/);
  }
  const none = trendAnswer(engine, parseQuestion("Is Ethel losing weight?", corpus), []);
  assert.match(none.claims[0].text, /No weight readings/);
});

test("the verifier separates supported, unsupported and invented claims", () => {
  const p = parseQuestion("Did Margaret fall in the lounge?", corpus);
  const r = engine.search(p);
  const e = r.evidence[0];
  const byId = new Map(r.evidence.map((x) => [x.id, x]));
  const facts = new Map();
  const ok = verifyClaim(engine, { text: e.text, cites: [{ id: e.id, key: e.doc.key, line: e.line }] }, byId, facts);
  assert.equal(ok.status, "supported");
  const wrongNum = verifyClaim(engine, { text: e.text.replace(/\d+/, "97"), cites: [{ id: e.id, key: e.doc.key, line: e.line }] }, byId, facts);
  if (/\d/.test(e.text)) assert.equal(wrongNum.status, "number");
  const unrelated = verifyClaim(engine, { text: "She was taken to hospital by ambulance with a fractured femur.", cites: [{ id: e.id, key: e.doc.key, line: e.line }] }, byId, facts);
  assert.ok(["weak", "partial"].includes(unrelated.status), unrelated.status);
  const g = generatedAnswer(engine, { insufficient: false, sentences: [{ text: "Made up.", cites: ["E99"] }] }, r.evidence, []);
  assert.equal(g.claims[0].check.status, "uncited");
});

test("table questions read every profile, and every cell cites its exact line", async () => {
  const { buildRoster } = await import("../js/roster.js");
  const q = "tell me names of the residents , theri primary diseases , any other diseaes  in a tabular form";
  const p = parseQuestion(q, corpus);
  assert.equal(p.intent, "roster");
  const t = buildRoster(engine, p.roster, p.residents);
  assert.equal(t.rows.length, corpus.residents.length);
  assert.deepEqual(t.columns, ["Resident", "Primary diagnosis", "Other diagnoses"]);
  assert.deepEqual(t.ignored.sort(), ["diseaes", "theri"]);
  for (const r of t.rows) for (const c of r.cells.slice(1)) {
    const line = engine.byKey.get(c.cites[0].key).lines[c.cites[0].line - 1];
    assert.ok(line.includes(c.text), `${c.text} not in ${line}`);
  }
  const dem = buildRoster(engine, parseQuestion("Which residents have dementia?", corpus).roster, []);
  const direct = corpus.docs.filter((d) => d[1] === "profile" && /dementia/i.test(d[5].slice(1).join(" "))).length;
  assert.equal(dem.rows.length, direct);
  assert.equal(parseQuestion("Was the GP called about Harold?", corpus).intent, "general");
  assert.equal(parseQuestion("How many falls did Edith have?", corpus).intent, "count");
});
