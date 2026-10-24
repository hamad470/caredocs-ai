import { CONFIG } from "./config.js";
import { Engine } from "./js/retrieve.js";
import { parseQuestion } from "./js/query.js";
import { escapeHtml as esc, longDate, stem, tokenize } from "./js/text.js";
import { buildRoster, summariseRoster } from "./js/roster.js";
import {
  quotedAnswer, countAnswer, trendAnswer, tableAnswer, buildPrompt, callGemini, parseModelAnswer,
  generatedAnswer, verifyClaim, sourceHeader, SYSTEM_PROMPT,
} from "./js/answer.js";

const $ = (id) => document.getElementById(id);
const EXAMPLE_GROUPS = [
  ["Find in the records", ["What happened when Arthur fell?", "Has anyone refused their medication?", "Was the GP called about Harold?"]],
  ["Tables", ["List every resident with their primary and other diagnoses", "Which residents have dementia?", "What medications is Joan taking?"]],
  ["Counts", ["How many falls did Edith have in the last 6 months?", "Who fell the most?", "How many residents have diabetes?"]],
  ["Trends", ["Is Doris's fluid intake dropping?", "Is Ethel losing weight?", "How has Vera's wellbeing changed?"]],
];
const STATUS = {
  supported: "Supported", partial: "Partly supported", weak: "Not found in source",
  number: "Number not in source", uncited: "No valid citation", computed: "Calculated",
  derived: "Read from records", quoted: "Exact quote", abstained: "Not enough evidence",
};

let engine = null;
let corpus = null;
let current = null;       // { p, answer, ... } for the answer on screen

// ── Gemini keys (the site owner's, added at build time) ────────────────────
let KEYS = [];
let keyIndex = 0;          // rotates so visitors spread across the keys
let modelIndex = 0;
const hasGemini = () => KEYS.length > 0 || !!CONFIG.PROXY_URL;

async function loadKeys() {
  try { KEYS = (await import("./keys.js")).GEMINI_KEYS || []; } catch { KEYS = []; }
  keyIndex = KEYS.length ? Math.floor(Math.random() * KEYS.length) : 0;
}

// Which model to use is discovered from each key, as the Flask app does, so the
// demo keeps working when Google retires a model name.
const FALLBACK_MODELS = ["gemini-flash-latest", ...CONFIG.MODELS];
const REJECT = /(embedding|aqa|vision|image|audio|tts|imagen|veo|gemma|learnlm|live|computer-use|robotics)/;
const DEMOTE = /(preview|exp|thinking)/;
const modelCache = new Map();          // key -> ordered model names
const badModels = new Set();           // "key|model" pairs that returned 404

function rank(name) {
  const low = name.toLowerCase();
  if (low === "gemini-flash-latest") return [DEMOTE.test(low) ? 1 : 0, 0];
  const version = parseFloat((low.match(/gemini-(\d+(?:\.\d+)?)/) || [])[1] || 0);
  return [DEMOTE.test(low) ? 1 : 0, 1, /flash/.test(low) ? 0 : 1, /lite/.test(low) ? 1 : 0,
    /pro/.test(low) ? 1 : 0, -version, low.length];
}
const byRank = (a, b) => { const ra = rank(a), rb = rank(b);
  for (let i = 0; i < Math.max(ra.length, rb.length); i++) if ((ra[i] ?? 0) !== (rb[i] ?? 0)) return (ra[i] ?? 0) - (rb[i] ?? 0);
  return 0; };

async function modelsFor(key) {
  if (modelCache.has(key)) return modelCache.get(key);
  let names = [];
  try {
    const res = await fetch("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
      { headers: { "x-goog-api-key": key } });
    const data = await res.json();
    names = (data.models || [])
      .filter((m) => (m.supportedGenerationMethods || []).includes("generateContent"))
      .map((m) => String(m.name || "").replace(/^models\//, ""))
      .filter((n) => n.startsWith("gemini") && !REJECT.test(n));
  } catch { /* fall back to the static list */ }
  const merged = [...new Set([...names, ...FALLBACK_MODELS])].sort(byRank).slice(0, 6);
  modelCache.set(key, merged);
  return merged;
}

// Try keys in turn (starting from a random one) and, for each key, the best
// models it offers. 404 = model retired, try the next; 429/5xx = busy, try the
// next model or key; 400/401/403 = key problem, try the next key.
async function generate(prompt) {
  if (!KEYS.length) {
    return callGemini({ prompt, model: "gemini-flash-latest", proxyUrl: CONFIG.PROXY_URL });
  }
  let lastErr, attempts = 0;
  for (let k = 0; k < KEYS.length; k++) {
    const i = (keyIndex + k) % KEYS.length;
    const key = KEYS[i];
    for (const model of await modelsFor(key)) {
      if (badModels.has(`${i}|${model}`) || attempts >= 10) continue;
      attempts++;
      try {
        const out = await callGemini({ prompt, key, model });
        keyIndex = i;
        modelCache.set(key, [model, ...modelCache.get(key).filter((m) => m !== model)]);
        return out;
      } catch (err) {
        lastErr = err;
        if (err.status === 404) { badModels.add(`${i}|${model}`); continue; }
        if ([400, 401, 403].includes(err.status)) break;
      }
    }
  }
  throw lastErr || new Error("no Gemini model answered");
}

// ── Loading ────────────────────────────────────────────────────────────────
async function load() {
  const res = await fetch(CONFIG.DATA_URL);
  if (!res.ok) throw new Error(`Could not download ${CONFIG.DATA_URL} (HTTP ${res.status}). Build it with: python build_rag_demo.py`);
  const reader = res.body.getReader();
  const chunks = [];
  let bytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    bytes += value.length;
    $("load-bar").style.width = `${Math.round(90 * (1 - Math.exp(-bytes / 7e6)))}%`;
    $("load-text").textContent = `Downloading records… ${(bytes / 1e6).toFixed(1)} MB`;
  }
  const buf = new Uint8Array(bytes);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  $("load-text").textContent = "Building the search index…";
  await new Promise((r) => setTimeout(r, 30));
  corpus = JSON.parse(new TextDecoder().decode(buf));
  const t0 = performance.now();
  engine = new Engine(corpus);
  const ms = Math.round(performance.now() - t0);
  $("load-bar").style.width = "100%";
  $("n-docs").textContent = corpus.meta.n_docs.toLocaleString("en-GB");
  $("build-info").textContent = `${corpus.meta.n_docs.toLocaleString("en-GB")} records (${corpus.meta.n_lines.toLocaleString("en-GB")} lines) for ${corpus.residents.length} residents, dated ${longDate(corpus.meta.date_from)} to ${longDate(corpus.meta.date_to)}. Search index built in your browser in ${(ms / 1000).toFixed(1)} seconds. Synthetic data, regenerated from a fixed seed so results are reproducible.`;
  $("loading").hidden = true;
  $("question").disabled = false;
  $("ask-btn").disabled = false;
  $("ask-btn").textContent = "Ask";
  return ms;
}

// ── Running a question ─────────────────────────────────────────────────────
async function ask(question) {
  question = question.trim();
  if (!question || !engine) return;
  const mode = document.querySelector('input[name="mode"]:checked').value;

  const url = new URL(location.href);
  url.searchParams.set("q", question);
  history.replaceState(null, "", url);

  const p = parseQuestion(question, corpus);
  const t0 = performance.now();
  const useAi = mode === "gemini" && hasGemini();
  const search = engine.search(p, useAi ? { k: 12, maxLines: 30 } : {});
  let base, counted = null, points = null, table = null;
  if (p.intent === "roster") { table = buildRoster(engine, p.roster, p.residents); base = tableAnswer(engine, p, table, summariseRoster(table)); }
  else if (p.intent === "count") { counted = engine.count(p); base = countAnswer(engine, p, counted); }
  else if (p.intent === "trend") { points = engine.series(p); base = trendAnswer(engine, p, points); }
  else base = quotedAnswer(engine, search);
  const evidence = base.evidence || search.evidence;
  checkDerived(base, evidence);
  current = { p, search, counted, points, table, base, evidence, mode: mode === "gemini" && hasGemini() ? "gemini" : "quoted", retrievalMs: performance.now() - t0, gemini: null };

  const wasHidden = $("workspace").hidden;
  $("workspace").hidden = false;
  if (wasHidden || $("answer").getBoundingClientRect().top < 0) $("workspace").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  if (mode === "gemini" && hasGemini()) {
    current.gemini = { state: "running", prompt: buildPrompt(engine, question, evidence, base.facts, base.promptNote || "") };
    render();
    await runGemini();
  } else {
    render();
  }
  const firstSrc = current.sources?.values().next().value;
  if (firstSrc) openRecord(firstSrc.key, [...firstSrc.lines].sort((a, b) => a - b), null, false);
  else showEmptyViewer();
}

function checkDerived(answer, evidence) {
  const byId = new Map(evidence.map((e) => [e.id, e]));
  for (const c of answer.claims) {
    if (c.kind !== "derived") continue;
    const v = verifyClaim(engine, c, byId, new Map());
    c.check = v.status === "number" ? v
      : { status: "derived", note: "Every number in this sentence appears in the cited lines." };
  }
}

async function runGemini() {
  const g = current.gemini;
  const t0 = performance.now();
  try {
    const out = await generate(g.prompt);
    g.raw = out.text; g.usage = out.usage; g.model = out.model; g.ms = performance.now() - t0;
    const parsed = parseModelAnswer(out.text);
    current.answer = generatedAnswer(engine, parsed, current.evidence, current.base.facts);
    g.state = "done";
  } catch (err) {
    g.state = "error"; g.error = err.message; g.ms = performance.now() - t0;
  }
  render();
}

// ── Rendering ──────────────────────────────────────────────────────────────
const fmt = (n) => Number(n).toLocaleString("en-GB");
const plural = (n, w, ws = w + "s") => `${fmt(n)} ${n === 1 ? w : ws}`;
const shownAnswer = () => (current.mode === "gemini" && current.answer ? current.answer : current.base);

// Number every cited record in order of first appearance: [1], [2], …
function collectSources() {
  const map = new Map();
  const add = (key, line) => {
    if (!key || !engine.byKey.has(key)) return;
    if (!map.has(key)) map.set(key, { n: map.size + 1, key, lines: new Set() });
    if (line) map.get(key).lines.add(line);
  };
  const A = shownAnswer();
  for (const c of A.claims) if (c.kind !== "computed") for (const x of c.cites) if (!x.fact) add(x.key, x.line);
  for (const r of (A.table?.generated ? A.table.rows : [])) for (const x of r.cites) if (!x.fact) add(x.key, x.line);
  if (A !== current.base) for (const c of current.base.claims) if (c.kind === "derived") for (const x of c.cites) add(x.key, x.line);
  for (const r of current.table?.rows || []) for (const cell of r.cells) for (const x of cell.cites) add(x.key, x.line);
  return map;
}

// One [n] button per cited record, carrying every line it cites.
function refsHtml(cites, set, i) {
  const byKey = new Map();
  for (const x of cites) {
    if (x.fact) { byKey.set(`fact:${x.id}`, { fact: x.id }); continue; }
    if (!byKey.has(x.key)) byKey.set(x.key, []);
    byKey.get(x.key).push(x.line);
  }
  return [...byKey].map(([key, lines]) => {
    if (lines.fact) return `<button class="ref fact" type="button" data-fact="${esc(lines.fact)}" title="Calculated fact ${esc(lines.fact)}">Σ</button>`;
    const s = current.sources.get(key);
    if (!s) return "";
    return `<button class="ref" type="button" data-key="${esc(key)}" data-lines="${lines.join(",")}" data-set="${set}" data-claim="${i}" aria-pressed="false" aria-label="Source ${s.n}: ${esc(key)}, line ${lines.join(", ")}" title="${esc(key)}, line ${lines.join(", ")}">${s.n}</button>`;
  }).join("");
}

function factsBox(facts, set) {
  if (!facts.length) return "";
  return `<ul class="facts" aria-label="Calculated facts">${facts.map((c) => {
    const keys = [...new Set(c.fact?.keys || c.cites.map((x) => x.key))];
    return `<li><span class="sig" aria-hidden="true">Σ</span><span>${esc(c.text)}${keys.length ? ` <button class="basis" type="button" data-basis="${esc(c.fact?.id || "")}" data-set="${set}">from ${plural(keys.length, "record")}</button>` : ""}</span></li>`;
  }).join("")}</ul>`;
}

function proseHtml(claims, set, offset) {
  if (!claims.length) return "";
  return `<p class="prose">${claims.map((c, j) => {
    const i = offset[j];
    const st = c.check?.status || "supported";
    const dot = c.kind === "generated" ? `<span class="dot ${st}" title="${esc(STATUS[st] || st)}: ${esc(c.check?.note || "")}"></span>` : "";
    return `<span class="sent s-${st}" data-set="${set}" data-claim="${i}">${esc(c.text)}</span><span class="refs">${refsHtml(c.cites, set, i)}</span>${dot}`;
  }).join(" ")}</p>`;
}

function excerptsHtml(claims, set, offset) {
  if (!claims.length) return "";
  return `<ol class="excerpts">${claims.map((c, j) => {
    const d = c.doc || engine.byKey.get(c.cites[0]?.key);
    const s = d && current.sources.get(d.key);
    const lines = c.cites.map((x) => x.line);
    return `<li class="excerpt"><button class="srcnum" type="button" data-key="${esc(d.key)}" data-lines="${lines.join(",")}" data-set="${set}" data-claim="${offset[j]}" aria-label="Open source ${s?.n}">${s?.n ?? "?"}</button>
      <q>${esc(c.text)}</q>
      <span class="src">${esc(engine.label(d.type))}, ${esc(engine.residentName(d) || "home-wide")}${d.date ? `, ${esc(longDate(d.date))}` : ""}. ${esc(d.key)}, line ${lines.join(", ")}</span></li>`;
  }).join("")}</ol>`;
}

function cellHtml(c) {
  if (!c.cites.length) return esc(c.text || "–");
  const parts = c.list ? c.text.split("; ") : [c.text];
  const one = (t, x) => `<button class="cell-cite" type="button" data-key="${esc(x.key)}" data-lines="${x.line}" title="${esc(x.key)}, line ${x.line}">${esc(t)}</button>`;
  return parts.length === c.cites.length ? parts.map((t, i) => one(t, c.cites[i])).join("; ") : one(c.text, c.cites[0]);
}

function tableHtml(table) {
  const gen = !!table.generated;
  const head = table.columns.map((c) => `<th>${esc(c)}</th>`).join("") + (gen ? "<th>Check</th><th>Sources</th>" : "");
  const body = table.rows.map((r, ri) => {
    if (!gen) return `<tr>${r.cells.map((c) => `<td>${cellHtml(c)}</td>`).join("")}</tr>`;
    const st = r.check.status;
    return `<tr>${r.cells.map((c) => `<td>${esc(c.text)}</td>`).join("")}<td><span class="dot ${st}"></span>${esc(STATUS[st] || st)}</td><td class="refs">${refsHtml(r.cites, "row", ri)}</td></tr>`;
  }).join("");
  return `<div class="data-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>
    <p class="small">${gen ? "Table written by the model. Each row is checked against the lines it cites." : `${plural(table.rows.length, "row")}. Every cell is read from a record; select a cell to open the line it came from.`}</p>`;
}

function sparkline(points, label) {
  if (!points || points.length < 2) return "";
  const W = 640, H = 150, L = 44, R = 12, T = 12, B = 26;
  const vs = points.map((p) => p.value);
  const lo = Math.min(...vs), hi = Math.max(...vs);
  const t = (d) => Date.parse(d);
  const t0 = t(points[0].date), t1 = t(points[points.length - 1].date) || t0 + 1;
  const x = (p) => L + ((t(p.date) - t0) / Math.max(1, t1 - t0)) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / Math.max(1e-9, hi - lo)) * (H - T - B);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p).toFixed(1)},${y(p.value).toFixed(1)}`).join("");
  const step = Math.max(1, Math.floor(points.length / 70));
  const dots = points.filter((_, i) => i % step === 0 || i === points.length - 1)
    .map((p) => `<circle class="pt" r="3" cx="${x(p).toFixed(1)}" cy="${y(p.value).toFixed(1)}" data-key="${esc(p.doc.key)}" data-lines="${p.line}"><title>${esc(p.text)} ${esc(p.doc.key)}, line ${p.line}, ${p.date}</title></circle>`).join("");
  return `<figure class="spark"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)} over time, from ${lo} to ${hi}">
    <line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>
    <path class="line" d="${path}"/>${dots}
    <text x="${L}" y="${H - 8}">${esc(longDate(points[0].date))}</text>
    <text x="${W - R}" y="${H - 8}" text-anchor="end">${esc(longDate(points[points.length - 1].date))}</text>
    <text x="${L - 6}" y="${y(hi) + 4}" text-anchor="end">${hi}</text><text x="${L - 6}" y="${y(lo) + 4}" text-anchor="end">${lo}</text></svg>
    <figcaption class="small">${esc(label)}: ${plural(points.length, "reading")}. Each point is one record; select a point to open it.</figcaption></figure>`;
}

function split(claims) {
  const out = { computed: [], quoted: [], prose: [], idx: { computed: [], quoted: [], prose: [] } };
  claims.forEach((c, i) => {
    const k = c.kind === "computed" ? "computed" : c.kind === "quoted" ? "quoted" : "prose";
    out[k].push(c); out.idx[k].push(i);
  });
  return out;
}

function verificationSummary() {
  if (current.mode !== "gemini" || current.gemini?.state !== "done") return null;
  const A = current.answer;
  const checks = [...A.claims.map((c) => c.check), ...(A.table?.generated ? A.table.rows : []).map((r) => r.check)];
  const ok = checks.filter((c) => c.status === "supported").length;
  const bad = checks.filter((c) => ["weak", "number", "uncited"].includes(c.status)).length;
  return { ok, bad, total: checks.length, partial: checks.length - ok - bad, cls: bad ? "bad" : ok === checks.length ? "ok" : "warn" };
}

function render() {
  const { p, base, gemini } = current;
  const ai = current.mode === "gemini";
  current.sources = collectSources();
  const v = verificationSummary();
  const tags = [];
  if (ai) tags.push(`<span class="tag">${gemini.state === "done" ? esc(gemini.model) : "Gemini"}</span>`);
  else tags.push(`<span class="tag">Retrieval only, no AI</span>`);
  if (v) tags.push(`<span class="tag ${v.cls}">${v.ok} of ${v.total} claims supported</span>`);
  else if (!ai) tags.push(`<span class="tag ok">Every sentence quoted or calculated</span>`);
  tags.push(`<span class="tag">${plural(current.sources.size, "source")}</span>`);
  tags.push(`<span class="tag">Retrieved in ${Math.max(1, Math.round(current.retrievalMs))} ms</span>`);
  if (ai && gemini.ms) tags.push(`<span class="tag">Generated in ${(gemini.ms / 1000).toFixed(1)} s</span>`);

  let body = "";
  if (ai && gemini.state === "running") {
    body = `<p class="lead">Writing the answer from ${plural(current.evidence.length, "evidence line")}${base.facts.length ? ` and ${plural(base.facts.length, "calculated fact")}` : ""}…</p><div class="skeleton" aria-hidden="true"><span></span><span></span><span></span></div>`;
  } else {
    if (ai && gemini.state === "error") body += `<div class="alert bad" role="alert">AI answer unavailable: ${esc(gemini.error)}. Showing the retrieved evidence instead.</div>`;
    const A = shownAnswer();
    const s = split(A.claims);
    const set = A === base ? "base" : "ai";
    if (A.insufficient) body += `<div class="alert info">The model reported that the evidence does not fully answer this question.</div>`;
    if (set === "ai") {
      body += proseHtml(s.prose, set, s.idx.prose);
      const bs = split(base.claims);
      if (bs.computed.length) body += `<h3 class="small" style="margin:1rem 0 .4rem">Calculated from the records</h3>` + factsBox(bs.computed, "base");
      if (A.table?.generated) body += tableHtml(A.table);
    } else {
      if (base.lead && !s.computed.length) body += `<p class="lead">${esc(base.lead)}</p>`;
      body += factsBox(s.computed, set);
      body += proseHtml(s.prose, set, s.idx.prose);
      if (s.quoted.length) {
        if (s.computed.length) body += `<h3 class="small" style="margin:1rem 0 .5rem">${esc(base.lead || "Records")}</h3>`;
        body += excerptsHtml(s.quoted, set, s.idx.quoted);
      }
    }
    if (current.table) body += tableHtml(current.table);
    if (current.points) body += sparkline(current.points, p.measureSpec?.label ? p.measureSpec.label[0].toUpperCase() + p.measureSpec.label.slice(1) : "Readings");
    if (!A.claims.length && !current.table) body += `<p class="empty">No record matched this question. Try naming a resident, an event such as a fall, or a time period.</p>`;
  }

  $("answer").innerHTML = `
    <header class="ac-head"><h2 class="ac-q" id="answer-q">${esc(p.question)}</h2><div class="ac-meta">${tags.join("")}</div></header>
    <div class="ac-body">${body}</div>
    <div class="ac-actions"><button class="btn-ghost" type="button" id="copy-answer">Copy answer with sources</button><button class="btn-ghost" type="button" id="copy-link">Copy link</button></div>`;
  $("copy-answer").onclick = () => copy(answerAsText(), $("copy-answer"), "Copied");
  $("copy-link").onclick = () => copy(location.href, $("copy-link"), "Link copied");
  renderSources();
  renderVerification();
  renderTrace();
  renderPrompt();
}

function answerAsText() {
  const A = shownAnswer();
  const lines = [`Q: ${current.p.question}`, ""];
  for (const c of A.claims) {
    const refs = [...new Set(c.cites.filter((x) => !x.fact).map((x) => current.sources.get(x.key)?.n).filter(Boolean))];
    lines.push(`${c.text}${refs.length ? " " + refs.map((n) => `[${n}]`).join("") : ""}`);
  }
  lines.push("", "Sources:");
  for (const s of current.sources.values()) {
    const d = engine.byKey.get(s.key);
    lines.push(`[${s.n}] ${s.key}, line ${[...s.lines].sort((a, b) => a - b).join(", ")}: ${engine.label(d.type)}, ${engine.residentName(d) || "home-wide"}${d.date ? `, ${d.date}` : ""}`);
  }
  lines.push("", location.href);
  return lines.join("\n");
}

async function copy(text, btn, done) {
  const old = btn.textContent;
  try { await navigator.clipboard.writeText(text); btn.textContent = done; } catch { btn.textContent = "Copy failed"; }
  setTimeout(() => { btn.textContent = old; }, 1600);
}

// ── Tabs: sources, verification, trace, prompt ─────────────────────────────
function renderSources() {
  const list = [...current.sources.values()];
  $("n-sources").textContent = list.length || "";
  if (!list.length) { $("pane-sources").innerHTML = `<p class="muted">This answer cites no records.</p>`; return; }
  const LIMIT = 12;
  const item = (s, hidden) => {
    const d = engine.byKey.get(s.key);
    const lines = [...s.lines].sort((a, b) => a - b);
    const snip = lines.slice(0, 2).map((l) => `<div><span class="ln">L${l}</span>${esc(d.lines[l - 1] || "")}</div>`).join("");
    return `<li class="source"${hidden ? " hidden" : ""}><button class="srcnum" type="button" data-key="${esc(s.key)}" data-lines="${lines.join(",")}" aria-label="Open source ${s.n}">${s.n}</button>
      <button class="source-title" type="button" data-key="${esc(s.key)}" data-lines="${lines.join(",")}">${esc(engine.label(d.type))}: ${esc(engine.residentName(d) || "home-wide")}${d.date ? `, ${esc(longDate(d.date))}` : ""}</button>
      <div class="source-meta">${esc(s.key)}, line${lines.length > 1 ? "s" : ""} ${lines.join(", ")}. Table <code>${esc(d.table)}</code>, row ${esc(d.row ?? "–")}</div>
      <div class="source-snip">${snip}</div></li>`;
  };
  $("pane-sources").innerHTML = `<ol class="sources">${list.map((s, i) => item(s, i >= LIMIT)).join("")}</ol>
    ${list.length > LIMIT ? `<div class="more-row"><button class="btn-ghost" type="button" id="show-all-sources">Show all ${list.length} sources</button></div>` : ""}`;
  const more = $("show-all-sources");
  if (more) more.onclick = () => { $("pane-sources").querySelectorAll(".source[hidden]").forEach((n) => { n.hidden = false; }); more.remove(); };
}

function renderVerification() {
  const pane = $("pane-verify");
  const ai = current.mode === "gemini";
  if (ai && current.gemini.state === "running") { $("n-verify").textContent = ""; pane.innerHTML = `<p class="muted">Waiting for the model…</p>`; return; }
  const A = shownAnswer();
  const v = verificationSummary();
  $("n-verify").textContent = v ? `${v.ok}/${v.total}` : "";
  const rows = A.claims.map((c, i) => {
    const st = c.check?.status || (c.kind === "quoted" ? "quoted" : "computed");
    const ev = c.check?.matched ? `${c.check.matched.length} of ${c.check.matched.length + c.check.missing.length} words found${c.check.missing.length ? `; missing: <span class="missing">${c.check.missing.map(esc).join(", ")}</span>` : ""}${c.check.badNums?.length ? `; number not in source: <span class="missing">${c.check.badNums.join(", ")}</span>` : ""}` : esc(c.check?.note || (c.kind === "quoted" ? "Copied word for word from the record." : ""));
    return `<tr><td>${i + 1}</td><td class="claim-text">${esc(c.text)}</td><td class="refs">${refsHtml(c.cites, A === current.base ? "base" : "ai", i)}</td><td><span class="dot ${st}"></span>${esc(STATUS[st] || st)}</td><td class="small">${ev}</td></tr>`;
  }).join("");
  const tRows = (A.table?.generated ? A.table.rows : []).map((r, i) => `<tr><td>T${i + 1}</td><td class="claim-text">${esc(r.cells.map((c) => c.text).join(" | "))}</td><td class="refs">${refsHtml(r.cites, "row", i)}</td><td><span class="dot ${r.check.status}"></span>${esc(STATUS[r.check.status])}</td><td class="small">${esc(r.check.note || "")}</td></tr>`).join("");
  pane.innerHTML = `
    ${v ? `<div class="verify-summary"><span class="tag ok">${v.ok} supported</span>${v.partial ? `<span class="tag warn">${v.partial} partly supported</span>` : ""}${v.bad ? `<span class="tag bad">${v.bad} not supported</span>` : ""}</div>`
      : `<p class="muted" style="margin-top:0">${ai ? "The AI answer was not available, so the evidence is shown as exact quotes." : "In retrieval-only mode every sentence is either an exact quote or a calculation over the records, so it is correct by construction."}</p>`}
    <div class="data-table-wrap"><table class="vtable"><thead><tr><th>#</th><th>Claim</th><th>Cites</th><th>Result</th><th>Evidence check</th></tr></thead><tbody>${rows}${tRows}</tbody></table></div>
    <div class="legend"><span><span class="dot supported"></span>Supported: at least 75% of content words found, all numbers match</span><span><span class="dot partial"></span>Partly: 45 to 75%</span><span><span class="dot weak"></span>Not found, wrong number or no valid citation</span></div>
    <p class="small">The check is lexical: it reliably catches invented content and wrong numbers, but cannot tell which resident a sentence is about. Confidentiality is therefore enforced by the filter before retrieval.</p>`;
}

function renderTrace() {
  const { p, search, counted, points, base } = current;
  const resNames = p.residents.map((i) => corpus.residents[i][1]);
  const intent = { count: "Count records", trend: "Measurement over time", general: "Search", roster: "Table from resident profiles" }[p.intent];
  const stages = [[fmt(corpus.meta.n_docs), "records indexed"], [fmt(search.allowedCount), "after filters"]];
  if (p.intent === "roster") stages.push([fmt(current.table.rows.length), "residents in table"]);
  else if (p.intent === "count") stages.push([fmt(counted.docs.length), "records counted"]);
  else if (p.intent === "trend") stages.push([fmt(points.length), "readings found"]);
  else stages.push([fmt(search.matchedCount), "matched by BM25"], [fmt(search.ranked.length), "kept after MMR"]);
  stages.push([fmt(current.evidence.length), "evidence lines"]);

  let html = `<div class="funnel">${stages.map(([n, l]) => `<div><b>${n}</b><span>${l}</span></div>`).join("")}</div>
    <h3>Query understanding</h3>
    <div class="chips">
      <span class="chip"><b>Type</b>${esc(intent)}</span>
      <span class="chip"><b>Resident</b>${esc(resNames.join(", ") || "any")}</span>
      <span class="chip"><b>Period</b>${p.dates ? `${esc(p.dates.label)} (${esc(longDate(p.dates.from))} to ${esc(longDate(p.dates.to))})` : "all dates"}</span>
      ${p.hardTypes.length ? `<span class="chip"><b>Only</b>${esc(p.hardTypes.map((t) => engine.label(t)).join(", "))}</span>` : ""}
      ${p.softTypes.length ? `<span class="chip"><b>Prefer</b>${esc(p.softTypes.map((t) => engine.label(t)).join(", "))}</span>` : ""}
      ${p.incidentType ? `<span class="chip"><b>Incident</b>${esc(p.incidentType)}</span>` : ""}
    </div>
    <p class="small">Search terms: ${p.terms.length ? p.terms.map((t) => `<b>${esc(t)}</b>`).join(", ") : "none"}${p.expanded.length ? `; expanded with ${p.expanded.map(esc).join(", ")}` : ""}.${p.dates ? ` Relative dates are measured back from the last day of data, ${esc(longDate(corpus.meta.date_to))}.` : ""}</p>`;

  if (p.intent === "roster") {
    const t = current.table;
    html += `<h3>Structured read</h3><p class="small">Every resident profile has the same labelled lines, so the table is read from all ${fmt(corpus.residents.length)} profiles rather than from the top search results. Columns: ${esc(t.columns.join(", "))}.${t.filterTerms.length ? ` Rows kept if they mention ${esc(t.filterTerms.join(", "))}.` : ""}${t.ignored.length ? ` Ignored (not in any record): ${esc(t.ignored.join(", "))}.` : ""}</p>`;
  } else if (p.intent === "count") {
    html += `<h3>Count by resident</h3><p class="small">Every record passing the filters is counted, so the total does not depend on how many results are shown.</p>
      ${counted.byResident.length ? `<div class="data-table-wrap"><table><thead><tr><th>Resident</th><th class="num">Records</th></tr></thead><tbody>${counted.byResident.slice(0, 10).map(([r, n]) => `<tr><td>${esc(r >= 0 ? corpus.residents[r][1] : "home-wide")}</td><td class="num">${n}</td></tr>`).join("")}</tbody></table></div>` : ""}`;
  } else if (p.intent === "trend") {
    html += `<h3>Measurement</h3><p class="small">One value is read from each ${esc(p.measureSpec.types.map((t) => engine.label(t)).join(" or "))} matching <code>${esc(String(p.measureSpec.line).slice(1, -1))}</code>.</p>`;
  } else {
    html += `<h3>Ranking</h3><p class="small">Okapi BM25 (k1 = 1.5, b = 0.75), identical to the Python implementation; maximal marginal relevance (λ = 0.75) then skips near-duplicates. ${search.ms.toFixed(0)} ms.</p>
      <div class="data-table-wrap"><table><thead><tr><th>#</th><th>Record</th><th>Type</th><th>Date</th><th class="num">BM25</th><th class="num">Boost</th><th class="num">Overlap</th></tr></thead><tbody>
      ${search.ranked.map((c, i) => { const d = engine.docs[c.idx]; return `<tr><td>${i + 1}</td><td><button class="linkbtn" type="button" data-key="${esc(d.key)}" data-lines="1">${esc(d.key)}</button></td><td>${esc(engine.label(d.type))}</td><td>${esc(d.date)}</td><td class="num">${c.bm25.toFixed(2)}</td><td class="num">×${c.boost.toFixed(2)}</td><td class="num">${(c.redundancy || 0).toFixed(2)}</td></tr>`; }).join("")}
      </tbody></table></div>`;
  }
  const stems = new Set(search.terms.map(stem));
  html += `<h3>Evidence passed to the answer</h3><ul class="ev-list">
    ${base.facts.map((f) => `<li><span class="eid">${esc(f.id)}</span><span class="small">calculated</span><span>${esc(f.text)}</span></li>`).join("")}
    ${current.evidence.slice(0, 60).map((e) => `<li><span class="eid">${esc(e.id)}</span><button class="linkbtn" type="button" data-key="${esc(e.doc.key)}" data-lines="${e.line}">${esc(e.doc.key)}:${e.line}</button><span>${highlight(e.text, stems)}</span></li>`).join("")}</ul>
    ${current.evidence.length > 60 ? `<p class="small">and ${fmt(current.evidence.length - 60)} more lines.</p>` : ""}`;
  $("pane-trace").innerHTML = html;
}

function renderPrompt() {
  const g = current.gemini;
  if (current.mode !== "gemini") {
    $("pane-prompt").innerHTML = `<p class="muted" style="margin-top:0">No language model was used. Switch to <b>AI answer</b> to see the exact prompt sent to Gemini and its raw response.</p>`;
    return;
  }
  const u = g.usage || {};
  $("pane-prompt").innerHTML = `
    <dl class="kv">
      <dt>Model</dt><dd>${esc(g.model || "selecting…")}</dd>
      <dt>Status</dt><dd>${{ running: "Waiting for the model", done: "Completed", error: `Failed: ${esc(g.error || "")}` }[g.state]}</dd>
      ${g.ms ? `<dt>Latency</dt><dd>${(g.ms / 1000).toFixed(2)} s</dd>` : ""}
      ${u.promptTokenCount ? `<dt>Tokens</dt><dd>${fmt(u.promptTokenCount)} in, ${fmt(u.candidatesTokenCount ?? 0)} out</dd>` : ""}
      <dt>Settings</dt><dd>temperature 0.1, JSON output, citations required</dd>
    </dl>
    <details><summary>System instruction</summary><pre class="prompt">${esc(SYSTEM_PROMPT)}</pre></details>
    <details open><summary>Prompt sent to the model</summary><pre class="prompt">${esc(g.prompt)}</pre></details>
    ${g.raw ? `<details><summary>Raw model response</summary><pre class="prompt">${esc(g.raw)}</pre></details>` : ""}`;
}

function highlight(text, stems) {
  return text.split(/([A-Za-z0-9]+)/).map((part, i) => {
    if (i % 2 === 0) return esc(part);
    const t = part.toLowerCase();
    return stems.has(stem(t)) && tokenize(t).length ? `<b class="hit">${esc(part)}</b>` : esc(part);
  }).join("");
}

function selectTab(id) {
  document.querySelectorAll('.tabs [role="tab"]').forEach((t) => {
    const on = t.id === id;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !on;
  });
}

// ── Record viewer ──────────────────────────────────────────────────────────
function openRecord(key, lines, stems, scroll = true) {
  const d = engine.byKey.get(key);
  if (!d) return;
  const cited = new Set(lines);
  const focusLine = lines[0];
  const hit = stems || new Set((current?.search.terms || []).map(stem));
  const n = current?.sources?.get(key)?.n;
  const extra = Object.entries(d.extra).filter(([k]) => k !== "ai_generated")
    .map(([k, v]) => `<dt>${esc(k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()))}</dt><dd>${esc(v)}</dd>`).join("");
  $("viewer-title").textContent = n ? `Source ${n}` : "Source record";
  $("viewer-body").className = "";
  $("viewer-body").innerHTML = `
    <div class="rec-head">
      <p class="rec-type">${esc(engine.label(d.type))}</p>
      <p class="rec-id">${esc(d.key)}</p>
      <dl class="rec-meta">
        <dt>Resident</dt><dd>${esc(engine.residentName(d) || "home-wide")}${d.res >= 0 ? ` (${esc(corpus.residents[d.res][0])})` : ""}</dd>
        ${d.date ? `<dt>Date</dt><dd>${esc(longDate(d.date))}</dd>` : ""}
        <dt>Section</dt><dd>${esc(d.section)}</dd>
        <dt>Stored in</dt><dd><code>carehome.db</code>, table <code>${esc(d.table)}</code>, row ${esc(d.row ?? "–")}</dd>
        ${extra}
      </dl>
    </div>
    <ol class="rec-lines">${d.lines.map((l, i) => `<li id="ln-${i + 1}" class="${cited.has(i + 1) ? "cited" : ""}${i + 1 === focusLine ? " focus" : ""}"><span class="ln">${i + 1}</span><span>${cited.has(i + 1) ? highlight(l, hit) : esc(l)}</span></li>`).join("")}</ol>
    <div class="rec-actions"><button class="btn-ghost" type="button" id="copy-cite">Copy citation</button><span class="small">${plural(d.lines.length, "line")}, ${plural(cited.size, "cited line")}</span></div>`;
  $("copy-cite").onclick = () => copy(`${d.key}, line${cited.size > 1 ? "s" : ""} ${[...cited].sort((a, b) => a - b).join(", ")} (${sourceHeader(engine, d)})`, $("copy-cite"), "Citation copied");
  const v = $("viewer");
  const el = v.querySelector(`#ln-${focusLine}`);
  if (el && !matchMedia("(max-width: 1020px)").matches) v.scrollTop = Math.max(0, el.offsetTop - v.clientHeight / 3);
  if (scroll && matchMedia("(max-width: 1020px)").matches) v.scrollIntoView({ behavior: "smooth", block: "start" });
  document.querySelectorAll(".ref[aria-pressed], .cell-cite").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key)));
}

function showEmptyViewer() {
  $("viewer-title").textContent = "Source record";
  $("viewer-body").className = "viewer-empty";
  $("viewer-body").innerHTML = "<p>This answer cites no records. Nothing in the data matched, and the system does not fill gaps.</p>";
}

// ── Events ─────────────────────────────────────────────────────────────────
function claimFor(el) {
  const set = el.dataset.set, i = el.dataset.claim;
  if (i === undefined || !current) return null;
  if (set === "row") { const r = shownAnswer().table?.rows?.[Number(i)]; return r ? { ...r, kind: "generated" } : null; }
  const list = set === "base" ? current.base.claims : (current.answer || current.base).claims;
  return list[Number(i)] || null;
}

document.addEventListener("click", (ev) => {
  const el = ev.target.closest("[data-key]");
  if (el) {
    const key = el.dataset.key;
    let lines = (el.dataset.lines || "1").split(",").map(Number);
    let stems = null;
    const claim = claimFor(el);
    if (claim) {
      lines = [...new Set([...lines, ...(claim.cites || []).filter((x) => x.key === key).map((x) => x.line)])];
      stems = new Set((claim.check?.matched?.length ? claim.check.matched : tokenize(claim.text || "")).map(stem));
    }
    openRecord(key, lines, stems);
    return;
  }
  const fact = ev.target.closest("[data-fact], [data-basis]");
  if (fact && current) {
    const id = fact.dataset.fact || fact.dataset.basis;
    const f = current.base.facts.find((x) => x.id === id);
    if (f?.keys?.length) openRecord(f.keys[0], [1], null);
    selectTab("tab-trace");
  }
});

document.addEventListener("mouseover", (ev) => {
  const r = ev.target.closest(".ref");
  document.querySelectorAll(".sent.active").forEach((s) => s.classList.remove("active"));
  if (r) document.querySelector(`.sent[data-set="${r.dataset.set}"][data-claim="${r.dataset.claim}"]`)?.classList.add("active");
});

document.querySelector(".tabs").addEventListener("click", (e) => { const t = e.target.closest('[role="tab"]'); if (t) selectTab(t.id); });
document.querySelector(".tabs").addEventListener("keydown", (e) => {
  if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
  const tabs = [...document.querySelectorAll('.tabs [role="tab"]')];
  const i = tabs.indexOf(document.activeElement);
  const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
  selectTab(next.id); next.focus();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement?.tagName !== "INPUT" && !$("question").disabled) { e.preventDefault(); $("question").focus(); }
});

$("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("question").value); });
$("examples").innerHTML = EXAMPLE_GROUPS.map(([title, qs]) => `<div class="ex-group"><h3>${esc(title)}</h3><ul>${qs.map((q) => `<li><button type="button">${esc(q)}</button></li>`).join("")}</ul></div>`).join("");
$("examples").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || !engine) return;
  $("question").value = b.textContent;
  ask(b.textContent);
});
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => {
  updateModeHint();
  if (r.checked && current) ask(current.p.question);
}));

function updateModeHint() {
  const ai = document.querySelector('input[value="gemini"]').checked;
  $("gemini-hint").textContent = !hasGemini() ? "AI answers are not configured on this copy of the demo."
    : ai ? "Gemini writes from the retrieved evidence; each sentence is then verified."
      : "Shows the retrieved evidence and calculations only. No language model is called.";
}

function updateModeUi() {
  if (!hasGemini()) {
    document.querySelector('input[value="quoted"]').checked = true;
    document.querySelector('input[value="gemini"]').disabled = true;
  }
  updateModeHint();
}

// ── Start ──────────────────────────────────────────────────────────────────
$("repo-link").href = CONFIG.REPO_URL;
$("repo-link-2").href = CONFIG.REPO_URL;
Promise.all([load(), loadKeys()]).then(() => {
  updateModeUi();
  $("question").placeholder = "Ask about residents, falls, medications, fluid intake…";
  const params = new URLSearchParams(location.search);
  const q = params.get("q");
  if (params.get("mode") === "quoted") { document.querySelector('input[value="quoted"]').checked = true; updateModeHint(); }
  if (q) { $("question").value = q; ask(q); }
}).catch((err) => {
  $("load-text").textContent = err.message;
  $("ask-btn").textContent = "Unavailable";
});
