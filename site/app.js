import { CONFIG } from "./config.js";
import { Engine } from "./js/retrieve.js";
import { parseQuestion } from "./js/query.js";
import { escapeHtml as esc, longDate, stem, tokenize } from "./js/text.js";
import {
  quotedAnswer, countAnswer, trendAnswer, buildPrompt, callGemini, parseModelAnswer,
  generatedAnswer, verifyClaim, sourceHeader, SYSTEM_PROMPT,
} from "./js/answer.js";

const $ = (id) => document.getElementById(id);
const EXAMPLES = [
  "How many falls did Edith have in the last 6 months?",
  "Who fell the most?",
  "What happened when Arthur fell?",
  "Is Doris's fluid intake dropping?",
  "What medications is Joan taking?",
  "Is Ethel losing weight?",
  "Has anyone refused their medication?",
  "What are Margaret's allergies?",
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

// Try each key, and each model, before giving up. A 429 (quota) or 403 moves
// to the next key; a 404 (model unavailable) moves to the next model.
async function generate(prompt) {
  if (!KEYS.length) {
    return callGemini({ prompt, model: CONFIG.MODELS[modelIndex], proxyUrl: CONFIG.PROXY_URL });
  }
  let lastErr;
  for (let m = modelIndex; m < CONFIG.MODELS.length; m++) {
    for (let k = 0; k < KEYS.length; k++) {
      const i = (keyIndex + k) % KEYS.length;
      try {
        const out = await callGemini({ prompt, key: KEYS[i], model: CONFIG.MODELS[m] });
        keyIndex = i; modelIndex = m;
        return out;
      } catch (err) {
        lastErr = err;
        if (err.status === 404) break;                       // try the next model
        if (![429, 403, 500, 503].includes(err.status)) throw err;
      }
    }
  }
  throw lastErr;
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
  $("build-info").textContent = `${corpus.meta.n_docs.toLocaleString("en-GB")} records and ${corpus.meta.n_lines.toLocaleString("en-GB")} lines, dated ${longDate(corpus.meta.date_from)} to ${longDate(corpus.meta.date_to)}. Index built in your browser in ${ms} ms. Data built ${corpus.meta.built} from seed ${corpus.meta.seed}.`;
  $("loading").hidden = true;
  $("question").disabled = false;
  $("ask-btn").disabled = false;
  $("ask-btn").textContent = "Ask";
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
  const search = engine.search(p);
  let base, counted = null, points = null;
  if (p.intent === "count") { counted = engine.count(p); base = countAnswer(engine, p, counted); }
  else if (p.intent === "trend") { points = engine.series(p); base = trendAnswer(engine, p, points); }
  else base = quotedAnswer(engine, search);
  const evidence = base.evidence || search.evidence;
  checkDerived(base, evidence);
  current = { p, search, counted, points, base, evidence, mode: mode === "gemini" && hasGemini() ? "gemini" : "quoted", retrievalMs: performance.now() - t0, gemini: null };

  $("workspace").hidden = false;
  if (mode === "gemini" && hasGemini()) {
    current.gemini = { state: "running", prompt: buildPrompt(engine, question, evidence, base.facts) };
    render();
    await runGemini();
  } else {
    render();
  }
  const first = (current.answer || base).claims.find((c) => c.cites.length);
  if (first) openRecord(first.cites[0].key, first.cites.map((c) => c.line).filter((l, i) => first.cites[i].key === first.cites[0].key), null, false);
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
function citeButtons(cites, claim) {
  const shown = cites.slice(0, 3);
  const extra = cites.length - shown.length;
  const btn = (c) => `<button class="cite" type="button" data-key="${esc(c.key)}" data-line="${c.line}" data-claim="${claim}" aria-pressed="false" title="Open ${esc(c.key)} at line ${c.line}"><span class="eid">${esc(c.id)}</span>${esc(c.key)}, line ${c.line}</button>`;
  return shown.map(btn).join("")
    + (extra > 0 ? `<button class="linkish more-cites" type="button" data-more="${claim}">and ${extra} more record${extra === 1 ? "" : "s"}</button>` : "");
}

function statusBadge(check) {
  if (!check) return "";
  return `<span class="status ${check.status}" tabindex="0" title="${esc(check.note || "")}">${STATUS[check.status] || check.status}</span>`;
}

function claimNote(check) {
  if (!check) return "";
  let html = esc(check.note || "");
  if (check.missing?.length && ["partial", "weak"].includes(check.status)) {
    html += ` Not found: ${check.missing.map((m) => `<span class="miss">${esc(m)}</span>`).join(", ")}.`;
  }
  return `<p class="check-note">${html}</p>`;
}

function renderClaims(answer) {
  let prevKey = null;
  return `<ol class="claims">${answer.claims.map((c, i) => {
    let src = "";
    if (c.kind === "quoted") {
      const d = c.doc;
      if (d.key !== prevKey) src = `<div class="claim-src">${esc(engine.label(d.type))}, ${esc(engine.residentName(d) || "home-wide")}${d.date ? `, ${esc(longDate(d.date))}` : ""}</div>`;
      prevKey = d.key;
    } else prevKey = null;
    const check = c.check || (c.kind === "quoted" ? { status: "quoted", note: "Copied word for word from the record." } : null);
    return `<li class="claim ${c.kind}" data-i="${i}">${src}<p class="claim-text">${esc(c.text)}</p>
      <div class="claim-foot">${statusBadge(check)}${citeButtons(c.cites, i)}</div>
      ${c.kind !== "quoted" ? claimNote(check) : ""}</li>`;
  }).join("")}</ol>`;
}

function sparkline(points) {
  if (!points || points.length < 2) return "";
  const W = 560, H = 120, P = 22;
  const vs = points.map((p) => p.value);
  const lo = Math.min(...vs), hi = Math.max(...vs);
  const t = (d) => Date.parse(d);
  const t0 = t(points[0].date), t1 = t(points[points.length - 1].date) || t0 + 1;
  const x = (p) => P + ((t(p.date) - t0) / Math.max(1, t1 - t0)) * (W - 2 * P);
  const y = (v) => H - P - ((v - lo) / Math.max(1e-9, hi - lo)) * (H - 2 * P);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p).toFixed(1)},${y(p.value).toFixed(1)}`).join("");
  const step = Math.max(1, Math.floor(points.length / 60));
  const dots = points.filter((_, i) => i % step === 0 || i === points.length - 1)
    .map((p) => `<circle class="pt" r="3.5" cx="${x(p).toFixed(1)}" cy="${y(p.value).toFixed(1)}" data-key="${esc(p.doc.key)}" data-line="${p.line}"><title>${esc(p.doc.key)}, line ${p.line}: ${esc(p.text)} (${p.date})</title></circle>`).join("");
  return `<figure class="spark"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Readings over time, ${lo} to ${hi}">
    <path class="line" d="${path}"/>${dots}
    <text x="${P}" y="${H - 4}">${esc(longDate(points[0].date))}</text>
    <text x="${W - P}" y="${H - 4}" text-anchor="end">${esc(longDate(points[points.length - 1].date))}</text>
    <text x="2" y="${y(hi) + 4}">${hi}</text><text x="2" y="${y(lo) + 4}">${lo}</text></svg>
    <figcaption class="small">Each point is one record. Select a point to open it.</figcaption></figure>`;
}

function render() {
  const { p, base, gemini } = current;
  const answer = current.answer && current.mode === "gemini" ? current.answer : base;
  let html = `<div class="answer-head"><h3 class="answer-q">${esc(p.question)}</h3>
    <span class="answer-meta">${current.mode === "gemini" ? `${gemini?.state === "done" ? `Written by ${esc(gemini.model)}, checked against the records` : "AI answer"}` : "Quoted from the records, no AI"}</span></div>`;

  if (current.mode === "gemini") {
    if (gemini.state === "running") {
      html += `<p class="lead">Writing with Gemini from ${current.evidence.length} evidence lines${base.facts.length ? ` and ${base.facts.length} calculated fact${base.facts.length > 1 ? "s" : ""}` : ""}…</p>`;
    } else if (gemini.state === "error") {
      html += `<div class="error">The AI answer is unavailable right now (${esc(gemini.error)}). Here are the matching lines from the records instead.</div>`;
      html += renderBase(base);
    } else {
      const counts = {};
      for (const c of answer.claims) counts[c.check.status] = (counts[c.check.status] || 0) + 1;
      html += `<div class="summary-strip">${Object.entries(counts).map(([s, n]) => `<span class="status ${s}">${n} ${STATUS[s].toLowerCase()}</span>`).join("")}</div>`;
      html += renderClaims(answer);
      if (current.points) html += sparkline(current.points);
    }
  } else {
    html += renderBase(base);
  }
  $("answer").innerHTML = html;
  renderPipeline();
}

function renderBase(base) {
  let html = base.lead ? `<p class="lead">${esc(base.lead)}</p>` : "";
  html += renderClaims(base);
  if (current.points) html += sparkline(current.points);
  return html;
}

function renderPipeline() {
  const { p, search, counted, points, base, gemini } = current;
  const steps = [];
  const resNames = p.residents.map((i) => corpus.residents[i][1]);
  const chips = [
    resNames.length ? `<span class="chip"><b>Resident</b> ${esc(resNames.join(", "))}</span>` : `<span class="chip"><b>Resident</b> any</span>`,
    p.dates ? `<span class="chip"><b>Period</b> ${esc(p.dates.label)}: ${esc(longDate(p.dates.from))} to ${esc(longDate(p.dates.to))}</span>` : `<span class="chip"><b>Period</b> all dates</span>`,
    p.hardTypes.length ? `<span class="chip"><b>Only</b> ${esc(p.hardTypes.map((t) => engine.label(t)).join(", "))}</span>` : "",
    p.softTypes.length ? `<span class="chip"><b>Preferred</b> ${esc(p.softTypes.map((t) => engine.label(t)).join(", "))}</span>` : "",
    p.incidentType ? `<span class="chip"><b>Incident type</b> ${esc(p.incidentType)}</span>` : "",
    `<span class="chip"><b>Question type</b> ${{ count: "count records", trend: "measurement over time", general: "find relevant lines" }[p.intent]}</span>`,
  ].join("");
  const terms = `<p>Search terms: ${p.terms.length ? p.terms.map((t) => `<b>${esc(t)}</b>`).join(", ") : "none (ranked by date)"}${p.expanded.length ? `, expanded with ${p.expanded.map(esc).join(", ")}` : ""}.</p>`;
  steps.push(step("Understand the question", `${resNames.length || p.dates || p.hardTypes.length ? "filters found" : "no filters"}`,
    `<div class="chips">${chips}</div>${terms}${p.dates ? `<p class="small">Dates are measured back from the last day in the dataset, ${esc(longDate(corpus.meta.date_to))}.</p>` : ""}`));

  steps.push(step("Filter by metadata", `${corpus.meta.n_docs.toLocaleString("en-GB")} → ${search.allowedCount.toLocaleString("en-GB")} records`,
    `<p>Records outside the resident, period and type filters are removed before searching. Profiles, care plans and prescriptions describe a standing state, so the period filter does not remove them.</p>`));

  if (p.intent === "count") {
    steps.push(step("Count matching records", `${counted.docs.length} records`,
      `<p>Every record that passes the filters is counted, so the number does not depend on how many results are shown.</p>${counted.byResident.length > 1 ? `<div class="tbl-wrap"><table><thead><tr><th>Resident</th><th class="num">Records</th></tr></thead><tbody>${counted.byResident.slice(0, 8).map(([r, n]) => `<tr><td>${esc(r >= 0 ? corpus.residents[r][1] : "home-wide")}</td><td class="num">${n}</td></tr>`).join("")}</tbody></table></div>` : ""}`));
  } else if (p.intent === "trend") {
    steps.push(step("Read the measurements", `${points.length} readings`,
      `<p>One value is read from each ${esc(p.measureSpec.types.map((t) => engine.label(t)).join(" or "))} using the line pattern <code>${esc(String(p.measureSpec.line).slice(1, -1))}</code>. Every number in the answer comes from a cited line.</p>`));
  } else {
    steps.push(step("Rank with BM25", `${search.matchedCount.toLocaleString("en-GB")} matched, top ${search.ranked.length} kept in ${search.ms.toFixed(0)} ms`,
      `<p>Okapi BM25 (k1 = 1.5, b = 0.75), the same scorer as the Flask app. Maximal marginal relevance (λ = 0.75) then skips records that repeat earlier picks.</p>
      <div class="tbl-wrap"><table><thead><tr><th>#</th><th>Record</th><th>Type</th><th>Date</th><th class="num">BM25</th><th class="num">Boost</th><th class="num">Overlap</th></tr></thead><tbody>
      ${search.ranked.map((c, i) => { const d = engine.docs[c.idx]; return `<tr><td>${i + 1}</td><td><button class="cite" type="button" data-key="${esc(d.key)}" data-line="1">${esc(d.key)}</button></td><td>${esc(engine.label(d.type))}</td><td>${esc(d.date)}</td><td class="num">${c.bm25.toFixed(2)}</td><td class="num">×${c.boost.toFixed(2)}</td><td class="num">${(c.redundancy || 0).toFixed(2)}</td></tr>`; }).join("")}
      </tbody></table></div>`));
  }

  const ev = current.evidence;
  steps.push(step("Choose evidence lines", `${ev.length} lines${base.facts.length ? `, ${base.facts.length} calculated fact${base.facts.length > 1 ? "s" : ""}` : ""}`,
    `<ul class="ev-list">${base.facts.map((f) => `<li><span class="eid">${f.id}</span>${esc(f.text)} <span class="small">(from ${f.keys.length} records)</span></li>`).join("")}
     ${ev.map((e) => `<li><span class="eid">${e.id}</span><button class="cite" type="button" data-key="${esc(e.doc.key)}" data-line="${e.line}">${esc(e.doc.key)}, line ${e.line}</button> ${highlight(e.text, new Set([...search.terms].map(stem)))}</li>`).join("")}</ul>`));

  if (current.mode === "gemini") {
    const g = gemini;
    const usage = g.usage ? `, ${g.usage.promptTokenCount ?? "?"} tokens in, ${g.usage.candidatesTokenCount ?? "?"} out` : "";
    steps.push(step("Write the answer with Gemini",
      g.state === "running" ? "waiting for the model…" : g.state === "error" ? "failed" : `${(g.ms / 1000).toFixed(1)} s${usage}`,
      `<p>The model sees only the evidence above and must cite an evidence ID for every sentence.</p>
       <details><summary>System instruction</summary><pre class="prompt">${esc(SYSTEM_PROMPT)}</pre></details>
       <details><summary>Prompt sent to the model</summary><pre class="prompt">${esc(g.prompt)}</pre></details>
       ${g.raw ? `<details><summary>Raw model response</summary><pre class="prompt">${esc(g.raw)}</pre></details>` : ""}`));
    if (g.state === "done") {
      steps.push(step("Check each claim", `${current.answer.claims.filter((c) => c.check.status === "supported").length} of ${current.answer.claims.length} supported`,
        `<p>For every sentence, the content words and all numbers are looked up in the lines it cites (plus each record's date, type and resident). Numbers must match exactly. Supported means at least 75% of the words are found; partly supported means at least 45%.</p>`));
    }
  } else {
    steps.push(step("Write the answer", "quoted, no language model",
      `<p>The answer is made of evidence lines copied word for word, so every sentence is in the record by construction. ${p.intent !== "general" ? "Calculated sentences show how they were computed and link to the records used." : ""} Choose <b>AI answer</b> to see Gemini write the answer, checked claim by claim.</p>`));
  }
  $("pipeline").innerHTML = steps.join("");
}

function step(title, summary, body) {
  return `<li><details><summary>${esc(title)} <span class="sum">${esc(summary)}</span></summary><div class="detail">${body}</div></details></li>`;
}

function highlight(text, stems) {
  return text.split(/([A-Za-z0-9]+)/).map((part, i) => {
    if (i % 2 === 0) return esc(part);
    const t = part.toLowerCase();
    return stems.has(stem(t)) && tokenize(t).length ? `<b class="hit">${esc(part)}</b>` : esc(part);
  }).join("");
}

// ── Record viewer ──────────────────────────────────────────────────────────
function openRecord(key, lines, stems, scroll = true) {
  const d = engine.byKey.get(key);
  if (!d) return;
  const cited = new Set(lines);
  const focusLine = lines[0];
  const hit = stems || new Set((current?.search.terms || []).map(stem));
  const extra = Object.entries(d.extra).filter(([k]) => !["ai_generated"].includes(k))
    .map(([k, v]) => `<dt>${esc(k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()))}</dt><dd>${esc(v)}</dd>`).join("");
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
    <div class="rec-actions"><button class="linkish" type="button" id="copy-cite">Copy citation</button><span class="small">${d.lines.length} lines</span></div>`;
  $("copy-cite").onclick = async () => {
    const text = `${d.key}, line${cited.size > 1 ? "s" : ""} ${[...cited].sort((a, b) => a - b).join(", ")} (${sourceHeader(engine, d)})`;
    try { await navigator.clipboard.writeText(text); $("copy-cite").textContent = "Citation copied"; } catch { $("copy-cite").textContent = text; }
  };
  const el = $("viewer").querySelector(`#ln-${focusLine}`);
  if (el) {
    const v = $("viewer");
    v.scrollTop = Math.max(0, el.offsetTop - v.clientHeight / 3);
  }
  if (scroll && matchMedia("(max-width: 960px)").matches) $("viewer").scrollIntoView({ behavior: "smooth", block: "start" });
  document.querySelectorAll(".cite[aria-pressed]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key)));
}

function showEmptyViewer() {
  $("viewer-body").className = "viewer-empty";
  $("viewer-body").innerHTML = "<p>This answer cites no records. Nothing in the data matched, which is itself the answer: the system does not fill gaps.</p>";
}

// ── Events ─────────────────────────────────────────────────────────────────
document.addEventListener("click", (ev) => {
  const c = ev.target.closest(".cite, .pt");
  if (c) {
    const key = c.dataset.key;
    let lines = [Number(c.dataset.line)];
    let stems = null;
    const ci = c.dataset.claim;
    if (ci !== undefined && current) {
      const answer = current.mode === "gemini" && current.answer ? current.answer : current.base;
      const claim = answer.claims[Number(ci)];
      if (claim) {
        const same = claim.cites.filter((x) => x.key === key).map((x) => x.line);
        lines = [Number(c.dataset.line), ...same.filter((l) => l !== Number(c.dataset.line))];
        if (claim.kind === "generated" && claim.check?.matched) stems = new Set(claim.check.matched.map(stem));
        else stems = new Set(tokenize(claim.text).map(stem));
      }
    }
    openRecord(key, lines, stems);
    return;
  }
  const more = ev.target.closest("[data-more]");
  if (more && current) {
    const answer = current.mode === "gemini" && current.answer ? current.answer : current.base;
    const i = Number(more.dataset.more);
    const foot = more.parentElement;
    foot.querySelectorAll(".cite, .more-cites").forEach((n) => n.remove());
    foot.insertAdjacentHTML("beforeend", answer.claims[i].cites.map((x) =>
      `<button class="cite" type="button" data-key="${esc(x.key)}" data-line="${x.line}" data-claim="${i}" aria-pressed="false"><span class="eid">${esc(x.id)}</span>${esc(x.key)}, line ${x.line}</button>`).join(""));
  }
});

$("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("question").value); });
$("examples").innerHTML = EXAMPLES.map((q) => `<button type="button">${esc(q)}</button>`).join("");
$("examples").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || !engine) return;
  $("question").value = b.textContent;
  ask(b.textContent);
});
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => {
  if (r.checked && current) ask(current.p.question);
}));

function updateModeUi() {
  if (hasGemini()) return;
  // No keys in this build: offer quotes only.
  document.querySelector('input[value="quoted"]').checked = true;
  const ai = document.querySelector('input[value="gemini"]');
  ai.disabled = true;
  $("gemini-hint").textContent = "not configured on this copy of the demo";
}

// ── Start ──────────────────────────────────────────────────────────────────
$("repo-link").href = CONFIG.REPO_URL;
Promise.all([load(), loadKeys()]).then(() => {
  updateModeUi();
  const params = new URLSearchParams(location.search);
  const q = params.get("q");
  if (params.get("mode") === "quoted") document.querySelector('input[value="quoted"]').checked = true;
  if (q) { $("question").value = q; ask(q); }
}).catch((err) => {
  $("load-text").textContent = err.message;
  $("ask-btn").textContent = "Unavailable";
});
