// Step 1 of the pipeline: turn a question into metadata filters plus search terms.
// Everything here is rule-based and shown to the user, so it is easy to see
// why a record was or was not considered.
import { tokenize, MONTHS } from "./text.js";
import { parseRoster } from "./roster.js";

const NUMBER_WORDS = { a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6,
  seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12 };

const INCIDENT_TYPES = [
  [/\b(fall|falls|fell|fallen|falling)\b/, "Fall"],
  [/\bchok(e|ed|ing)\b/, "Choking risk"],
  [/\bmedication errors?\b/, "Medication error"],
  [/\bnear miss(es)?\b/, "Near miss"],
  [/\bskin (concern|tear|damage)s?\b/, "Skin concern"],
  [/\bbehaviou?ral\b/, "Behavioural"],
];

// Words that name a record type. "hard" restricts the search to that type;
// otherwise the type is only preferred when ranking.
const TYPE_RULES = [
  { re: /\bhandovers?\b/, type: "handover", hard: true },
  { re: /\bcare ?plans?\b/, type: "care_plan", hard: true },
  { re: /\bwell-?being (assessment|score|review)s?\b/, type: "wellbeing", hard: true },
  { re: /\brisk assessments?\b/, type: "risk", hard: true },
  { re: /\bincidents?\b/, type: "incident", hard: true },
  { re: /\bfamily (communications?|contacts?|calls?|letters?|emails?|updates?)\b|\b(called|phoned|emailed|contacted) the family\b/, type: "family_comm", hard: true },
  { re: /\bwhat (medications?|medicines?|meds|tablets)\b|\b(currently )?(prescribed|taking)\b/, type: "medication", hard: true },
  { re: /\b(medications?|medicines?|meds|tablets?|doses?)\b/, type: "medication", hard: false },
  { re: /\b(allerg\w*|diagnos\w*|dnacpr|resuscitat\w*|next of kin|gp|room|admitted|capacity|key worker|how old|age)\b/, type: "profile", hard: false },
  { re: /\bcare notes?\b|\bdaily notes?\b/, type: "care_note", hard: true },
];

// Small synonym list so "fall" also finds "slipped" and "found on the floor".
const EXPANSIONS = {
  called: ["notified", "informed", "contacted"], phoned: ["notified", "informed", "contacted"],
  rang: ["notified", "informed", "contacted"], told: ["informed", "notified"],
  fall: ["fell", "falls", "slipped", "floor"], falls: ["fell", "fall", "slipped", "floor"],
  fell: ["fall", "slipped", "floor"], agitated: ["agitation", "distressed", "unsettled"],
  agitation: ["agitated", "distressed", "unsettled"], refused: ["refusal", "declined", "refusing"],
  refuse: ["refused", "refusal", "declined"], drinking: ["fluid", "fluids", "intake"],
  hydration: ["fluid", "fluids", "intake"], dehydration: ["fluid", "fluids", "intake"],
  infection: ["uti", "antibiotics"], sleep: ["sleep", "night", "slept"],
  mood: ["mood", "low", "settled"], eating: ["appetite", "meal", "diet"], appetite: ["eating", "meal"],
};

// Words that carry no search meaning once filters have been extracted.
const FILLER = new Set(["how", "many", "much", "what", "when", "who", "which", "why", "any",
  "can", "could", "would", "should", "tell", "me", "about", "show", "list", "give", "all",
  "last", "past", "previous", "recent", "recently", "since", "during", "between", "this",
  "year", "years", "month", "months", "week", "weeks", "day", "days", "today", "yesterday",
  "record", "records", "recorded", "number", "count", "often", "times", "time", "resident",
  "residents", "our", "home", "summary", "summarise", "summarize", "happened", "been",
  "has", "have", "get", "got", "were", "there", "are", "is", "doing", "over", "ago",
  "anyone", "someone", "anybody", "everyone", "signs", "showing", "their", "most", "least",
  "called", "phoned", "rang", "told", "trend", "changed", "change", "losing", "gaining", "lost", "gained", "track", "going",
  ...MONTHS, ...Object.keys(NUMBER_WORDS)]);

const MEASURES = {
  fluid: { re: /\b(fluids?|hydrat\w*|drink\w*|intake)\b/, label: "fluid intake", unit: "ml",
    line: /^Fluid intake (\d+) ml\.?$/, types: ["care_note"] },
  wellbeing: { re: /\bwell-?being\b/, label: "overall wellbeing score", unit: "/10",
    line: /^Overall score (\d+)\/10/, types: ["wellbeing"] },
  weight: { re: /\b(weight|weigh\w*)\b/, label: "weight", unit: "kg",
    line: /^Weight (\d+(?:\.\d+)?) kg/, types: ["care_note"] },
};
const TREND_WORDS = /\b(trend|losing|gaining|lost|gained|chang\w*|increas\w*|decreas\w*|drop\w*|declin\w*|improv\w*|worse\w*|better|over time|track\w*|average|low|high)\b/;

function isoDay(d) { return d.toISOString().slice(0, 10); }
function addDays(iso, n) { const d = new Date(iso + "T00:00:00Z"); d.setUTCDate(d.getUTCDate() + n); return isoDay(d); }
function addMonths(iso, n) { const d = new Date(iso + "T00:00:00Z"); d.setUTCMonth(d.getUTCMonth() + n); return isoDay(d); }
function monthEnd(y, m) { return isoDay(new Date(Date.UTC(y, m, 0))); }

// Relative dates are anchored to the last day in the dataset, not to today,
// so "last month" means the last month of records.
export function parseDates(q, anchor) {
  const s = q.toLowerCase();
  const [ay, am] = anchor.split("-").map(Number);
  let m = s.match(/\b(?:last|past|previous)\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)?\s*(day|week|month|year)s?\b/);
  if (m) {
    const n = m[1] ? (NUMBER_WORDS[m[1]] || Number(m[1])) : 1;
    const unit = m[2];
    const from = unit === "day" ? addDays(anchor, -n + 1) : unit === "week" ? addDays(anchor, -7 * n + 1)
      : unit === "month" ? addDays(addMonths(anchor, -n), 1) : addDays(addMonths(anchor, -12 * n), 1);
    return { from, to: anchor, label: `the last ${n === 1 ? "" : n + " "}${unit}${n === 1 ? "" : "s"}` };
  }
  if (/\bthis month\b/.test(s)) return { from: `${anchor.slice(0, 7)}-01`, to: anchor, label: "this month" };
  if (/\bthis year\b/.test(s)) return { from: `${ay}-01-01`, to: anchor, label: `${ay} so far` };
  if (/\b(yesterday|today)\b/.test(s)) return { from: anchor, to: anchor, label: "the last day of records" };
  if (/\brecent(ly)?\b/.test(s)) return { from: addDays(anchor, -29), to: anchor, label: "the last 30 days" };
  m = s.match(new RegExp(`\\b(since|from|after)?\\s*(${MONTHS.join("|")})\\s*(20\\d\\d)?\\b`));
  if (m && !(m[2] === "may" && !m[3] && !m[1])) {          // "may" alone is usually the verb
    const mi = MONTHS.indexOf(m[2]) + 1;
    let y = m[3] ? Number(m[3]) : (mi <= am ? ay : ay - 1);
    const from = `${y}-${String(mi).padStart(2, "0")}-01`;
    const name = m[2][0].toUpperCase() + m[2].slice(1);
    if (m[1]) return { from, to: anchor, label: `since ${name} ${y}` };
    return { from, to: monthEnd(y, mi), label: `${name} ${y}` };
  }
  m = s.match(/\b(20\d\d)\b/);
  if (m) return { from: `${m[1]}-01-01`, to: `${m[1]}-12-31`, label: m[1] };
  return null;
}

export function parseQuestion(question, corpus) {
  const q = question.trim();
  const s = q.toLowerCase();
  const words = s.match(/[a-z0-9]+/g) || [];
  const wordSet = new Set(words);

  // Residents: first name, surname or ID.
  const residents = [];
  const nameTokens = new Set();
  corpus.residents.forEach(([rid, full, preferred], idx) => {
    const first = preferred.toLowerCase();
    const last = full.split(" ").slice(-1)[0].toLowerCase();
    if (wordSet.has(rid.toLowerCase()) || wordSet.has(first) || wordSet.has(last)) {
      residents.push(idx);
      nameTokens.add(first); nameTokens.add(last); nameTokens.add(rid.toLowerCase());
    }
  });

  const dates = parseDates(q, corpus.meta.date_to);
  const count = /\bhow many\b|\bnumber of\b|\bcount\b|\bhow often\b/.test(s)
    || /\b(who|which residents?)\b.*\b(most|least)\b/.test(s);

  let incidentType = null;
  for (const [re, t] of INCIDENT_TYPES) if (re.test(s)) { incidentType = t; break; }

  const hardTypes = new Set();
  const softTypes = new Set();
  for (const r of TYPE_RULES) {
    if (!r.re.test(s)) continue;
    if (r.type === "medication" && incidentType === "Medication error") continue;
    (r.hard ? hardTypes : softTypes).add(r.type);
  }
  if (incidentType) (count ? hardTypes : softTypes).add("incident");
  if (hardTypes.has("medication") && /\brefus|\bmissed\b|\bgiven\b/.test(s)) hardTypes.delete("medication");

  let measure = null;
  for (const [key, m] of Object.entries(MEASURES)) {
    if (m.re.test(s)) { measure = key; break; }
  }
  const roster = parseRoster(q, nameTokens);
  let intent = "general";
  if (count && (incidentType || hardTypes.size) && !(roster && !incidentType)) intent = "count";
  else if (roster) intent = "roster";
  else if (measure && residents.length === 1 && (TREND_WORDS.test(s) || measure === "weight")) intent = "trend";

  const base = tokenize(q).filter((t) => !FILLER.has(t) && !nameTokens.has(t)
    && !(dates && /^\d+$/.test(t)));
  const expanded = [];
  for (const t of base) for (const e of EXPANSIONS[t] || []) if (!base.includes(e) && !expanded.includes(e)) expanded.push(e);

  return {
    question: q, residents, dates, intent, incidentType, measure, roster,
    measureSpec: measure ? MEASURES[measure] : null,
    hardTypes: [...hardTypes], softTypes: [...softTypes].filter((t) => !hardTypes.has(t)),
    terms: base, expanded,
  };
}
