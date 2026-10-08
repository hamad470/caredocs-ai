// Optional Gemini proxy for the live demo, so visitors need no key of their own.
// The keys live in Cloudflare as a secret and never reach the browser.
//
//   GEMINI_API_KEYS   secret, comma-separated; on a 429 the next key is tried
//   ALLOWED_ORIGIN    var, e.g. https://hamad470.github.io

// Any Gemini text model name is accepted; if Google has retired it (404), the
// Worker asks Google which models the key can use and retries with the best Flash.
const MODEL_NAME = /^gemini-[a-z0-9.\-]+$/;
const REJECT = /(embedding|aqa|vision|image|audio|tts|imagen|veo|gemma|live|computer-use|robotics)/;

async function bestModel(key) {
  const r = await fetch("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", { headers: { "x-goog-api-key": key } });
  const data = await r.json().catch(() => ({}));
  const names = (data.models || [])
    .filter((m) => (m.supportedGenerationMethods || []).includes("generateContent"))
    .map((m) => String(m.name).replace(/^models\//, ""))
    .filter((n) => n.startsWith("gemini") && !REJECT.test(n) && !/(preview|exp|lite|pro)/.test(n) && /flash/.test(n));
  const ver = (n) => parseFloat((n.match(/gemini-(\d+(?:\.\d+)?)/) || [])[1] || 0);
  names.sort((a, b) => ver(b) - ver(a) || a.length - b.length);
  return names[0] || null;
}
const MAX_BODY = 60_000;              // bytes; the demo's prompts are ~10 kB
const PER_MINUTE = 8;                 // requests per IP per minute (best effort, per isolate)
const hits = new Map();

function cors(origin, allowed) {
  return {
    "Access-Control-Allow-Origin": origin === allowed ? origin : allowed,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
}
const reply = (status, obj, headers) =>
  new Response(JSON.stringify(obj), { status, headers: { ...headers, "Content-Type": "application/json" } });

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const allowed = env.ALLOWED_ORIGIN || "";
    const h = cors(origin, allowed);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: h });
    if (request.method !== "POST") return reply(405, { error: { message: "POST only" } }, h);
    if (!allowed || origin !== allowed) return reply(403, { error: { message: "Origin not allowed" } }, h);

    const ip = request.headers.get("CF-Connecting-IP") || "?";
    const now = Date.now();
    const recent = (hits.get(ip) || []).filter((t) => now - t < 60_000);
    if (recent.length >= PER_MINUTE) return reply(429, { error: { message: "Too many requests, wait a minute." } }, h);
    recent.push(now);
    hits.set(ip, recent);

    const text = await request.text();
    if (text.length > MAX_BODY) return reply(413, { error: { message: "Request too large" } }, h);
    let payload;
    try { payload = JSON.parse(text); } catch { return reply(400, { error: { message: "Invalid JSON" } }, h); }
    let model = MODEL_NAME.test(payload.model || "") && !REJECT.test(payload.model) ? payload.model : "gemini-flash-latest";

    const keys = (env.GEMINI_API_KEYS || "").split(",").map((k) => k.trim()).filter(Boolean);
    if (!keys.length) return reply(503, { error: { message: "No Gemini key configured on the proxy." } }, h);
    let last;
    const call = (key, m) => fetch(`https://generativelanguage.googleapis.com/v1beta/models/${m}:generateContent`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify(payload.body),
    });
    for (const key of keys) {
      last = await call(key, model);
      if (last.status === 404) {
        const better = await bestModel(key);
        if (better && better !== model) { model = better; last = await call(key, model); }
      }
      if (last.status !== 429) break;
    }
    return new Response(last.body, { status: last.status, headers: { ...h, "Content-Type": "application/json" } });
  },
};
