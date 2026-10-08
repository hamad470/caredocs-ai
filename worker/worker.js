// Optional Gemini proxy for the live demo, so visitors need no key of their own.
// The keys live in Cloudflare as a secret and never reach the browser.
//
//   GEMINI_API_KEYS   secret, comma-separated; on a 429 the next key is tried
//   ALLOWED_ORIGIN    var, e.g. https://hamad470.github.io

const MODELS = new Set(["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]);
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
    const model = MODELS.has(payload.model) ? payload.model : "gemini-2.5-flash";

    const keys = (env.GEMINI_API_KEYS || "").split(",").map((k) => k.trim()).filter(Boolean);
    if (!keys.length) return reply(503, { error: { message: "No Gemini key configured on the proxy." } }, h);
    let last;
    for (const key of keys) {
      last = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": key },
        body: JSON.stringify(payload.body),
      });
      if (last.status !== 429) break;
    }
    return new Response(last.body, { status: last.status, headers: { ...h, "Content-Type": "application/json" } });
  },
};
