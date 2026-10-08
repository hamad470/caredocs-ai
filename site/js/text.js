// Tokeniser shared by retrieval and verification.
// Mirrors rag_advanced.tokenize() in Python exactly, so the browser index and
// the app's index see the same terms (checked by test/retrieval.test.mjs).

export const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "did", "do",
  "does", "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
  "is", "it", "no", "not", "of", "on", "or", "she", "that", "the", "their",
  "them", "there", "they", "this", "to", "was", "we", "were", "with", "you",
]);

const TOKEN_RE = /[a-z0-9]+/g;

export function tokenize(text) {
  const found = (text || "").toLowerCase().match(TOKEN_RE) || [];
  return found.filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

// Light suffix stripping, used only when checking whether a claim's words
// appear in its source ("refused" in the answer, "refusing" in the record).
export function stem(t) {
  if (/^\d/.test(t) || t.length <= 4) return t;
  for (const suf of ["ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s", "ly"]) {
    if (t.endsWith(suf) && t.length - suf.length >= 3) {
      return suf === "ies" ? t.slice(0, -3) + "y" : t.slice(0, -suf.length);
    }
  }
  return t;
}

export const MONTHS = ["january", "february", "march", "april", "may", "june", "july",
  "august", "september", "october", "november", "december"];

// "2026-04-28" -> "28 April 2026"
export function longDate(iso) {
  if (!iso || iso.length < 10) return iso || "";
  const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
  const name = MONTHS[m - 1];
  return `${d} ${name[0].toUpperCase()}${name.slice(1)} ${y}`;
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
