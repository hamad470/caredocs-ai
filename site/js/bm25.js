// Okapi BM25, the same formula and constants as rag_advanced.BM25 in Python:
//   score(D,Q) = Σ IDF(q) · f(q,D)(k1+1) / (f(q,D) + k1(1 − b + b·|D|/avgdl))
//   IDF(q)     = max(ln((N − df + 0.5)/(df + 0.5) + 1), 0.01)

export class BM25 {
  constructor(corpusTokens, k1 = 1.5, b = 0.75) {
    this.k1 = k1;
    this.b = b;
    this.nDocs = corpusTokens.length;
    this.docLen = new Float64Array(this.nDocs);
    this.inverted = new Map();          // term -> Map(docIndex -> tf)
    let total = 0;
    corpusTokens.forEach((tokens, idx) => {
      this.docLen[idx] = tokens.length;
      total += tokens.length;
      const tf = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
      for (const [t, f] of tf) {
        let postings = this.inverted.get(t);
        if (!postings) this.inverted.set(t, (postings = new Map()));
        postings.set(idx, f);
      }
    });
    this.avgdl = this.nDocs ? total / this.nDocs : 0;
    this.idf = new Map();
    for (const [t, postings] of this.inverted) {
      const df = postings.size;
      this.idf.set(t, Math.max(Math.log((this.nDocs - df + 0.5) / (df + 0.5) + 1), 0.01));
    }
  }

  // Returns Map(docIndex -> score) for documents matching at least one term.
  // `allowed` is an optional Set of document indexes (metadata pre-filter).
  scores(queryTokens, allowed = null) {
    const out = new Map();
    for (const t of queryTokens) {
      const postings = this.inverted.get(t);
      if (!postings) continue;
      const idf = this.idf.get(t);
      for (const [idx, f] of postings) {
        if (allowed && !allowed.has(idx)) continue;
        const denom = f + this.k1 * (1 - this.b + (this.b * this.docLen[idx]) / (this.avgdl || 1));
        out.set(idx, (out.get(idx) || 0) + (idf * (f * (this.k1 + 1))) / denom);
      }
    }
    return out;
  }

  // Score a short piece of text (one line) against the corpus statistics.
  scoreText(queryTokens, textTokens) {
    const tf = new Map();
    for (const t of textTokens) tf.set(t, (tf.get(t) || 0) + 1);
    let s = 0;
    for (const t of new Set(queryTokens)) {
      const f = tf.get(t);
      if (!f) continue;
      const denom = f + this.k1 * (1 - this.b + (this.b * textTokens.length) / 12);
      s += ((this.idf.get(t) || 0) * (f * (this.k1 + 1))) / denom;
    }
    return s;
  }
}
