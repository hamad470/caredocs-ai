"""
rag_chat_eval.py — retrieval ablation study for the advanced RAG pipeline
Produces the evidence needed to justify each pipeline stage in the
dissertation, rather than asserting that hybrid retrieval "is better".

METHOD (known-item retrieval)
    A gold set is built automatically from the index itself. For each sampled
    chunk we synthesise a query from its content — either a paraphrase-style
    keyword bag (drops stopwords and shuffles order, so exact string matching
    cannot trivially win) or a verbatim sentence — and record the source chunk
    as the single relevant document. This is the standard known-item protocol
    used to evaluate retrieval without human relevance judgements
    (Voorhees & Harman, TREC; Craswell, 2009, "Mean Reciprocal Rank").

Metrics
    Recall@k   proportion of queries whose gold chunk appears in the top k
    MRR@k      mean of 1/rank of the gold chunk (0 if not retrieved)
    nDCG@k     discounted gain, single relevant item (Järvelin & Kekäläinen, 2002)
    latency    mean milliseconds per query

ARMS
    bm25        lexical only
    dense       vector only (TF-IDF+LSA, or neural if installed)
    hybrid      BM25 + dense fused with RRF
    hybrid+mmr  adds MMR diversification
    full        adds recency weighting (the configuration the chat uses)

CAVEAT to state in the write-up: synthetic known-item queries measure whether
the retriever can find a specific record it has seen; they do not measure
answer quality, which needs human judgement. Treat this as a component-level
ablation, not an end-to-end evaluation.

    python rag_chat_eval.py                # 120 queries, k=8
    python rag_chat_eval.py --n 300 --k 5
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import random

import rag_advanced

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "carehome.db")
OUT = os.path.join(rag_advanced.INDEX_DIR, "ablation_results.json")

ARMS = [
    ("bm25",       dict(retriever="bm25",   use_mmr=False, use_recency=False)),
    ("dense",      dict(retriever="dense",  use_mmr=False, use_recency=False)),
    ("hybrid",     dict(retriever="hybrid", use_mmr=False, use_recency=False)),
    ("hybrid+mmr", dict(retriever="hybrid", use_mmr=True,  use_recency=False)),
    ("full",       dict(retriever="hybrid", use_mmr=True,  use_recency=True)),
]


JACCARD_DUPLICATE = 0.85


def build_gold_set(n: int, seed: int = 42) -> list[dict]:
    """
    Sample chunks and synthesise a query for each.

    IMPORTANT — duplicate-aware relevance. The corpus is generated care
    documentation, so it contains many near-identical chunks ("Supported X with
    personal care this morning shift. Mood observed as low."). Under strict
    known-item scoring, retrieving one of those twins counts as a miss even
    though it is an equally correct answer, which systematically understates
    every arm and penalises MMR hardest (its entire job is to drop twins).
    The relevance set for each query is therefore the gold chunk plus every
    chunk whose token set overlaps it by Jaccard ≥ 0.85.
    """
    cache = rag_advanced._load()
    if cache is None:
        raise SystemExit("No index found. Run:  python rag_advanced.py")
    chunks = cache["chunks"]
    rng = random.Random(seed)
    token_sets = [set(rag_advanced.tokenize(c["text"])) for c in chunks]

    # Only sample chunks with enough content to make a non-trivial query.
    pool = [i for i, c in enumerate(chunks) if len(c["text"].split()) >= 25]
    rng.shuffle(pool)
    gold = []

    for idx in pool[: n * 2]:
        c = chunks[idx]
        sentences = rag_advanced._split_sentences(c["text"])
        if not sentences:
            continue
        sent = max(sentences, key=len)

        if len(gold) % 2 == 0:
            # keyword-bag query: content words only, order shuffled
            words = [w for w in rag_advanced.tokenize(sent) if len(w) > 3]
            if len(words) < 5:
                continue
            picked = rng.sample(words, min(8, len(words)))
            rng.shuffle(picked)
            query, style = " ".join(picked), "keywords"
        else:
            query, style = sent[:220], "verbatim"

        gold_tokens = token_sets[idx]
        relevant = {c["chunk_id"]}
        for j, other in enumerate(token_sets):
            if j == idx or not other:
                continue
            inter = len(gold_tokens & other)
            if inter and inter / len(gold_tokens | other) >= JACCARD_DUPLICATE:
                relevant.add(chunks[j]["chunk_id"])

        gold.append({"query": query, "style": style, "gold_chunk": c["chunk_id"],
                     "relevant": sorted(relevant), "num_relevant": len(relevant),
                     "resident_id": c["resident_id"], "source_type": c["source_type"]})
        if len(gold) >= n:
            break
    return gold


def evaluate(gold: list[dict], k: int) -> dict:
    results = {}
    for arm_name, opts in ARMS:
        hit, rr, ndcg, latencies = 0, 0.0, 0.0, []
        for item in gold:
            t0 = time.time()
            hits = rag_advanced.retrieve(item["query"], top_k=k, **opts)
            latencies.append((time.time() - t0) * 1000)
            ids = [h["chunk_id"] for h in hits]
            relevant = set(item.get("relevant") or [item["gold_chunk"]])
            rank = next((i + 1 for i, cid in enumerate(ids) if cid in relevant), None)
            if rank:
                hit += 1
                rr += 1.0 / rank
                ndcg += 1.0 / math.log2(rank + 1)      # first relevant item, gain 1
        n = len(gold)
        results[arm_name] = {
            "recall_at_k": round(hit / n, 4),
            "mrr": round(rr / n, 4),
            "ndcg": round(ndcg / n, 4),
            "mean_latency_ms": round(sum(latencies) / n, 1),
            "queries": n,
        }
        print(f"  {arm_name:<12} Recall@{k} {results[arm_name]['recall_at_k']:.3f}   "
              f"MRR {results[arm_name]['mrr']:.3f}   "
              f"nDCG {results[arm_name]['ndcg']:.3f}   "
              f"{results[arm_name]['mean_latency_ms']:>6.1f} ms/query")
    return results


def main():
    n = 120
    k = 8
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    if "--k" in sys.argv:
        k = int(sys.argv[sys.argv.index("--k") + 1])

    if not rag_advanced.get_status().get("built"):
        print("[*] Building index first …")
        rag_advanced.build_index(DB)

    print(f"[*] Building gold set of {n} known-item queries …")
    gold = build_gold_set(n)
    styles = {}
    for g in gold:
        styles[g["style"]] = styles.get(g["style"], 0) + 1
    print(f"    {len(gold)} queries ({', '.join(f'{v} {k2}' for k2, v in styles.items())})\n")

    print(f"[*] Running ablation at k={k} …")
    results = evaluate(gold, k)

    best = max(results.items(), key=lambda kv: kv[1]["mrr"])
    baseline = results["dense"]["mrr"] or 1e-9
    lift = (results["full"]["mrr"] - results["dense"]["mrr"]) / baseline * 100

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "k": k, "num_queries": len(gold),
        "index": rag_advanced.get_status(),
        "arms": results,
        "best_arm": best[0],
        "full_vs_dense_mrr_lift_pct": round(lift, 1),
    }
    os.makedirs(rag_advanced.INDEX_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Best arm by MRR: {best[0]}")
    print(f"  Full pipeline vs dense-only: {lift:+.1f}% MRR")
    print(f"  Saved → {OUT}")

    # Honest interpretation — MMR and recency usually cost a little on a
    # known-item metric by design, because both deliberately demote the
    # highest-scoring twin of an already-selected chunk. Their benefit shows up
    # in answer synthesis (a context window of six distinct records beats six
    # paraphrases of one), which this protocol does not measure.
    mmr_delta = results["hybrid+mmr"]["mrr"] - results["hybrid"]["mrr"]
    print("\n  Interpretation:")
    print(f"   • fusion vs single retriever: hybrid MRR {results['hybrid']['mrr']:.3f} "
          f"vs bm25 {results['bm25']['mrr']:.3f} / dense {results['dense']['mrr']:.3f}")
    print(f"   • MMR changes MRR by {mmr_delta:+.3f} — expected to be slightly negative on a "
          f"known-item metric,\n     since MMR trades rank precision for coverage; judge it on "
          f"answer quality, not this table")
    print("   • state in the write-up that these are synthetic known-item queries with "
          "duplicate-aware\n     relevance, so they test component retrieval, not end-to-end "
          "answer correctness")
    print("\n  Markdown table for the dissertation:\n")
    print(f"  | Configuration | Recall@{k} | MRR | nDCG@{k} | ms/query |")
    print("  |---|---|---|---|---|")
    for name, r in results.items():
        print(f"  | {name} | {r['recall_at_k']:.3f} | {r['mrr']:.3f} | "
              f"{r['ndcg']:.3f} | {r['mean_latency_ms']:.1f} |")


if __name__ == "__main__":
    main()
