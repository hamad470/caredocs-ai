"""
rag_experiments.py — reproducible retrieval evaluation for the v2 engine

Why this file exists
Chapter 6 reports four retrieval results. Until v3 those numbers came from
throwaway scripts, which meant they could not be re-run, re-checked, or updated
when the corpus changed. That is not an acceptable basis for a dissertation
result table, so the whole evaluation now lives here, is deterministic given a
seed, and writes machine-readable JSON that the write-up reads directly.

The four experiments

E1 — RETRIEVAL ABLATION (which stages of the pipeline actually earn their place?)
    Known-item retrieval. A query is synthesised from a randomly chosen chunk of
    the corpus; the chunk it came from is the single known relevant document;
    the retriever is asked to find it again. This is the standard way to build a
    relevance-judged benchmark without human annotators, and it is honest about
    what it measures: it tests whether the pipeline can find a passage it has
    definitely seen, not whether a clinician would judge the result useful.

    Metrics: MRR, Recall@k, nDCG@k, and median wall-clock latency per query.

    Arms, added one stage at a time so each row isolates one design decision:
        1. BM25 only              — lexical baseline
        2. Dense only             — LSA/embedding baseline
        3. Hybrid (RRF)           — does fusing the two beat either alone?
        4. Hybrid + recency       — does time weighting help or hurt?
        5. Hybrid + recency + MMR — does diversification cost accuracy?

E2 — contextual enrichment ablation
    Every chunk is indexed with a generated header naming the resident, record
    type and date (Anthropic's "contextual retrieval", 2024). This experiment
    rebuilds the query set to use *resident-agnostic* phrasing and measures what
    happens with and without that header in the indexed text. It is the only
    experiment here that tests a design decision rather than a component.

E3 — RESIDENT ISOLATION (a safety property, measured rather than asserted)
    Confidentiality in a care home is not a nice-to-have; disclosing one
    resident's record to a query about another is a data breach. The system
    applies the resident filter as a PRE-filter — out-of-scope chunks are
    removed from the candidate pool before either retriever runs — so leakage
    should be structurally impossible rather than merely unlikely.

    "Should be" is not evidence. This experiment fires adversarial queries built
    from resident A's own records while the filter is set to resident B, and
    counts leaked chunks. With 50 residents there are 2,450 ordered pairs to
    draw from, which is a far stronger test than the 5-resident version allowed.
    A zero count is reported with its Wilson 95 % upper bound, because "0 out of
    n" is not the same claim as "never".

E4 — chunk-size sensitivity
    Reports how retrieval quality varies with the word budget, so the chosen
    value (130 words, 35-word overlap) is a measured choice rather than a
    default that happened to be in the first tutorial.

Usage
    python rag_experiments.py                     # all experiments, default sizes
    python rag_experiments.py --queries 300
    python rag_experiments.py --json results/rag_experiments.json
    python rag_experiments.py --only isolation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import rag_advanced as R  # noqa: E402

STOP = {
    "the", "and", "was", "with", "for", "her", "his", "she", "him", "had", "has",
    "have", "that", "this", "from", "were", "been", "are", "but", "not", "all",
    "any", "out", "over", "into", "than", "then", "them", "they", "their", "there",
    "which", "when", "what", "who", "whom", "will", "would", "could", "should",
    "shift", "resident", "staff", "care", "day",
}


# Query synthesis

def _content_terms(text: str, n: int = 8) -> list[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)]
    seen, out = set(), []
    for w in words:
        if w in STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= n:
            break
    return out


def build_query_set(chunks: list[dict], n_queries: int, seed: int,
                    include_resident_name: bool = True) -> list[dict]:
    """
    Synthesise known-item queries.

    Each query is a bag of content terms lifted from one chunk, optionally
    prefixed with the resident's name. The chunk it came from is the ground
    truth. Chunks shorter than a floor are skipped: a three-word chunk produces
    a query with no discriminating content and would measure noise.
    """
    rng = random.Random(seed)
    pool = [c for c in chunks if len(c["text"].split()) >= 25]
    rng.shuffle(pool)
    qs = []
    for c in pool[:n_queries]:
        terms = _content_terms(c["text"], 8)
        if len(terms) < 4:
            continue
        q = " ".join(terms)
        if include_resident_name and c.get("resident_name"):
            q = f"{c['resident_name']} {q}"
        qs.append({"query": q, "gold_chunk_id": c["chunk_id"],
                   "resident_id": c["resident_id"],
                   "source_type": c["source_type"]})
    return qs


# Metrics

def _rank_of(results: list[dict], gold_id: str) -> int | None:
    for i, r in enumerate(results, start=1):
        if r.get("chunk_id") == gold_id:
            return i
    return None


def _ndcg_at(rank: int | None, k: int) -> float:
    """nDCG with a single relevant document: 1/log2(rank+1) if inside k."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _wilson_upper(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson upper bound — the right interval when the observed count is 0."""
    if n == 0:
        return 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / denom)


# E1 — retrieval ablation

ARMS = [
    ("BM25 only",                 dict(retriever="bm25",   use_recency=False, use_mmr=False)),
    ("Dense only (LSA)",          dict(retriever="dense",  use_recency=False, use_mmr=False)),
    ("Hybrid (RRF)",              dict(retriever="hybrid", use_recency=False, use_mmr=False)),
    ("Hybrid + recency",          dict(retriever="hybrid", use_recency=True,  use_mmr=False)),
    ("Hybrid + recency + MMR",    dict(retriever="hybrid", use_recency=True,  use_mmr=True)),
]


def run_ablation(queries: list[dict], top_k: int = 8,
                 scope_to_resident: bool = True, progress=True) -> list[dict]:
    rows = []
    for label, kwargs in ARMS:
        ranks, lat = [], []
        for q in queries:
            t0 = time.perf_counter()
            res = R.retrieve(
                q["query"], top_k=top_k,
                resident_ids=[q["resident_id"]] if scope_to_resident else None,
                **kwargs)
            lat.append((time.perf_counter() - t0) * 1000)
            ranks.append(_rank_of(res, q["gold_chunk_id"]))
        n = len(ranks)
        found = [r for r in ranks if r is not None]
        rows.append({
            "arm": label,
            "n_queries": n,
            "mrr": round(sum(1.0 / r for r in found) / n, 4) if n else 0.0,
            "recall_at_1": round(sum(1 for r in found if r <= 1) / n, 4) if n else 0.0,
            "recall_at_3": round(sum(1 for r in found if r <= 3) / n, 4) if n else 0.0,
            f"recall_at_{top_k}": round(len(found) / n, 4) if n else 0.0,
            f"ndcg_at_{top_k}": round(sum(_ndcg_at(r, top_k) for r in ranks) / n, 4) if n else 0.0,
            "median_latency_ms": round(statistics.median(lat), 2) if lat else None,
            "p95_latency_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2) if lat else None,
        })
        if progress:
            print(f"    {label:<28} MRR {rows[-1]['mrr']:.4f}  "
                  f"R@{top_k} {rows[-1][f'recall_at_{top_k}']:.4f}  "
                  f"{rows[-1]['median_latency_ms']} ms", flush=True)
    return rows


# E1b — recency weight sweep

def run_recency_sweep(queries: list[dict], weights=(0.0, 0.10, 0.15, 0.20, 0.35, 0.50),
                      top_k: int = 8, scope_to_resident: bool = True,
                      progress: bool = True) -> list[dict]:
    """
    Recency weighting was adopted on the reasonable-sounding argument that a
    note from last week matters more than the same note from last year. E1
    measures it at one setting; this sweeps it, because a stage that helps at
    one weight and hurts at another is not really "justified" by a single arm.

    Everything but the weight is held fixed: hybrid fusion, no MMR, the same
    query set, the same k. RECENCY_WEIGHT is restored afterwards so the module
    is left in its production configuration whatever happens.
    """
    original = R.RECENCY_WEIGHT
    rows = []
    try:
        for w in weights:
            R.RECENCY_WEIGHT = float(w)
            ranks, lat = [], []
            for q in queries:
                t0 = time.perf_counter()
                res = R.retrieve(
                    q["query"], top_k=top_k,
                    resident_ids=[q["resident_id"]] if scope_to_resident else None,
                    retriever="hybrid", use_recency=(w > 0), use_mmr=False)
                lat.append((time.perf_counter() - t0) * 1000)
                ranks.append(_rank_of(res, q["gold_chunk_id"]))
            n = len(ranks)
            found = [r for r in ranks if r is not None]
            rows.append({
                "recency_weight": w,
                "n_queries": n,
                "mrr": round(sum(1.0 / r for r in found) / n, 4) if n else 0.0,
                "recall_at_1": round(sum(1 for r in found if r <= 1) / n, 4) if n else 0.0,
                f"recall_at_{top_k}": round(len(found) / n, 4) if n else 0.0,
                f"ndcg_at_{top_k}": round(sum(_ndcg_at(r, top_k) for r in ranks) / n, 4) if n else 0.0,
                "median_latency_ms": round(statistics.median(lat), 2) if lat else None,
            })
            if progress:
                print(f"    w = {w:<5} MRR {rows[-1]['mrr']:.4f}  "
                      f"R@{top_k} {rows[-1][f'recall_at_{top_k}']:.4f}", flush=True)
    finally:
        R.RECENCY_WEIGHT = original
    return rows


# E2 — contextual enrichment ablation

def run_header_ablation(db_path: str, n_queries: int, seed: int,
                        top_k: int = 8) -> dict:
    """
    Does the generated context header earn its place?

    The header ("Margaret Brown (RES001) | Care Note | 2026-06-12 | Night shift")
    is prepended to every chunk BEFORE vectorisation, so scope information lives
    in the representation rather than beside it (Anthropic, contextual retrieval,
    2024). It can only help when the query carries scope the chunk body does not,
    so the test is run in the setting that reproduces that: queries that name the
    resident, retrieved with NO resident filter, against a 50-resident corpus in
    which any one query is competing with 49 other people's records.

    Both arms require a full index rebuild, because the ablation is on what gets
    vectorised. The index is rebuilt with headers on at the end so the working
    tree is left in the production configuration.
    """
    out = {}
    try:
        for arm, flag in (("with_header", True), ("without_header", False)):
            R.USE_CONTEXT_HEADER = flag
            R.invalidate_cache()
            R.build_index(db_path, use_neural=False, verbose=False)
            cache = R._load()
            qs = build_query_set(cache["chunks"], n_queries, seed,
                                 include_resident_name=True)
            ranks, right_person = [], 0
            for q in qs:
                res = R.retrieve(q["query"], top_k=top_k, resident_ids=None,
                                 retriever="hybrid", use_recency=True, use_mmr=True)
                ranks.append(_rank_of(res, q["gold_chunk_id"]))
                if res:
                    right_person += sum(
                        1 for r in res if r.get("resident_id") == q["resident_id"]
                    ) / len(res)
            n = len(ranks)
            found = [r for r in ranks if r is not None]
            out[arm] = {
                "n_queries": n,
                "mrr": round(sum(1.0 / r for r in found) / n, 4) if n else 0.0,
                f"recall_at_{top_k}": round(len(found) / n, 4) if n else 0.0,
                "mean_share_of_results_from_the_named_resident":
                    round(right_person / n, 4) if n else 0.0,
            }
            print(f"    {arm:<20} MRR {out[arm]['mrr']:.4f}  "
                  f"R@{top_k} {out[arm][f'recall_at_{top_k}']:.4f}  "
                  f"right-resident share "
                  f"{out[arm]['mean_share_of_results_from_the_named_resident']:.4f}",
                  flush=True)
    finally:
        R.USE_CONTEXT_HEADER = True
        R.invalidate_cache()
        R.build_index(db_path, use_neural=False, verbose=False)

    if "with_header" in out and "without_header" in out:
        out["delta_mrr"] = round(out["with_header"]["mrr"] - out["without_header"]["mrr"], 4)
        out["delta_right_resident_share"] = round(
            out["with_header"]["mean_share_of_results_from_the_named_resident"]
            - out["without_header"]["mean_share_of_results_from_the_named_resident"], 4)
    out["note"] = (
        "Unfiltered retrieval over 50 residents. Without the header, a chunk "
        "reading 'declined breakfast, low mood' is indistinguishable from the same "
        "sentence about a different resident eight months earlier, so the retriever "
        "has nothing to key the resident's name against. The header puts that "
        "information inside the vector.")
    return out


# E3 — resident isolation

def run_isolation(chunks: list[dict], n_queries: int, seed: int,
                  top_k: int = 8) -> dict:
    """
    Adversarial cross-resident retrieval.

    For each trial: take a chunk belonging to resident A, build a query from it
    that also names A explicitly, then ask the retriever for it while scoping the
    filter to a DIFFERENT resident B. Any returned chunk not belonging to B is a
    confidentiality breach.
    """
    rng = random.Random(seed)
    residents = sorted({c["resident_id"] for c in chunks})
    by_res = {}
    for c in chunks:
        if len(c["text"].split()) >= 25:
            by_res.setdefault(c["resident_id"], []).append(c)
    residents = [r for r in residents if by_res.get(r)]
    if len(residents) < 2:
        return {"error": "need at least two residents"}

    leaked_chunks = 0
    total_chunks = 0
    leaked_queries = 0
    gold_returned = 0
    for _ in range(n_queries):
        a, b = rng.sample(residents, 2)
        src = rng.choice(by_res[a])
        terms = _content_terms(src["text"], 8)
        q = f"{src.get('resident_name') or a} {' '.join(terms)}"
        res = R.retrieve(q, top_k=top_k, resident_ids=[b],
                         retriever="hybrid", use_recency=True, use_mmr=True)
        total_chunks += len(res)
        bad = [r for r in res if r.get("resident_id") != b]
        leaked_chunks += len(bad)
        if bad:
            leaked_queries += 1
        if any(r.get("chunk_id") == src["chunk_id"] for r in res):
            gold_returned += 1

    return {
        "n_queries": n_queries,
        "n_residents": len(residents),
        "ordered_resident_pairs_available": len(residents) * (len(residents) - 1),
        "chunks_returned": total_chunks,
        "chunks_leaked": leaked_chunks,
        "queries_with_any_leak": leaked_queries,
        "target_chunk_returned": gold_returned,
        "leak_rate": round(leaked_chunks / total_chunks, 6) if total_chunks else None,
        # The independent unit is the QUERY, not the chunk. All k chunks a query
        # returns pass through one filter decision on one resident pair, so they
        # are one Bernoulli trial with a multiplicity of k. Bounding on the chunk
        # count would claim roughly k times more precision than the design
        # supports; both are reported, and the query-level bound is the one to
        # quote.
        "leak_rate_wilson_upper_95": round(
            _wilson_upper(leaked_queries, n_queries), 6) if n_queries else None,
        "leak_rate_wilson_upper_95_unit": "query",
        "leak_rate_wilson_upper_95_per_chunk_optimistic": round(
            _wilson_upper(leaked_chunks, total_chunks), 6) if total_chunks else None,
        "mechanism": "metadata pre-filter applied to the candidate pool before "
                     "BM25 and dense retrieval run",
        "note": ("Pre-filtering rather than post-filtering is the design decision "
                 "under test. A post-filter is correct only when every call site "
                 "remembers to apply it, so correctness would depend on the "
                 "discipline of future callers; a pre-filter makes the failure "
                 "mode unreachable. A leak count of 0 is reported with its Wilson "
                 "95 % upper bound because 0/n is a bounded claim, not proof of "
                 "impossibility. The bound is computed on queries rather than on "
                 "returned chunks: the chunks a single query returns share one "
                 "filter decision and are not independent trials."),
    }


# E4 — chunk-size sensitivity

def run_chunk_sensitivity(db_path: str, sizes, n_queries: int, seed: int,
                          top_k: int = 8) -> list[dict]:
    """
    Rebuild the index at several word budgets and re-measure. Slow by nature —
    each setting is a full index build — which is why it is opt-in.
    """
    original = (R.CHUNK_WORDS, R.CHUNK_OVERLAP)
    rows = []
    try:
        for words, overlap in sizes:
            R.CHUNK_WORDS, R.CHUNK_OVERLAP = words, overlap
            R.invalidate_cache()
            info = R.build_index(db_path, use_neural=False, verbose=False)
            cache = R._load()
            qs = build_query_set(cache["chunks"], n_queries, seed)
            ranks = []
            for q in qs:
                res = R.retrieve(q["query"], top_k=top_k,
                                 resident_ids=[q["resident_id"]],
                                 retriever="hybrid", use_recency=True, use_mmr=True)
                ranks.append(_rank_of(res, q["gold_chunk_id"]))
            n = len(ranks)
            found = [r for r in ranks if r is not None]
            rows.append({
                "chunk_words": words, "overlap_words": overlap,
                "num_chunks": info.get("num_chunks"),
                "mrr": round(sum(1.0 / r for r in found) / n, 4) if n else 0.0,
                f"recall_at_{top_k}": round(len(found) / n, 4) if n else 0.0,
            })
            print(f"    {words}w/{overlap}o -> {info.get('num_chunks')} chunks, "
                  f"MRR {rows[-1]['mrr']:.4f}", flush=True)
    finally:
        R.CHUNK_WORDS, R.CHUNK_OVERLAP = original
        R.invalidate_cache()
        R.build_index(db_path, use_neural=False, verbose=False)
    return rows



def corpus_profile(cache: dict) -> dict:
    chunks = cache["chunks"]
    by_type, by_res = {}, {}
    for c in chunks:
        by_type[c["source_type"]] = by_type.get(c["source_type"], 0) + 1
        by_res[c["resident_id"]] = by_res.get(c["resident_id"], 0) + 1
    lens = [len(c["text"].split()) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "num_residents": len(by_res),
        "chunks_by_source_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "chunks_per_resident": {
            "min": min(by_res.values()), "max": max(by_res.values()),
            "mean": round(statistics.mean(by_res.values()), 1)},
        "chunk_length_words": {
            "mean": round(statistics.mean(lens), 1),
            "median": statistics.median(lens),
            "p95": sorted(lens)[int(0.95 * (len(lens) - 1))]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible RAG retrieval experiments.")
    ap.add_argument("--db", default=os.path.join(_BASE, "carehome.db"))
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--isolation-queries", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--json", metavar="PATH", default=None)
    ap.add_argument("--only", choices=["ablation", "recency", "header", "isolation", "chunks"],
                    default=None)
    ap.add_argument("--chunk-sweep", action="store_true",
                    help="run E4 (slow: rebuilds the index once per setting)")
    args = ap.parse_args()

    cache = R._load()
    if cache is None:
        print("[*] No index found — building it now ...")
        R.build_index(args.db, use_neural=False, verbose=True)
        cache = R._load()
    chunks = cache["chunks"]

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed, "top_k": args.top_k,
        "corpus": corpus_profile(cache),
        "index": {k: v for k, v in (cache.get("manifest") or {}).items()},
    }
    BAR = "=" * 78
    print(BAR)
    print(" RAG RETRIEVAL EXPERIMENTS")
    print(BAR)
    print(f" Corpus: {out['corpus']['num_chunks']:,} chunks over "
          f"{out['corpus']['num_residents']} residents")
    print(f" By type: {out['corpus']['chunks_by_source_type']}")
    print()

    run_all = args.only is None

    if run_all or args.only == "ablation":
        print(" E1 — retrieval ablation (known-item, resident-scoped)")
        qs = build_query_set(chunks, args.queries, args.seed)
        print(f"    {len(qs)} synthesised known-item queries")
        out["e1_ablation"] = run_ablation(qs, top_k=args.top_k)
        print()
        print(" E1b — recency weight sweep (hybrid, no MMR)")
        out["e1b_recency_sweep"] = run_recency_sweep(qs, top_k=args.top_k)
        print()

    if run_all or args.only == "header":
        print(" E2 — contextual enrichment (unfiltered vs filtered)")
        out["e2_context_header"] = run_header_ablation(
            args.db, min(args.queries, 200), args.seed + 1, top_k=args.top_k)
        print(f"    delta MRR from the context header: "
              f"{out['e2_context_header'].get('delta_mrr')}")
        print()

    if run_all or args.only == "isolation":
        print(" E3 — resident isolation (adversarial cross-resident queries)")
        out["e3_isolation"] = run_isolation(
            chunks, args.isolation_queries, args.seed + 2, top_k=args.top_k)
        iso = out["e3_isolation"]
        print(f"    {iso['n_queries']} adversarial queries over "
              f"{iso['ordered_resident_pairs_available']:,} ordered resident pairs")
        print(f"    leaked {iso['chunks_leaked']} of {iso['chunks_returned']:,} "
              f"returned chunks  (Wilson 95 % upper bound "
              f"{iso['leak_rate_wilson_upper_95']:.6f})")
        print()

    if args.chunk_sweep or args.only == "chunks":
        print(" E4 — chunk-size sensitivity (rebuilds the index per setting)")
        out["e4_chunk_sensitivity"] = run_chunk_sensitivity(
            args.db, [(80, 20), (130, 35), (200, 50)],
            min(args.queries, 150), args.seed + 3, top_k=args.top_k)
        print()

    print(BAR)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f" Written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
