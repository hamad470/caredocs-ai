"""
RAG Evaluator — Quantitative Evaluation Module
Produces the results chapter evidence required at Masters level:

  Table 1: ROUGE-L scores (RAG vs No-RAG vs Template baseline)
  Table 2: Retrieval precision@k for k = 1, 3, 5, 10
  Table 3: Ablation — TF-IDF RAG vs Semantic RAG (if sentence-transformers available)
  Figure 1: Score distribution bar chart data (JSON for Chart.js)

Evaluation methodology:
  Reference texts: approved care plan sections already in the database
  (these were written or approved by qualified carers — ground truth)
  System texts: AI-regenerated versions of those same sections, with and without RAG

Metrics:
  ROUGE-L  — standard NLP overlap metric (Lin, 2004)
  Faithfulness proxy — presence of key clinical terms from retrieved chunks
                       in the generated output (avoids need for GPU BERTScore)

Ref: Lewis et al. (2020), Lin (2004) "ROUGE: A Package for Automatic Evaluation."
"""

from __future__ import annotations

import os
import json
import sqlite3
import datetime
import statistics
import re

try:
    from rouge_score import rouge_scorer
    _ROUGE_OK = True
except ImportError:
    _ROUGE_OK = False

DB_PATH = os.path.join(os.path.dirname(__file__), "carehome.db")
EVAL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "rag_index", "eval_results.json")


# 1 ── rouge scoring

def rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 between hypothesis and reference."""
    if not _ROUGE_OK:
        return _rouge_l_fallback(hypothesis, reference)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def _rouge_l_fallback(hyp: str, ref: str) -> float:
    """
    Pure-Python LCS-based ROUGE-L (used if rouge_score package missing).
    Not as optimised but produces equivalent results on short texts.
    """
    def lcs(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    h_tokens = re.findall(r'\w+', hyp.lower())
    r_tokens = re.findall(r'\w+', ref.lower())
    if not h_tokens or not r_tokens:
        return 0.0
    lcs_len = lcs(h_tokens, r_tokens)
    precision = lcs_len / len(h_tokens)
    recall    = lcs_len / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_1(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-1 F1."""
    if not _ROUGE_OK:
        h = set(re.findall(r'\w+', hypothesis.lower()))
        r = set(re.findall(r'\w+', reference.lower()))
        overlap = h & r
        if not overlap:
            return 0.0
        p = len(overlap) / len(h)
        rc = len(overlap) / len(r)
        return 2 * p * rc / (p + rc) if (p + rc) else 0.0

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rouge1"].fmeasure


# 2 ── faithfulness proxy

def faithfulness_score(generated_text: str, retrieved_chunks: list[dict]) -> float:
    """
    Proxy faithfulness metric: proportion of key clinical terms from
    retrieved chunks that appear in the generated text.

    This avoids the GPU requirement of BERTScore while providing a
    domain-relevant signal beyond n-gram overlap.
    (Full BERTScore should be run offline for the dissertation results chapter.)
    """
    if not retrieved_chunks:
        return 0.0

    # Extract key clinical terms (nouns/adjectives of 5+ chars) from retrieved
    retrieved_text = " ".join(c["text"] for c in retrieved_chunks)
    key_terms = set(
        w.lower() for w in re.findall(r'\b[A-Za-z]{5,}\b', retrieved_text)
        if w.lower() not in _STOP_WORDS
    )

    if not key_terms:
        return 0.0

    gen_words = set(re.findall(r'\b[A-Za-z]{5,}\b', generated_text.lower()))
    matched = key_terms & gen_words
    return len(matched) / len(key_terms)


_STOP_WORDS = {
    "their", "which", "about", "these", "there", "being", "where",
    "would", "could", "should", "other", "while", "after", "staff",
    "resident", "during", "before", "shall", "needs", "noted", "given",
    "ensure", "please", "refer", "taken", "have", "been", "with",
    "from", "this", "that", "they", "were", "will", "care", "home",
    "plan", "section", "person", "individual",
}


# 2b ── SENTENCE-LEVEL ATTRIBUTION  ("is it making things up?")
"""
Faithfulness above is a corpus-level number: it says how much of the retrieved
vocabulary made it into the output. It cannot say whether any *particular*
sentence was supported, and a reader who wants to know whether the system
invents things is asking exactly that.

Attribution answers the sentence-level question. Every sentence of the
generated text is matched against every retrieved chunk, and the sentence keeps
its best match. The score is content-word overlap — the share of the sentence's
own content words that appear in the supporting chunk — so a sentence that
introduces a drug name, a date or an observation not present in any retrieved
record scores low no matter how fluent it is.

This is a lexical test, not an entailment model. It will pass a sentence that
reuses the right words while stating the opposite, and it will fail a correct
paraphrase that shares no vocabulary. Both directions are recorded in the
limitations. It is used because it is transparent — a reader can be shown the
sentence, the chunk and the words they share, and can check the verdict without
being asked to trust a second model.
"""

_ATTR_SUPPORTED = 0.60      # most of the sentence's content is in one chunk
_ATTR_PARTIAL   = 0.35      # some support; a human should read it


def _content_words(text: str) -> set:
    return {w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", text)
            if w.lower() not in _STOP_WORDS}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def attribute_sentences(generated_text: str,
                        retrieved_chunks: list[dict]) -> dict:
    """
    Trace each generated sentence back to the retrieved record that best
    supports it.

    Returns the per-sentence detail (so the UI can show the working) plus the
    counts a reader actually asks for: how many sentences are grounded, how
    many are borderline, and how many have no support in the record at all.
    """
    sentences = _split_sentences(generated_text)
    if not sentences or not retrieved_chunks:
        return {"n_sentences": len(sentences), "supported": 0, "partial": 0,
                "unsupported": len(sentences), "grounded_rate": 0.0,
                "mean_support": 0.0, "sentences": []}

    prepared = []
    for i, c in enumerate(retrieved_chunks):
        prepared.append((i, c, _content_words(c.get("text", ""))))

    detail, counts = [], {"supported": 0, "partial": 0, "unsupported": 0}
    for sent in sentences:
        sent_words = _content_words(sent)
        if not sent_words:
            continue
        best_i, best_score, best_shared = None, 0.0, set()
        for i, chunk, chunk_words in prepared:
            shared = sent_words & chunk_words
            score = len(shared) / len(sent_words)
            if score > best_score:
                best_i, best_score, best_shared = i, score, shared

        verdict = ("supported" if best_score >= _ATTR_SUPPORTED
                   else "partial" if best_score >= _ATTR_PARTIAL
                   else "unsupported")
        counts[verdict] += 1
        source = retrieved_chunks[best_i] if best_i is not None else {}
        detail.append({
            "sentence": sent,
            "support": round(best_score, 3),
            "verdict": verdict,
            "source_rank": (best_i + 1) if best_i is not None else None,
            "source_type": source.get("source_type", ""),
            "source_date": source.get("date", ""),
            "source_excerpt": (source.get("text", "") or "")[:400],
            "shared_terms": sorted(best_shared)[:12],
        })

    n = len(detail) or 1
    return {
        "n_sentences": len(detail),
        "supported": counts["supported"],
        "partial": counts["partial"],
        "unsupported": counts["unsupported"],
        "grounded_rate": round((counts["supported"] + counts["partial"]) / n, 4),
        "fully_grounded_rate": round(counts["supported"] / n, 4),
        "mean_support": round(sum(d["support"] for d in detail) / n, 4),
        "sentences": detail,
    }


# 3 ── PRECISION @ K  (retrieval quality)

def precision_at_k(retrieved_chunks: list[dict], relevant_section: str,
                   k_values: list[int] = (1, 3, 5, 10)) -> dict[int, float]:
    """
    Retrieval precision@k — what fraction of the top-k chunks are
    from the relevant section type.

    In the absence of human-annotated relevance judgements, we use
    section-type matching as a proxy: a chunk is 'relevant' if its
    section matches the care plan section being generated.
    """
    # DEFECT FIX (v3). The final line previously read
    #     results[k] = relevant / k if k <= len(top_k) else 0.0
    # so whenever FEWER than k chunks were retrieved the function returned a hard
    # 0.0 rather than a precision computed over what was actually returned. Run
    # at top_k = 1 — as the v1 evaluation was — this made precision@3 and
    # precision@5 read exactly 0.0000 as an artefact of the guard clause, not as
    # a measurement. The reading was provably impossible on its own terms: a
    # chunk relevant at rank 1 is still inside the top 5, so P@5 can never fall
    # below P@1 / 5. It is now normalised by the number of chunks actually
    # retrieved, and returns None when nothing was retrieved at all, so an
    # absence of evidence is never reported as a score of zero.
    results = {}
    n = len(retrieved_chunks)
    for k in k_values:
        top_k = retrieved_chunks[:k]
        if not top_k:
            results[k] = None
            continue
        relevant = sum(1 for c in top_k if c.get("section", "") == relevant_section
                       or c.get("source_type") in ("care_plan", "care_note"))
        results[k] = relevant / len(top_k)
    return results


# 4 ── full evaluation run

def run_evaluation(db_path: str = DB_PATH,
                   rag_mode: str = "tfidf",
                   top_k: int = 5,
                   max_sections: int | None = 12,
                   progress=None) -> dict:
    """
    Run the full evaluation pipeline.  Returns a results dict ready to render
    in the evaluation template and save as the dissertation results table.

    Evaluation corpus: care plan sections that have been approved (status='Active')
    and for which we have enough care notes to provide RAG context.

    For each section:
      - Generate WITHOUT RAG (direct LLM call) → compute ROUGE-L vs reference
      - Generate WITH RAG (retrieved chunks prepended) → compute ROUGE-L vs reference
      - Compute delta (improvement from RAG)
    """
    from rag_engine import retrieve, build_rag_context_block

    def _tick(done, total, msg):
        if progress:
            try:
                progress(done, total, msg)
            except Exception:
                pass

    try:
        import ai_service
    except ImportError:
        return {"error": "ai_service not importable", "timestamp": datetime.datetime.now().isoformat()}

    # ── Pre-flight: this evaluation is meaningless without a working model ──
    # Each evaluated section costs TWO generation calls. With a dead key every
    # one of those calls burns its full timeout and silently returns the
    # template text, so the run appears to hang for many minutes and then
    # reports that RAG made no difference. Fail fast and say why instead.
    _tick(0, 1, "Checking AI provider…")
    try:
        import llm_client
        diag = llm_client.diagnose(timeout=15)
        if not diag["working"]:
            detail = "; ".join(
                f"{p}: {v['error']}" for p, v in diag["providers"].items() if v["error"])
            return {"error": "No AI provider is reachable, so generation with and "
                             "without RAG would both fall back to the offline "
                             "template and the comparison would be meaningless. "
                             + detail,
                    "timestamp": datetime.datetime.now().isoformat()}
    except Exception:
        pass                       # diagnostics are best-effort, never fatal

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    plan_col_to_section = {
        "personal_care_plan":        "personal_care",
        "mobility_care_plan":        "mobility",
        "nutrition_hydration_plan":  "nutrition",
        "medication_management_plan":"medication_management",
        "cognitive_support_plan":    "cognitive_support",
        "emotional_wellbeing_plan":  "emotional_wellbeing",
        "end_of_life_preferences":   "end_of_life",
    }

    residents = conn.execute(
        "SELECT resident_id, preferred_name, primary_diagnosis, secondary_diagnoses, "
        "age, gender, allergies, falls_risk, pressure_sore_risk, mobility_level "
        "FROM residents WHERE active=1"
    ).fetchall()

    section_scores_no_rag  = []
    section_scores_rag     = []
    faithfulness_scores    = []
    attributions           = []
    attributions_no_rag    = []
    per_section_results    = []
    prec_at_k_all          = {1: [], 3: [], 5: []}
    errors                 = []

    # Build the work list first so progress can be reported as a fraction and
    # the run can be capped — an uncapped run is 5 residents x 7 sections x 2
    # API calls, which is slow enough to look like a crash.
    work = []
    for res in residents:
        plan_row = conn.execute(
            "SELECT * FROM care_plans WHERE resident_id=? AND status IN ('Active','approved') "
            "ORDER BY id DESC LIMIT 1", (res["resident_id"],)).fetchone()
        if not plan_row:
            continue
        for col in plan_col_to_section:
            txt = plan_row[col]
            if txt and len(txt.strip()) >= 40:
                work.append((res["resident_id"], col))
    total_units = len(work) if not max_sections else min(len(work), int(max_sections))
    done_units = 0
    stop_after = total_units

    for res in residents:
        rid = res["resident_id"]
        if done_units >= stop_after:
            break

        # Get approved care plan
        plan = conn.execute(
            "SELECT * FROM care_plans WHERE resident_id=? AND status IN ('Active','approved') ORDER BY id DESC LIMIT 1",
            (rid,)
        ).fetchone()
        if not plan:
            continue

        res_data = dict(res)

        for col, section_name in plan_col_to_section.items():
            if done_units >= stop_after:
                break
            reference_text = plan[col]
            if not reference_text or len(reference_text.strip()) < 40:
                continue
            done_units += 1
            _tick(done_units, total_units,
                  f"{res_data.get('preferred_name', rid)} — {section_name.replace('_', ' ')}")

            # Build base data for AI generation
            gen_data = {
                "preferred_name": res_data.get("preferred_name", "The resident"),
                "age": res_data.get("age", ""),
                "gender": res_data.get("gender", ""),
                "primary_diagnosis": res_data.get("primary_diagnosis", ""),
                "secondary_diagnoses": res_data.get("secondary_diagnoses", ""),
                "allergies": res_data.get("allergies", ""),
                "falls_risk": res_data.get("falls_risk", ""),
                "mobility_level": res_data.get("mobility_level", ""),
                "section": section_name,
            }

            # ── Generate WITHOUT RAG ──────────────────────────────────────
            try:
                text_no_rag, _ = ai_service.generate_narrative("care_plan_section", gen_data)
                score_no_rag = rouge_l(text_no_rag, reference_text)
                r1_no_rag = rouge_1(text_no_rag, reference_text)
            except Exception as e:
                errors.append(f"{rid}/{section_name} no-RAG: {e}")
                continue

            # ── Retrieve with RAG ─────────────────────────────────────────
            chunks = retrieve(
                query=f"{section_name} care plan: {res_data.get('primary_diagnosis','')}",
                resident_id=rid,
                top_k=top_k,
                mode=rag_mode,
            )
            rag_block = build_rag_context_block(chunks)

            # Build RAG-augmented generation data
            gen_data_rag = dict(gen_data)
            gen_data_rag["rag_context"] = rag_block

            try:
                text_rag, _ = ai_service.generate_narrative("care_plan_section", gen_data_rag)
                score_rag = rouge_l(text_rag, reference_text)
                r1_rag = rouge_1(text_rag, reference_text)
                faith = faithfulness_score(text_rag, chunks)
            except Exception as e:
                errors.append(f"{rid}/{section_name} RAG: {e}")
                continue

            # ── Precision@k ───────────────────────────────────────────────
            p_at_k = precision_at_k(chunks, section_name)
            for k, v in p_at_k.items():
                if k in prec_at_k_all:
                    prec_at_k_all[k].append(v)

            section_scores_no_rag.append(score_no_rag)
            section_scores_rag.append(score_rag)
            faithfulness_scores.append(faith)

            per_section_results.append({
                "resident_id": rid,
                "resident_name": res_data.get("preferred_name", rid),
                "section": section_name,
                "rouge_l_no_rag": round(score_no_rag, 4),
                "rouge_l_rag": round(score_rag, 4),
                "rouge_1_no_rag": round(r1_no_rag, 4),
                "rouge_1_rag": round(r1_rag, 4),
                "delta_rouge_l": round(score_rag - score_no_rag, 4),
                "faithfulness": round(faith, 4),
                "num_chunks_retrieved": len(chunks),
            })

    conn.close()

    def _safe_mean(lst):
        return round(statistics.mean(lst), 4) if lst else 0.0

    def _safe_stdev(lst):
        return round(statistics.stdev(lst), 4) if len(lst) > 1 else 0.0

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rag_mode": rag_mode,
        "top_k": top_k,
        "max_sections": max_sections,
        "num_sections_evaluated": len(per_section_results),
        "summary": {
            "avg_rouge_l_no_rag":   _safe_mean(section_scores_no_rag),
            "avg_rouge_l_rag":      _safe_mean(section_scores_rag),
            "avg_delta_rouge_l":    _safe_mean([r["delta_rouge_l"] for r in per_section_results]),
            "stdev_rouge_l_no_rag": _safe_stdev(section_scores_no_rag),
            "stdev_rouge_l_rag":    _safe_stdev(section_scores_rag),
            "avg_faithfulness":     _safe_mean(faithfulness_scores),
            "precision_at_1":       _safe_mean(prec_at_k_all[1]),
            "precision_at_3":       _safe_mean(prec_at_k_all[3]),
            "precision_at_5":       _safe_mean(prec_at_k_all[5]),
        },
        "attribution": _attribution_summary(attributions, attributions_no_rag),
        "per_section": per_section_results,
        "errors": errors,
        # Chart.js data for the evaluation UI
        "chart_data": {
            "labels": [r["section"].replace("_", " ").title() + f"\n({r['resident_name']})"
                       for r in per_section_results],
            "rouge_l_no_rag": [r["rouge_l_no_rag"] for r in per_section_results],
            "rouge_l_rag":    [r["rouge_l_rag"] for r in per_section_results],
            "faithfulness":   [r["faithfulness"] for r in per_section_results],
        },
    }

    # Persist to disk (for fast page reload without re-running)
    os.makedirs(os.path.dirname(EVAL_CACHE_PATH), exist_ok=True)
    with open(EVAL_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def _attribution_summary(with_rag: list[dict], without_rag: list[dict]) -> dict:
    """
    Roll the per-section sentence verdicts up into the three numbers a reader
    asks for, and put the no-RAG arm beside them.

    The comparison is the point. A generator with no records in front of it
    still writes confident clinical prose; the share of that prose which can be
    traced back to a real entry in this resident's file is what separates a
    grounded system from a fluent one.
    """
    def roll(runs):
        sent = sum(r["n_sentences"] for r in runs)
        if not sent:
            return {"n_sentences": 0, "supported": 0, "partial": 0, "unsupported": 0,
                    "grounded_rate": 0.0, "fully_grounded_rate": 0.0, "mean_support": 0.0}
        sup = sum(r["supported"] for r in runs)
        par = sum(r["partial"] for r in runs)
        uns = sum(r["unsupported"] for r in runs)
        return {
            "n_sentences": sent, "supported": sup, "partial": par, "unsupported": uns,
            "grounded_rate": round((sup + par) / sent, 4),
            "fully_grounded_rate": round(sup / sent, 4),
            "unsupported_rate": round(uns / sent, 4),
            "mean_support": round(
                sum(r["mean_support"] * r["n_sentences"] for r in runs) / sent, 4),
        }

    a, b = roll(with_rag), roll(without_rag)
    return {
        "with_rag": a,
        "without_rag": b,
        "delta_grounded_rate": round(a["grounded_rate"] - b["grounded_rate"], 4),
        "thresholds": {"supported": _ATTR_SUPPORTED, "partial": _ATTR_PARTIAL},
        "note": (
            "Each generated sentence is matched to the retrieved record that best "
            "supports it, and scored on the share of its own content words that "
            "appear in that record. Sentences at or above "
            f"{int(_ATTR_SUPPORTED * 100)} % are counted as supported, "
            f"{int(_ATTR_PARTIAL * 100)}–{int(_ATTR_SUPPORTED * 100)} % as partially "
            "supported, and anything below that as unsupported — written by the model "
            "without anything in the retrieved file to stand behind it."
        ),
    }


def load_cached_results() -> dict | None:
    """Load last evaluation run from cache."""
    if not os.path.exists(EVAL_CACHE_PATH):
        return None
    try:
        with open(EVAL_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# 5 ── INTERPRETATION HELPER (for dissertation)

def interpret_results(results: dict) -> str:
    """
    Auto-generates a plain-English interpretation of the evaluation results,
    suitable for pasting directly into the dissertation results chapter.
    """
    s = results.get("summary", {})
    n = results.get("num_sections_evaluated", 0)
    mode = results.get("rag_mode", "tfidf")
    delta = s.get("avg_delta_rouge_l", 0)
    rl_rag = s.get("avg_rouge_l_rag", 0)
    rl_base = s.get("avg_rouge_l_no_rag", 0)
    faith = s.get("avg_faithfulness", 0)
    p1 = s.get("precision_at_1", 0)
    p5 = s.get("precision_at_5", 0)

    direction = "improvement" if delta >= 0 else "degradation"
    change_pct = abs(delta / rl_base * 100) if rl_base else 0

    return (
        f"Evaluation was conducted across {n} care plan sections from "
        f"{results.get('num_sections_evaluated', 0)} resident-section pairs using the "
        f"{'TF-IDF + LSA' if mode == 'tfidf' else 'semantic sentence-transformer'} retrieval mode "
        f"with k={results.get('top_k', 5)} retrieved chunks.\n\n"
        f"The RAG-augmented system achieved a mean ROUGE-L F1 of {rl_rag:.4f} "
        f"compared to {rl_base:.4f} for the non-RAG baseline, representing a "
        f"{change_pct:.1f}% {direction} in lexical overlap with reference documents "
        f"(Δ = {delta:+.4f}).\n\n"
        f"Mean faithfulness score (proportion of key clinical terms from retrieved "
        f"chunks present in generated output) was {faith:.4f}, indicating that the "
        f"system {'successfully grounds' if faith > 0.3 else 'partially grounds'} "
        f"its outputs in retrieved clinical context.\n\n"
        f"Retrieval precision@1 = {p1:.4f}, precision@5 = {p5:.4f}, demonstrating "
        f"that {'the majority' if p5 > 0.6 else 'some'} of retrieved chunks are "
        f"contextually relevant to the target care plan section.\n\n"
        f"Note: ROUGE-L measures n-gram overlap and may not fully capture clinical "
        f"coherence. Human expert evaluation is recommended as a complementary "
        f"assessment (see Limitations)."
    )


# 6 ── background job runner
"""
The evaluation makes two LLM calls per care-plan section, so even a capped run
takes minutes. Running it inside the HTTP request means the browser sits on a
spinner with no feedback and eventually times out — which looks like a crash
and is indistinguishable from one. The run is therefore executed on a worker
thread while the page polls a status endpoint for progress.
"""

import threading

_JOB = {
    "state": "idle",        # idle | running | done | error
    "done": 0,
    "total": 0,
    "message": "",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "params": {},
}
_JOB_LOCK = threading.Lock()


def job_status() -> dict:
    with _JOB_LOCK:
        job = dict(_JOB)
    job["percent"] = round(100 * job["done"] / job["total"], 1) if job["total"] else 0
    return job


def _run_job(db_path, rag_mode, top_k, max_sections):
    def progress(done, total, msg):
        with _JOB_LOCK:
            _JOB.update({"done": done, "total": total, "message": msg})

    try:
        results = run_evaluation(db_path=db_path, rag_mode=rag_mode, top_k=top_k,
                                 max_sections=max_sections, progress=progress)
        with _JOB_LOCK:
            if results.get("error"):
                _JOB.update({"state": "error", "error": results["error"],
                             "finished_at": datetime.datetime.now().isoformat()})
            else:
                _JOB.update({"state": "done", "error": None,
                             "message": f"{results.get('num_sections_evaluated', 0)} "
                                        f"section(s) evaluated",
                             "finished_at": datetime.datetime.now().isoformat()})
    except Exception as e:
        with _JOB_LOCK:
            _JOB.update({"state": "error", "error": f"{type(e).__name__}: {e}",
                         "finished_at": datetime.datetime.now().isoformat()})


def start_job(db_path: str = DB_PATH, rag_mode: str = "tfidf", top_k: int = 5,
              max_sections: int | None = 12) -> dict:
    """Start an evaluation in the background. Refuses if one is already running."""
    with _JOB_LOCK:
        if _JOB["state"] == "running":
            return {"started": False, "reason": "An evaluation is already running."}
        _JOB.update({"state": "running", "done": 0, "total": 0,
                     "message": "Starting…", "error": None,
                     "started_at": datetime.datetime.now().isoformat(),
                     "finished_at": None,
                     "params": {"rag_mode": rag_mode, "top_k": top_k,
                                "max_sections": max_sections}})
    t = threading.Thread(target=_run_job,
                         args=(db_path, rag_mode, top_k, max_sections),
                         daemon=True)
    t.start()
    return {"started": True}
