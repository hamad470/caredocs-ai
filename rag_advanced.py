"""
rag_advanced.py — Advanced Retrieval-Augmented Generation engine (v2)
This module upgrades the original `rag_engine.py` (single sparse TF-IDF
retriever, care-plan sections only) into a research-grade hybrid retrieval
pipeline suitable for conversational question answering over 12+ months of
care-home records.

WHAT MAKES IT "ADVANCED" (each stage is individually ablatable for the
dissertation's evaluation chapter):

  1. Contextual chunk enrichment  — every chunk is prefixed with a generated
     header ("Margaret Brown (RES001) | Care Note | 2026-06-12 | Night shift")
     before indexing. Retrieval quality on isolated chunks collapses without
     this because a chunk such as "declined breakfast, low mood" carries no
     resident, date or document type of its own.
     Ref: Anthropic (2024) "Contextual Retrieval".

  2. Sentence-aware chunking       — chunks never split mid-sentence; a
     configurable word budget with overlap preserves cross-sentence context.

  3. Hybrid retrieval              — a lexical BM25 index (Robertson &
     Zaragoza, 2009) runs alongside a dense vector index (TF-IDF → TruncatedSVD
     latent-semantic projection → FAISS inner-product). Lexical retrieval wins
     on names, drug names, MRN-style IDs and rare terms; dense retrieval wins
     on paraphrase ("wasn't eating" vs "poor appetite"). Care records need both.

  4. Reciprocal Rank Fusion (RRF)  — the two ranked lists are merged with
     RRF (Cormack et al., 2009), score = Σ 1/(k + rank). Rank-based fusion
     needs no score normalisation between two incomparable scoring functions.

  5. Metadata pre-filtering        — resident, date window and document type
     are applied as a hard mask BEFORE the vector search (FAISS IDSelector),
     never as a post-filter. Cross-resident leakage is a clinical-safety and
     UK GDPR failure mode, so it is made architecturally impossible rather
     than merely unlikely.

  6. Recency weighting             — exponential time decay, because in a care
     setting "what is happening with her mobility" almost always means now,
     not eighteen months ago.

  7. MMR diversification           — Maximal Marginal Relevance (Carbonell &
     Goldstein, 1998) removes near-duplicate chunks, which matter here because
     daily care notes are highly repetitive; without it the top-k is often six
     paraphrases of the same sentence.

  8. Optional neural upgrade       — if `sentence-transformers` is installed
     the dense stage transparently switches to all-MiniLM-L6-v2 embeddings and,
     if present, a cross-encoder reranker. Nothing else in the pipeline changes,
     which is what makes the sparse/dense comparison a fair ablation.

No new hard dependencies: numpy + scikit-learn + (optional) faiss only.
"""

from __future__ import annotations

import os
import re
import json
import math
import pickle
import sqlite3
import hashlib
import datetime
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ── Optional accelerators / upgrades ────────────────────────────────────────
try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False


# Configuration

INDEX_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_index_v2")
STORE_PATH  = os.path.join(INDEX_DIR, "store.pkl")       # vectorizer, svd, bm25, matrix
META_PATH   = os.path.join(INDEX_DIR, "chunks.json")     # chunk text + metadata
FAISS_PATH  = os.path.join(INDEX_DIR, "dense.faiss")
MANIFEST    = os.path.join(INDEX_DIR, "manifest.json")

CHUNK_WORDS      = 130     # target words per chunk (~170 tokens)
CHUNK_OVERLAP    = 35      # word overlap between neighbouring chunks
LSA_DIMS         = 256     # TruncatedSVD output dimensionality
RRF_K            = 60      # RRF smoothing constant (Cormack et al., 2009)
CANDIDATE_POOL   = 40      # chunks entering fusion from each retriever
MMR_LAMBDA       = 0.72    # 1.0 = pure relevance, 0.0 = pure diversity
RECENCY_TAU_DAYS = 240.0   # exponential decay constant for time weighting
RECENCY_WEIGHT   = 0.10    # how much recency can lift a fused score
# NOTE (v3). This was 0.35 until the ablation in rag_experiments.py measured it.
# At 0.35 the recency stage COST 0.120 MRR on known-item retrieval (0.431 ->
# 0.311): it promotes recent chunks over the actually-relevant one whenever the
# query is about something older. The measured sweep is monotone — every gram of
# recency costs ranking accuracy — so the value is now 0.10, where the cost is
# 0.026 MRR and Recall@8 is UNCHANGED at 0.583. That is the honest trade: the
# stage buys clinical currency (a nurse asking "how is she doing" means now) and
# it is paid for in rank position, not in whether the evidence is found at all.
# The full sweep is reported in Chapter 6 as a parameter chosen by measurement
# rather than by taste.

# Set False to index chunk bodies WITHOUT the generated context header, which is
# how the contextual-enrichment ablation (E2) is run. Production is always True.
USE_CONTEXT_HEADER = True
ST_MODEL_NAME    = "all-MiniLM-L6-v2"

SOURCE_LABELS = {
    "care_note":    "Care Note",
    "incident":     "Incident Report",
    "care_plan":    "Care Plan",
    "wellbeing":    "Wellbeing Assessment",
    "risk":         "Risk Assessment",
    "handover":     "Shift Handover",
    "family_comm":  "Family Communication",
    "medication":   "Medication Record",
    "profile":      "Resident Profile",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "was",
    "were", "is", "are", "be", "been", "at", "as", "by", "that", "this", "it",
    "her", "his", "she", "he", "they", "them", "has", "had", "have", "from",
    "but", "not", "no", "did", "does", "do", "we", "i", "you", "there", "their",
}


# 1 ── BM25 LEXICAL INDEX  (pure-python, no extra dependency)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed, 2-char minimum."""
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if len(t) > 1 and t not in _STOPWORDS]


class BM25:
    """
    Okapi BM25 (Robertson & Zaragoza, 2009).

        score(D,Q) = Σ_{q∈Q} IDF(q) · f(q,D)·(k1+1) / (f(q,D) + k1·(1-b+b·|D|/avgdl))

    Implemented directly rather than via `rank_bm25` so the project keeps a
    short dependency list and the scoring function is inspectable in the
    dissertation appendix. An inverted index (term → {doc: tf}) keeps scoring
    proportional to the number of query terms rather than corpus size.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        self.inverted: dict[str, dict[int, int]] = defaultdict(dict)
        for idx, tokens in enumerate(corpus_tokens):
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, freq in tf.items():
                self.inverted[tok][idx] = freq
        self.inverted = dict(self.inverted)

        # Robertson-Sparck-Jones IDF with +0.5 smoothing, floored at a small
        # positive value so that very common terms cannot contribute negatively.
        self.idf: dict[str, float] = {}
        for tok, postings in self.inverted.items():
            df = len(postings)
            self.idf[tok] = max(
                math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0), 0.01
            )

    # ── persistence ──────────────────────────────────────────────────────
    # The index is stored as plain dicts rather than a pickled BM25 instance,
    # so that a file written by `python rag_advanced.py` (module __main__) can
    # still be read by the Flask app (module rag_advanced). Pickling class
    # instances across differing __main__ contexts is a classic silent break.

    def to_state(self) -> dict:
        return {"k1": self.k1, "b": self.b, "n_docs": self.n_docs,
                "doc_len": self.doc_len, "avgdl": self.avgdl,
                "inverted": self.inverted, "idf": self.idf}

    @classmethod
    def from_state(cls, state: dict) -> "BM25":
        obj = cls.__new__(cls)
        obj.k1 = state["k1"]
        obj.b = state["b"]
        obj.n_docs = state["n_docs"]
        obj.doc_len = state["doc_len"]
        obj.avgdl = state["avgdl"]
        obj.inverted = state["inverted"]
        obj.idf = state["idf"]
        return obj

    def scores(self, query_tokens: list[str], allowed: np.ndarray | None = None) -> dict[int, float]:
        """Return {doc_index: score} for docs matching ≥1 query term."""
        allowed_set = set(allowed.tolist()) if allowed is not None else None
        out: dict[int, float] = defaultdict(float)
        for tok in query_tokens:
            postings = self.inverted.get(tok)
            if not postings:
                continue
            idf = self.idf[tok]
            for doc_idx, freq in postings.items():
                if allowed_set is not None and doc_idx not in allowed_set:
                    continue
                dl = self.doc_len[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                out[doc_idx] += idf * (freq * (self.k1 + 1)) / denom
        return dict(out)


# 2 ── DOCUMENT LOADING  (every table that carries narrative meaning)

def _clean(*parts) -> str:
    return " ".join(str(p).strip() for p in parts if p and str(p).strip())


def _residents_map(conn) -> dict[str, dict]:
    out = {}
    for r in conn.execute("SELECT * FROM residents"):
        d = dict(r)
        out[d["resident_id"]] = d
    return out


def load_documents(db_path: str) -> list[dict]:
    """
    Read every narrative-bearing record out of SQLite and return a flat list of
    documents. Structured fields (fluid intake, mood, pain, scores) are folded
    into the narrative text so that a numeric fact such as "450 ml" is
    retrievable by the same index that serves free text.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    residents = _residents_map(conn)
    docs: list[dict] = []

    def add(text, resident_id, source_type, source_id, section, date, extra=None):
        if not text or len(text.strip()) < 15:
            return
        docs.append({
            "text": text.strip(),
            "resident_id": resident_id or "",
            "resident_name": (residents.get(resident_id, {}) or {}).get("full_name", ""),
            "source_type": source_type,
            "source_id": source_id,
            "section": section,
            "date": (date or "")[:10],
            "extra": extra or {},
        })

    # ── Resident profiles ────────────────────────────────────────────────
    for rid, r in residents.items():
        profile = _clean(
            f"{r.get('full_name')} (known as {r.get('preferred_name')}), age {r.get('age')},",
            f"room {r.get('room_number')}, admitted {r.get('admission_date')}.",
            f"Care type: {r.get('care_type')}.",
            f"Primary diagnosis: {r.get('primary_diagnosis')}.",
            f"Secondary diagnoses: {r.get('secondary_diagnoses')}.",
            f"Allergies: {r.get('allergies')}.",
            f"Current medications: {r.get('medications_summary')}.",
            f"Mobility: {r.get('mobility_level')}. Continence: {r.get('continence_needs')}.",
            f"Diet texture: {r.get('nutrition_texture')}. Dietary restrictions: {r.get('dietary_restrictions')}.",
            f"Falls risk: {r.get('falls_risk')}. Pressure sore risk: {r.get('pressure_sore_risk')}.",
            f"DNACPR status: {r.get('dnacpr_status')}. Mental capacity: {r.get('mental_capacity')}.",
            f"GP: {r.get('gp_name')}. Next of kin: {r.get('nok_name')} ({r.get('nok_relationship')}).",
            f"Key worker: {r.get('key_worker')}.",
        )
        add(profile, rid, "profile", f"PROF-{rid}", "resident_profile",
            r.get("admission_date"))

    # ── Care notes ───────────────────────────────────────────────────────
    for row in conn.execute("SELECT * FROM care_notes"):
        r = dict(row)
        observations = _clean(
            f"Mood recorded as {r.get('mood')}." if r.get("mood") else "",
            f"Appetite {r.get('appetite')}." if r.get("appetite") else "",
            f"Fluid intake {r.get('fluid_intake_ml')} ml." if r.get("fluid_intake_ml") else "",
            f"Weight {r.get('weight_kg')} kg." if r.get("weight_kg") else "",
            f"Sleep quality {r.get('sleep_quality')}." if r.get("sleep_quality") else "",
            f"Pain observed: {r.get('pain_observed')} {r.get('pain_location') or ''}." if r.get("pain_observed") else "",
            f"Skin check: {r.get('skin_checked')} {r.get('skin_concern') or ''}." if r.get("skin_checked") else "",
            f"Falls this shift: {r.get('falls_this_shift')}." if r.get("falls_this_shift") else "",
            f"Activity: {r.get('activity')} {r.get('activity_description') or ''}." if r.get("activity") else "",
            f"Personal care: {r.get('personal_care')}." if r.get("personal_care") else "",
        )
        body = _clean(r.get("care_narrative"), observations,
                      f"Concerns: {r['concerns']}" if r.get("concerns") else "",
                      f"Actions taken: {r['actions_taken']}" if r.get("actions_taken") else "",
                      f"Handover: {r['handover_notes']}" if r.get("handover_notes") else "")
        add(body, r["resident_id"], "care_note", f"CN-{r['id']}",
            f"{r.get('note_type', 'daily')} / {r.get('shift', '')} shift", r.get("date"),
            {"staff": r.get("staff_name"), "shift": r.get("shift"),
             "status": r.get("status"), "ai_generated": r.get("ai_generated")})

    # ── Incidents ────────────────────────────────────────────────────────
    for row in conn.execute("SELECT * FROM incidents"):
        r = dict(row)
        body = _clean(
            f"{r.get('incident_type')} incident, severity {r.get('severity')},",
            f"at {r.get('location')} on {r.get('date')} {r.get('time') or ''}.",
            r.get("description"),
            f"Immediate actions: {r['immediate_actions']}" if r.get("immediate_actions") else "",
            f"Injuries: {r['injuries']}" if r.get("injuries") else "",
            f"Medical attention: {r['medical_attention']}" if r.get("medical_attention") else "",
            f"Outcome: {r['outcome']}" if r.get("outcome") else "",
            f"Lessons learned: {r['lessons_learned']}" if r.get("lessons_learned") else "",
            f"Preventative actions: {r['preventative_actions']}" if r.get("preventative_actions") else "",
            f"GP notified: {r.get('gp_notified')}. Family notified: {r.get('family_notified')}.",
            f"CQC notification: {r.get('cqc_notification')}. Status: {r.get('status')}.",
        )
        add(body, r["resident_id"], "incident",
            r.get("incident_id") or f"INC-{r['id']}", r.get("incident_type", "incident"),
            r.get("date"),
            {"severity": r.get("severity"), "status": r.get("status"),
             "incident_type": r.get("incident_type")})

    # ── Care plans (one document per section) ────────────────────────────
    plan_sections = [
        "personal_identity_summary", "mobility_care_plan", "personal_care_plan",
        "continence_care_plan", "nutrition_hydration_plan", "medication_management_plan",
        "cognitive_support_plan", "emotional_wellbeing_plan", "social_activity_plan",
        "end_of_life_preferences", "risk_summary", "goals_of_care",
        "family_involvement_plan",
    ]
    for row in conn.execute("SELECT * FROM care_plans"):
        r = dict(row)
        for sec in plan_sections:
            add(r.get(sec), r["resident_id"], "care_plan", f"CP-{r['id']}",
                sec.replace("_", " "), r.get("effective_from"),
                {"version": r.get("version"), "status": r.get("status")})

    # ── Wellbeing assessments ────────────────────────────────────────────
    for row in conn.execute("SELECT * FROM wellbeing"):
        r = dict(row)
        body = _clean(
            f"Wellbeing assessment covering {r.get('period_covered')}.",
            f"Overall score {r.get('overall_score')}/10 (physical {r.get('physical_health_score')},",
            f"mental {r.get('mental_health_score')}, social {r.get('social_engagement_score')},",
            f"personal care {r.get('personal_care_score')}, nutrition {r.get('nutrition_score')},",
            f"pain management {r.get('pain_management_score')}).",
            r.get("summary"),
            f"Physical: {r['physical_notes']}" if r.get("physical_notes") else "",
            f"Mental health: {r['mental_notes']}" if r.get("mental_notes") else "",
            f"Social: {r['social_notes']}" if r.get("social_notes") else "",
            f"Goals progress: {r['goals_progress']}" if r.get("goals_progress") else "",
            f"Concerns: {r['concerns']}" if r.get("concerns") else "",
            f"Positive outcomes: {r['positive_outcomes']}" if r.get("positive_outcomes") else "",
            f"Resident voice: {r['resident_voice']}" if r.get("resident_voice") else "",
            f"Family feedback: {r['family_feedback']}" if r.get("family_feedback") else "",
        )
        add(body, r["resident_id"], "wellbeing", f"WB-{r['id']}",
            "wellbeing assessment", r.get("assessment_date"),
            {"overall_score": r.get("overall_score")})

    # ── Risk assessments ─────────────────────────────────────────────────
    for row in conn.execute("SELECT * FROM risk_assessments"):
        r = dict(row)
        body = _clean(
            f"{r.get('assessment_type')} risk assessment scored {r.get('score')},",
            f"risk level {r.get('risk_level')}, assessed {r.get('date_assessed')}",
            f"by {r.get('assessed_by')}, review due {r.get('review_date')}.",
            r.get("narrative"),
            f"Risk factors: {r['risk_factors']}" if r.get("risk_factors") else "",
            f"Interventions: {r['interventions']}" if r.get("interventions") else "",
            f"Additional actions: {r['additional_actions']}" if r.get("additional_actions") else "",
            f"Outcome measures: {r['outcome_measures']}" if r.get("outcome_measures") else "",
        )
        add(body, r["resident_id"], "risk", f"RISK-{r['id']}",
            f"{r.get('assessment_type')} risk", r.get("date_assessed"),
            {"risk_level": r.get("risk_level"), "score": r.get("score"),
             "assessment_type": r.get("assessment_type")})

    # ── Handovers ────────────────────────────────────────────────────────
    try:
        for row in conn.execute("SELECT * FROM handovers"):
            r = dict(row)
            body = _clean(
                f"Handover from {r.get('shift_ending')} to {r.get('shift_starting')} shift",
                f"compiled by {r.get('compiled_by')}.",
                r.get("overall_summary"),
                f"Care completed: {r['care_completed']}" if r.get("care_completed") else "",
                f"Concerns for next shift: {r['concerns_next_shift']}" if r.get("concerns_next_shift") else "",
                f"Outstanding tasks: {r['outstanding_tasks']}" if r.get("outstanding_tasks") else "",
                f"Medication notes: {r['medication_notes']}" if r.get("medication_notes") else "",
                f"Escalation required: {r.get('escalation_required')} {r.get('escalation_details') or ''}",
            )
            add(body, r["resident_id"], "handover", f"HO-{r['id']}",
                f"{r.get('shift_ending')}→{r.get('shift_starting')} handover", r.get("date"),
                {"escalation": r.get("escalation_required")})
    except sqlite3.Error:
        pass

    # ── Family communications ────────────────────────────────────────────
    try:
        for row in conn.execute("SELECT * FROM family_comms"):
            r = dict(row)
            body = _clean(
                f"{r.get('direction')} {r.get('comm_type')} with {r.get('family_contact')}",
                f"handled by {r.get('staff_member')}. Subject: {r.get('subject')}.",
                f"Trigger: {r.get('trigger_event')}." if r.get("trigger_event") else "",
                r.get("body"),
                f"Family response: {r['family_response']}" if r.get("family_response") else "",
                f"Follow-up: {r.get('follow_up')} {r.get('follow_up_actions') or ''}",
            )
            add(body, r["resident_id"], "family_comm", f"FC-{r['id']}",
                f"{r.get('comm_type')} with family", r.get("date"),
                {"comm_type": r.get("comm_type")})
    except sqlite3.Error:
        pass

    # ── Medications (one document per prescription) ──────────────────────
    try:
        for row in conn.execute("SELECT * FROM medications"):
            r = dict(row)
            body = _clean(
                f"{r.get('medication_name')} ({r.get('generic_name')}) {r.get('dose')}",
                f"{r.get('route')} {r.get('frequency')}.",
                f"Indication: {r.get('indication')}.",
                f"Prescribed by {r.get('prescribing_gp')}, started {r.get('start_date')},",
                f"review {r.get('review_date')}. Status: {r.get('status')}.",
                "Controlled drug." if r.get("is_controlled") else "",
                f"PRN — {r.get('prn_instructions')}" if r.get("is_prn") else "",
                f"Administration notes: {r['admin_notes']}" if r.get("admin_notes") else "",
                f"Side effects to monitor: {r['side_effects']}" if r.get("side_effects") else "",
                f"Stopped because: {r['stopped_reason']}" if r.get("stopped_reason") else "",
            )
            add(body, r["resident_id"], "medication", f"MED-{r['id']}",
                r.get("medication_name", "medication"), r.get("start_date"),
                {"status": r.get("status"), "is_prn": r.get("is_prn")})
    except sqlite3.Error:
        pass

    conn.close()
    return docs


# 3 ── SENTENCE-AWARE CHUNKING WITH CONTEXTUAL HEADERS

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def context_header(doc: dict) -> str:
    """
    The contextual prefix attached to every chunk of a document.
    This is what lets a lone sentence ("refused lunch again") still be found by
    a query like "Dorothy's appetite in March 2026".
    """
    label = SOURCE_LABELS.get(doc["source_type"], doc["source_type"])
    bits = [doc.get("resident_name") or doc.get("resident_id"), label]
    if doc.get("date"):
        bits.append(doc["date"])
    if doc.get("section"):
        bits.append(str(doc["section"]))
    return " | ".join(b for b in bits if b)


def chunk_document(doc: dict) -> list[dict]:
    """Sentence-aware chunking with word budget, overlap and context headers."""
    sentences = _split_sentences(doc["text"])
    if not sentences:
        return []

    header = context_header(doc)
    chunks, current, current_words, idx = [], [], 0, 0

    def flush(sents):
        nonlocal idx
        body = " ".join(sents).strip()
        if len(body) < 25:
            return
        chunks.append({
            "chunk_id": f"{doc['source_id']}#c{idx}",
            "text": body,                                 # raw text shown to the LLM
            "indexed_text": (f"{header}. {body}" if USE_CONTEXT_HEADER else body),
            "header": header,
            "resident_id": doc["resident_id"],
            "resident_name": doc["resident_name"],
            "source_type": doc["source_type"],
            "source_id": doc["source_id"],
            "section": doc["section"],
            "date": doc["date"],
            "extra": doc.get("extra", {}),
        })
        idx += 1

    for sent in sentences:
        w = len(sent.split())
        if current_words + w > CHUNK_WORDS and current:
            flush(current)
            # carry an overlap tail into the next chunk
            tail, tw = [], 0
            for s in reversed(current):
                sw = len(s.split())
                if tw + sw > CHUNK_OVERLAP:
                    break
                tail.insert(0, s)
                tw += sw
            current, current_words = list(tail), tw
        current.append(sent)
        current_words += w

    if current:
        flush(current)
    return chunks


# 4 ── index build

def build_index(db_path: str, use_neural: bool | None = None, verbose: bool = False) -> dict:
    """
    Build and persist the hybrid index.

    use_neural=None  → automatic (neural if sentence-transformers is installed)
    use_neural=False → force TF-IDF + LSA (fast, deterministic, offline)
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = load_documents(db_path)
    if not docs:
        return {"error": "No documents found in the database.", "num_chunks": 0}

    chunks: list[dict] = []
    for d in docs:
        chunks.extend(chunk_document(d))
    if not chunks:
        return {"error": "Chunking produced no output.", "num_chunks": 0}

    indexed_texts = [c["indexed_text"] for c in chunks]

    # ── lexical index ────────────────────────────────────────────────────
    corpus_tokens = [tokenize(t) for t in indexed_texts]
    bm25 = BM25(corpus_tokens)

    # ── dense index ──────────────────────────────────────────────────────
    neural = _ST_OK if use_neural is None else (use_neural and _ST_OK)
    vectorizer = svd = st_model = None

    if neural:
        st_model = SentenceTransformer(ST_MODEL_NAME)
        dense = np.asarray(
            st_model.encode(indexed_texts, batch_size=64,
                            show_progress_bar=verbose, normalize_embeddings=True),
            dtype="float32")
        dense_backend = f"sentence-transformers/{ST_MODEL_NAME}"
    else:
        vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1, 2),
                                     min_df=1, sublinear_tf=True,
                                     strip_accents="unicode")
        tfidf = vectorizer.fit_transform(indexed_texts)
        n_comp = int(min(LSA_DIMS, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        n_comp = max(n_comp, 2)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        dense = normalize(svd.fit_transform(tfidf)).astype("float32")
        dense_backend = f"TF-IDF + LSA ({n_comp}d)"

    # ── FAISS store (exact inner-product search over unit vectors = cosine)
    if _FAISS_OK:
        index = faiss.IndexFlatIP(dense.shape[1])
        index.add(dense)
        faiss.write_index(index, FAISS_PATH)

    with open(STORE_PATH, "wb") as f:
        pickle.dump({"bm25_state": bm25.to_state(), "vectorizer": vectorizer,
                     "svd": svd, "dense": dense, "neural": neural}, f)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    dates = sorted({c["date"] for c in chunks if c["date"]})
    manifest = {
        "built": True,
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "num_residents": len({c["resident_id"] for c in chunks if c["resident_id"]}),
        "dense_backend": dense_backend,
        "lexical_backend": "BM25 (Okapi, k1=1.5, b=0.75)",
        "fusion": f"Reciprocal Rank Fusion (k={RRF_K}) + MMR (λ={MMR_LAMBDA})",
        "faiss": _FAISS_OK,
        "neural_available": _ST_OK,
        "dims": int(dense.shape[1]),
        "date_range": [dates[0], dates[-1]] if dates else [None, None],
        "source_types": sorted({c["source_type"] for c in chunks}),
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    invalidate_cache()
    return manifest


# 5 ── INDEX LOADING (cached)

_CACHE: dict = {}


def invalidate_cache() -> None:
    _CACHE.clear()


def _load() -> dict | None:
    """Load index artefacts once per process."""
    if _CACHE.get("ready"):
        return _CACHE
    if not (os.path.exists(STORE_PATH) and os.path.exists(META_PATH)):
        return None

    with open(META_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(STORE_PATH, "rb") as f:
        store = pickle.load(f)

    faiss_index = None
    if _FAISS_OK and os.path.exists(FAISS_PATH):
        try:
            faiss_index = faiss.read_index(FAISS_PATH)
        except Exception:
            faiss_index = None

    # Pre-compute per-chunk day numbers for O(1) date filtering and decay.
    today = datetime.date.today()
    ages = np.zeros(len(chunks), dtype=np.float32)
    day_ord = np.zeros(len(chunks), dtype=np.int64)
    for i, c in enumerate(chunks):
        try:
            d = datetime.date.fromisoformat(c["date"])
            day_ord[i] = d.toordinal()
            ages[i] = max((today - d).days, 0)
        except Exception:
            day_ord[i] = 0
            ages[i] = 9999

    _CACHE.update({
        "ready": True,
        "chunks": chunks,
        "bm25": (BM25.from_state(store["bm25_state"]) if "bm25_state" in store
                 else store["bm25"]),
        "vectorizer": store.get("vectorizer"),
        "svd": store.get("svd"),
        "dense": store["dense"],
        "neural": store.get("neural", False),
        "faiss": faiss_index,
        "ages": ages,
        "day_ord": day_ord,
        "st_model": None,
        "resident_ids": np.array([c["resident_id"] for c in chunks]),
        "source_types": np.array([c["source_type"] for c in chunks]),
    })
    return _CACHE


def _embed_query(cache: dict, query: str) -> np.ndarray:
    if cache["neural"]:
        if cache["st_model"] is None:
            cache["st_model"] = SentenceTransformer(ST_MODEL_NAME)
        return np.asarray(cache["st_model"].encode([query], normalize_embeddings=True),
                          dtype="float32")
    q = cache["vectorizer"].transform([query])
    return normalize(cache["svd"].transform(q)).astype("float32")


# 6 ── RETRIEVAL:  filter → BM25 ∥ dense → RRF → recency → MMR

def _candidate_mask(cache: dict, resident_ids, source_types, date_from, date_to,
                    exclude_source_types) -> np.ndarray:
    """Hard metadata pre-filter. Returns array of allowed chunk indices."""
    n = len(cache["chunks"])
    mask = np.ones(n, dtype=bool)

    if resident_ids:
        wanted = set(resident_ids if isinstance(resident_ids, (list, tuple, set))
                     else [resident_ids])
        mask &= np.isin(cache["resident_ids"], list(wanted))

    if source_types:
        mask &= np.isin(cache["source_types"], list(source_types))
    if exclude_source_types:
        mask &= ~np.isin(cache["source_types"], list(exclude_source_types))

    if date_from:
        try:
            lo = datetime.date.fromisoformat(str(date_from)[:10]).toordinal()
            mask &= (cache["day_ord"] >= lo) | (cache["day_ord"] == 0)
        except Exception:
            pass
    if date_to:
        try:
            hi = datetime.date.fromisoformat(str(date_to)[:10]).toordinal()
            mask &= (cache["day_ord"] <= hi) | (cache["day_ord"] == 0)
        except Exception:
            pass

    return np.flatnonzero(mask)


def _dense_ranking(cache: dict, q_vec: np.ndarray, allowed: np.ndarray,
                   pool: int) -> list[tuple[int, float]]:
    """
    Top-`pool` dense hits restricted to `allowed`.

    When FAISS is present the restriction is pushed into the search itself via
    an IDSelector, so excluded residents are never scored — the isolation
    guarantee holds at the index level, not as a post-hoc filter.
    """
    if allowed.size == 0:
        return []
    k = int(min(pool, allowed.size))

    if cache["faiss"] is not None:
        try:
            sel = faiss.IDSelectorArray(allowed.astype("int64"))
            params = faiss.SearchParameters()
            params.sel = sel
            scores, ids = cache["faiss"].search(q_vec, k, params=params)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        except Exception:
            pass  # fall through to numpy

    sub = cache["dense"][allowed]                       # (m, d)
    sims = (sub @ q_vec[0]).astype(np.float32)          # cosine, unit vectors
    top = np.argpartition(-sims, k - 1)[:k] if k < sims.size else np.arange(sims.size)
    top = top[np.argsort(-sims[top])]
    return [(int(allowed[t]), float(sims[t])) for t in top]


def _mmr(cache: dict, q_vec: np.ndarray, candidates: list[int],
         top_k: int, lambda_: float = MMR_LAMBDA) -> list[int]:
    """
    Maximal Marginal Relevance (Carbonell & Goldstein, 1998):
        MMR = argmax [ λ·sim(c,q) − (1−λ)·max_{s∈selected} sim(c,s) ]
    Prevents six paraphrases of the same daily note filling the context window.
    """
    if not candidates:
        return []
    vecs = cache["dense"][candidates]
    rel = vecs @ q_vec[0]
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])
        else:
            sel_vecs = vecs[selected]
            best, best_score = None, -1e9
            for i in remaining:
                redundancy = float(np.max(sel_vecs @ vecs[i]))
                score = lambda_ * float(rel[i]) - (1 - lambda_) * redundancy
                if score > best_score:
                    best, best_score = i, score
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]


def retrieve(query: str,
             resident_ids=None,
             top_k: int = 8,
             date_from: str | None = None,
             date_to: str | None = None,
             source_types: list[str] | None = None,
             exclude_source_types: list[str] | None = None,
             use_recency: bool = True,
             use_mmr: bool = True,
             use_hybrid: bool = True,
             retriever: str = "hybrid",
             trace: dict | None = None) -> list[dict]:
    """
    Hybrid retrieval over the care record.

    Returns a list of dicts:
        text, score, chunk_id, source_id, source_type, section, date,
        resident_id, resident_name, bm25_rank, dense_rank, header

    `trace` (optional dict) is populated with per-stage diagnostics so the UI
    can show exactly how an answer was assembled — the transparency requirement
    that makes an AI clinical-documentation tool defensible to a CQC inspector.
    """
    cache = _load()
    if cache is None or not query or not query.strip():
        return []

    # `retriever` selects the ablation arm: "hybrid" (both), "dense" (vector
    # only) or "bm25" (lexical only). `use_hybrid=False` is kept as the legacy
    # spelling of "dense" so existing call sites keep working.
    retriever = (retriever or "hybrid").lower()
    if not use_hybrid:
        retriever = "dense"
    use_lexical = retriever in ("hybrid", "bm25")
    use_dense = retriever in ("hybrid", "dense")

    allowed = _candidate_mask(cache, resident_ids, source_types,
                              date_from, date_to, exclude_source_types)
    if trace is not None:
        trace["candidates_after_filter"] = int(allowed.size)
        trace["corpus_size"] = len(cache["chunks"])
    if allowed.size == 0:
        return []

    # ── Stage 1: lexical (BM25) ──────────────────────────────────────────
    bm25_ranked: list[tuple[int, float]] = []
    if use_lexical:
        bm25_scores = cache["bm25"].scores(tokenize(query), allowed)
        bm25_ranked = sorted(bm25_scores.items(), key=lambda kv: -kv[1])[:CANDIDATE_POOL]

    # ── Stage 2: dense (LSA/neural + FAISS) ──────────────────────────────
    # The query vector is always computed: MMR needs it even in the BM25-only
    # arm, where it is used for diversity rather than for candidate generation.
    q_vec = _embed_query(cache, query)
    dense_ranked = _dense_ranking(cache, q_vec, allowed, CANDIDATE_POOL) if use_dense else []

    # ── Stage 3: Reciprocal Rank Fusion ──────────────────────────────────
    fused: dict[int, float] = defaultdict(float)
    bm25_rank_of, dense_rank_of = {}, {}
    for rank, (idx, _s) in enumerate(bm25_ranked, start=1):
        fused[idx] += 1.0 / (RRF_K + rank)
        bm25_rank_of[idx] = rank
    for rank, (idx, _s) in enumerate(dense_ranked, start=1):
        fused[idx] += 1.0 / (RRF_K + rank)
        dense_rank_of[idx] = rank

    if not fused:
        return []

    # ── Stage 4: recency weighting ───────────────────────────────────────
    if use_recency:
        ages = cache["ages"]
        for idx in list(fused):
            decay = math.exp(-float(ages[idx]) / RECENCY_TAU_DAYS)
            fused[idx] *= (1.0 + RECENCY_WEIGHT * decay)

    ordered = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]

    # Exact-duplicate suppression before selection (not after), so that
    # near-identical daily notes do not consume top-k slots.
    deduped, seen_fp = [], set()
    for idx in ordered:
        fp = hashlib.md5(cache["chunks"][idx]["text"][:120].encode()).hexdigest()
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        deduped.append(idx)
    ordered = deduped

    # ── Stage 5: MMR diversification ─────────────────────────────────────
    pool = ordered[: max(top_k * 4, 20)]
    final_ids = _mmr(cache, q_vec, pool, top_k) if use_mmr else ordered[:top_k]

    if trace is not None:
        trace.update({
            "bm25_hits": len(bm25_ranked),
            "dense_hits": len(dense_ranked),
            "fused_pool": len(fused),
            "returned": len(final_ids),
            "recency_weighting": use_recency,
            "mmr": use_mmr,
            "hybrid": retriever == "hybrid",
            "retriever": retriever,
            "backend": "neural" if cache["neural"] else "tfidf-lsa",
            "faiss": cache["faiss"] is not None,
        })

    results = []
    for idx in final_ids:
        c = cache["chunks"][idx]
        results.append({
            "text": c["text"],
            "header": c["header"],
            "score": round(float(fused[idx]), 6),
            "chunk_id": c["chunk_id"],
            "source_id": c["source_id"],
            "source_type": c["source_type"],
            "source_label": SOURCE_LABELS.get(c["source_type"], c["source_type"]),
            "section": c["section"],
            "date": c["date"],
            "resident_id": c["resident_id"],
            "resident_name": c["resident_name"],
            "bm25_rank": bm25_rank_of.get(idx),
            "dense_rank": dense_rank_of.get(idx),
            "extra": c.get("extra", {}),
        })
    return results


# 7 ── CONTEXT BLOCK FOR THE LLM

def build_context_block(chunks: list[dict], max_chars: int = 9000) -> str:
    """
    Render retrieved chunks as a numbered evidence block. Each carries an [S#]
    handle that the model is instructed to cite, which is what turns a fluent
    answer into an auditable one.
    """
    if not chunks:
        return "(no matching records were retrieved)"
    out, used = [], 0
    for i, c in enumerate(chunks, 1):
        head = (f"[S{i}] {c['source_label']} — {c['resident_name'] or c['resident_id']}"
                f" — {c['date']} — {c['section']} (ref {c['source_id']})")
        body = c["text"]
        block = f"{head}\n{body}"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


# 8 ── status

def get_status() -> dict:
    """Index manifest for the admin/knowledge-base UI."""
    if not os.path.exists(MANIFEST):
        return {"built": False, "num_chunks": 0, "num_docs": 0,
                "neural_available": _ST_OK, "faiss": _FAISS_OK}
    try:
        with open(MANIFEST, "r") as f:
            data = json.load(f)
        data["built"] = os.path.exists(STORE_PATH) and os.path.exists(META_PATH)
        data["neural_available"] = _ST_OK
        return data
    except Exception:
        return {"built": False, "num_chunks": 0, "num_docs": 0,
                "neural_available": _ST_OK, "faiss": _FAISS_OK}


if __name__ == "__main__":       # quick CLI:  python rag_advanced.py [db_path]
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "carehome.db")
    print(json.dumps(build_index(db, verbose=True), indent=2))
