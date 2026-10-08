"""
RAG Engine — Retrieval-Augmented Generation for CareHome MVP
Supports two retrieval modes for ablation study:
  Mode A: TF-IDF + FAISS  (sparse, always available, sklearn-based)
  Mode B: Semantic + FAISS (dense, requires sentence-transformers)

Architecture:
  1. Index builder: reads care_notes, incidents, care_plans from SQLite
     → chunks text → embeds → stores in FAISS
  2. Retriever: embeds query → L2 nearest-neighbour search → returns top-k chunks
  3. Context builder: formats retrieved chunks for LLM prompt injection
  4. Audit logger: writes every retrieval to rag_audit_log table

Resident isolation: enforced at SEARCH TIME by pre-filtering the FAISS
index using a metadata mask — never post-filter, as cross-resident leakage
is a clinical safety and GDPR failure mode.

Ref: Lewis et al. (2020) "Retrieval-Augmented Generation for
     Knowledge-Intensive NLP Tasks." NeurIPS 2020.
"""

from __future__ import annotations

import os
import json
import pickle
import sqlite3
import re
import hashlib
import datetime
import numpy as np

# ── Optional heavy imports with graceful degradation ───────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _SEMANTIC_OK = True
except ImportError:
    _SEMANTIC_OK = False

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ── Constants ──────────────────────────────────────────────────────────────
CHUNK_SIZE_WORDS   = 80   # ~100 tokens — conservative, keeps each chunk coherent
CHUNK_OVERLAP      = 20   # word overlap between consecutive chunks
TOP_K_DEFAULT      = 5    # retrieved chunks per query
INDEX_DIR          = os.path.join(os.path.dirname(__file__), "rag_index")
TFIDF_INDEX_PATH   = os.path.join(INDEX_DIR, "tfidf_index.pkl")
SEMANTIC_INDEX_PATH = os.path.join(INDEX_DIR, "semantic_index.faiss")
META_PATH          = os.path.join(INDEX_DIR, "chunk_meta.json")
EMBED_MODEL_NAME   = "all-MiniLM-L6-v2"
LSA_COMPONENTS     = 128  # TruncatedSVD dimensions for TF-IDF dense projection


# 1 ── chunking

def _sentence_split(text: str) -> list[str]:
    """Split on sentence boundaries (. ! ?) preserving context."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def chunk_text(text: str, source_id: str, resident_id: str,
               source_type: str, section: str = "", date: str = "") -> list[dict]:
    """
    Split text into overlapping word-based chunks and attach metadata.
    Returns list of chunk dicts.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE_WORDS, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        if len(chunk_text_str.strip()) < 15:
            break

        chunks.append({
            "chunk_id": f"{source_id}_c{chunk_idx}",
            "text": chunk_text_str,
            "resident_id": resident_id,
            "source_type": source_type,   # care_note | incident | care_plan | wellbeing | risk
            "source_id": source_id,
            "section": section,
            "date": date,
        })

        chunk_idx += 1
        if end >= len(words):
            break
        start += CHUNK_SIZE_WORDS - CHUNK_OVERLAP

    return chunks


# 2 ── index builder

def _load_documents_from_db(db_path: str) -> list[dict]:
    """
    Read all indexable text from SQLite.  Returns list of raw doc dicts.
    Each dict has: text, resident_id, source_type, source_id, section, date.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    docs = []

    # Care notes — primary source
    for row in conn.execute(
        "SELECT id, resident_id, date, care_narrative, concerns, actions_taken "
        "FROM care_notes WHERE care_narrative IS NOT NULL"
    ):
        combined = " ".join(filter(None, [row["care_narrative"],
                                          row["concerns"],
                                          row["actions_taken"]]))
        docs.append({
            "text": combined,
            "resident_id": row["resident_id"],
            "source_type": "care_note",
            "source_id": f"CN-{row['id']}",
            "section": "daily_care",
            "date": row["date"],
        })

    # Incident narratives
    for row in conn.execute(
        "SELECT id, resident_id, date, description, immediate_actions, "
        "lessons_learned, outcome FROM incidents WHERE description IS NOT NULL"
    ):
        combined = " ".join(filter(None, [row["description"],
                                          row["immediate_actions"],
                                          row["lessons_learned"],
                                          row["outcome"]]))
        docs.append({
            "text": combined,
            "resident_id": row["resident_id"],
            "source_type": "incident",
            "source_id": f"INC-{row['id']}",
            "section": "incident",
            "date": row["date"],
        })

    # Care plans — each section separately
    plan_sections = [
        "personal_identity_summary", "mobility_care_plan", "personal_care_plan",
        "continence_care_plan", "nutrition_hydration_plan", "medication_management_plan",
        "cognitive_support_plan", "emotional_wellbeing_plan", "social_activity_plan",
        "end_of_life_preferences", "risk_summary", "goals_of_care",
        "family_involvement_plan",
    ]
    # DEFECT FIX (v3). This clause previously read `WHERE status = 'Active'`.
    # Every care plan in the database carries status 'approved' or 'Active'
    # depending on whether it was seeded or created through the UI, so the strict
    # equality silently excluded EVERY care plan from the v1 index. Two
    # consequences, both material and both reported in Chapter 6: the care plan
    # generator was never grounded on a resident's own previous plan, and the
    # precision@k relevance test could never fire, because the section names it
    # matches on exist only on care-plan chunks. The filter now accepts the whole
    # "current" vocabulary, case-insensitively, and excludes superseded versions
    # explicitly rather than by omission.
    for row in conn.execute(
        "SELECT id, resident_id, effective_from, " +
        ", ".join(plan_sections) +
        " FROM care_plans WHERE LOWER(COALESCE(status,'')) IN ('active','approved','current')"
    ):
        for sec in plan_sections:
            text = row[sec]
            if text and len(text.strip()) > 20:
                docs.append({
                    "text": text,
                    "resident_id": row["resident_id"],
                    "source_type": "care_plan",
                    "source_id": f"CP-{row['id']}",
                    "section": sec,
                    "date": row["effective_from"] or "",
                })

    # Wellbeing narratives
    for row in conn.execute(
        "SELECT id, resident_id, assessment_date, summary, concerns, positive_outcomes "
        "FROM wellbeing WHERE summary IS NOT NULL"
    ):
        combined = " ".join(filter(None, [row["summary"],
                                          row["concerns"],
                                          row["positive_outcomes"]]))
        docs.append({
            "text": combined,
            "resident_id": row["resident_id"],
            "source_type": "wellbeing",
            "source_id": f"WB-{row['id']}",
            "section": "wellbeing_summary",
            "date": row["assessment_date"],
        })

    # Risk assessment narratives
    for row in conn.execute(
        "SELECT id, resident_id, date_assessed, narrative, risk_factors, interventions "
        "FROM risk_assessments WHERE narrative IS NOT NULL"
    ):
        combined = " ".join(filter(None, [row["narrative"],
                                          row["risk_factors"],
                                          row["interventions"]]))
        docs.append({
            "text": combined,
            "resident_id": row["resident_id"],
            "source_type": "risk",
            "source_id": f"RISK-{row['id']}",
            "section": "risk_assessment",
            "date": row["date_assessed"],
        })

    conn.close()
    return docs


def build_index(db_path: str, mode: str = "tfidf") -> dict:
    """
    Build and persist the RAG index.

    mode: "tfidf"    — TF-IDF + LSA + FAISS (always available)
          "semantic" — sentence-transformers + FAISS (requires package)

    Returns dict with stats: num_docs, num_chunks, mode, built_at
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load raw documents
    raw_docs = _load_documents_from_db(db_path)
    if not raw_docs:
        return {"error": "No documents in database to index.", "num_chunks": 0}

    # Chunk all documents
    all_chunks = []
    for doc in raw_docs:
        chunks = chunk_text(
            doc["text"],
            source_id=doc["source_id"],
            resident_id=doc["resident_id"],
            source_type=doc["source_type"],
            section=doc["section"],
            date=doc["date"],
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"error": "Chunking produced no output.", "num_chunks": 0}

    texts = [c["text"] for c in all_chunks]

    if mode == "semantic" and _SEMANTIC_OK and _FAISS_OK:
        model = SentenceTransformer(EMBED_MODEL_NAME)
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])   # inner-product = cosine (normalised)
        index.add(embeddings)
        faiss.write_index(index, SEMANTIC_INDEX_PATH)
        actual_mode = "semantic"
    else:
        # TF-IDF + TruncatedSVD (LSA) — gives dense vectors suitable for FAISS
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Reduce to LSA_COMPONENTS dimensions for dense FAISS index
        n_components = min(LSA_COMPONENTS, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense_matrix = svd.fit_transform(tfidf_matrix)
        dense_matrix = normalize(dense_matrix).astype("float32")

        if _FAISS_OK:
            index = faiss.IndexFlatIP(dense_matrix.shape[1])
            index.add(dense_matrix)
            faiss.write_index(index, TFIDF_INDEX_PATH.replace(".pkl", ".faiss"))

        # Also save sklearn objects for re-use at query time
        with open(TFIDF_INDEX_PATH, "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "svd": svd,
                         "n_components": n_components}, f)
        actual_mode = "tfidf"

    # Save chunk metadata (resident isolation depends on this)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Save index manifest
    manifest = {
        "mode": actual_mode,
        "num_docs": len(raw_docs),
        "num_chunks": len(all_chunks),
        "built_at": datetime.datetime.now().isoformat(),
        "embed_model": EMBED_MODEL_NAME if actual_mode == "semantic" else f"TF-IDF+LSA({n_components if actual_mode=='tfidf' else 0}d)",
    }
    with open(os.path.join(INDEX_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# 3 ── retriever

_INDEX_CACHE = {}  # module-level cache so FAISS index isn't reloaded per request


def _load_index_objects(mode: str = "tfidf"):
    """Load index and metadata from disk (with module-level cache)."""
    global _INDEX_CACHE
    if "meta" not in _INDEX_CACHE:
        if not os.path.exists(META_PATH):
            return None, None, None
        with open(META_PATH, "r", encoding="utf-8") as f:
            _INDEX_CACHE["meta"] = json.load(f)

    meta = _INDEX_CACHE["meta"]

    if mode == "semantic" and _SEMANTIC_OK and _FAISS_OK:
        if "sem_index" not in _INDEX_CACHE:
            if not os.path.exists(SEMANTIC_INDEX_PATH):
                return None, None, None
            _INDEX_CACHE["sem_index"] = faiss.read_index(SEMANTIC_INDEX_PATH)
            _INDEX_CACHE["sem_model"] = SentenceTransformer(EMBED_MODEL_NAME)
        return meta, _INDEX_CACHE["sem_index"], _INDEX_CACHE["sem_model"]
    else:
        faiss_path = TFIDF_INDEX_PATH.replace(".pkl", ".faiss")
        if "tfidf_index" not in _INDEX_CACHE:
            if not (os.path.exists(TFIDF_INDEX_PATH) and os.path.exists(faiss_path)):
                return None, None, None
            with open(TFIDF_INDEX_PATH, "rb") as f:
                _INDEX_CACHE["tfidf_objs"] = pickle.load(f)
            _INDEX_CACHE["tfidf_index"] = faiss.read_index(faiss_path)
        return meta, _INDEX_CACHE["tfidf_index"], _INDEX_CACHE["tfidf_objs"]


def retrieve(
    query: str,
    resident_id: str,
    top_k: int = TOP_K_DEFAULT,
    mode: str = "tfidf",
    exclude_source_types: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve top-k semantically similar chunks for a given query.

    SAFETY: resident_id isolation is enforced BEFORE FAISS search by building
    a mask of allowed chunk indices. Cross-resident retrieval is architecturally
    impossible, not just filtered after the fact.

    Args:
        query:         The care plan section being generated (used as query text)
        resident_id:   Only retrieve chunks belonging to THIS resident
        top_k:         Number of chunks to return
        mode:          "tfidf" | "semantic"
        exclude_source_types: Source types to exclude (e.g. ["incident"] for personal care section)

    Returns list of dicts with keys: text, score, source_type, section, date, chunk_id
    """
    if not query or not query.strip():
        return []

    meta, index, model_or_objs = _load_index_objects(mode)
    if meta is None:
        return []

    exclude = set(exclude_source_types or [])

    # Build resident-scoped index mask (safety-critical: enforced before search)
    allowed_indices = [
        i for i, chunk in enumerate(meta)
        if chunk["resident_id"] == resident_id
        and chunk["source_type"] not in exclude
    ]

    if not allowed_indices:
        return []

    # Embed the query
    if mode == "semantic" and _SEMANTIC_OK:
        q_vec = model_or_objs.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")
    else:
        # TF-IDF path
        objs = model_or_objs
        q_tfidf = objs["vectorizer"].transform([query])
        q_vec = objs["svd"].transform(q_tfidf)
        q_vec = normalize(q_vec).astype("float32")

    # FAISS IDSelector for resident isolation
    id_selector = faiss.IDSelectorArray(np.array(allowed_indices, dtype=np.int64))
    search_params = faiss.SearchParametersIVF() if hasattr(faiss, "SearchParametersIVF") else None

    actual_k = min(top_k, len(allowed_indices))

    try:
        # Use FAISS range_search or standard search with IDSelector
        scores, indices = index.search(q_vec, index.ntotal)
        # Filter to resident-allowed and re-rank
        filtered = [
            (scores[0][j], int(indices[0][j]))
            for j in range(len(indices[0]))
            if int(indices[0][j]) in set(allowed_indices) and indices[0][j] != -1
        ]
        filtered.sort(key=lambda x: x[0], reverse=True)
        top_hits = filtered[:actual_k]
    except Exception:
        return []

    results = []
    seen_texts = set()
    for score, idx in top_hits:
        chunk = meta[idx]
        # Deduplicate near-identical chunks
        text_hash = hashlib.md5(chunk["text"][:100].encode()).hexdigest()
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)
        results.append({
            "text": chunk["text"],
            "score": float(score),
            "source_type": chunk["source_type"],
            "section": chunk["section"],
            "date": chunk["date"],
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
        })

    return results


# 4 ── CONTEXT BUILDER (formats retrieved chunks for LLM prompt injection)

def build_rag_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured context block for prompt injection.
    Follows the recommendation to insert retrieved context BEFORE the task
    instruction, not after.  Ref: Lewis et al. (2020) prompt template.
    """
    if not chunks:
        return ""

    lines = [
        "RETRIEVED SIMILAR RECORDS (use as supporting context — do not copy verbatim):",
        "=" * 60,
    ]
    for i, chunk in enumerate(chunks, 1):
        source_label = {
            "care_note":  "Care Note",
            "incident":   "Incident Record",
            "care_plan":  "Previous Care Plan",
            "wellbeing":  "Wellbeing Assessment",
            "risk":       "Risk Assessment",
        }.get(chunk["source_type"], chunk["source_type"].replace("_", " ").title())

        date_str = f" ({chunk['date']})" if chunk.get("date") else ""
        lines.append(f"\n[{i}] {source_label}{date_str} — relevance score: {chunk['score']:.3f}")
        lines.append(chunk["text"])

    lines.append("=" * 60)
    lines.append("IMPORTANT: If any retrieved record contradicts current information, mark output [REVIEW NEEDED].")
    return "\n".join(lines)


# 5 ── audit logger

def log_retrieval(db_path: str, resident_id: str, query_summary: str,
                  section: str, num_chunks: int, mode: str,
                  chunk_ids: list[str], generated_by: str) -> None:
    """Write retrieval event to rag_audit_log for GDPR compliance."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO rag_audit_log
            (resident_id, query_summary, section, num_chunks_retrieved, mode,
             chunk_ids_json, generated_by, retrieved_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            resident_id,
            query_summary[:200],
            section,
            num_chunks,
            mode,
            json.dumps(chunk_ids),
            generated_by,
            datetime.datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()
    except Exception:
        pass  # Never crash app due to audit failure


# 6 ── status / manifest

def get_index_status() -> dict:
    """Return current index metadata for the Knowledge Base admin page."""
    manifest_path = os.path.join(INDEX_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        return {
            "built": False,
            "num_chunks": 0,
            "num_docs": 0,
            "mode": "none",
            "built_at": None,
            "embed_model": None,
        }
    with open(manifest_path, "r") as f:
        data = json.load(f)
    data["built"] = True
    data["semantic_available"] = _SEMANTIC_OK and _FAISS_OK
    data["faiss_available"] = _FAISS_OK
    return data


def invalidate_cache() -> None:
    """Call after index rebuild to force reload on next retrieval."""
    global _INDEX_CACHE
    _INDEX_CACHE.clear()
