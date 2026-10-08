"""Build the data for the live "Ask the Records" demo in site/.

The browser demo searches exactly the documents the Flask app indexes: they
come from rag_advanced.load_documents(), so a record ID in the demo is the same
record ID the app cites. Each document is split into numbered lines (one
sentence per line, using the same splitter as the app's chunker), which is what
a citation such as "CN-1042, line 3" points at. Lines are sentences, except
that a record field ("Injuries: ...", "Outcome: ...") always starts its own line
and abbreviations such as "Dr. B. Osei" are never split.

    python build_rag_demo.py               # writes site/data/corpus.json
    python build_rag_demo.py --reference   # also writes site/test/reference.json

It also writes site/keys.js from the GEMINI_API_KEYS environment variable
(comma-separated). That file is git-ignored: on GitHub the keys come from the
repository secret GEMINI_API_KEYS and are added only to the published site.

--reference scores a fixed set of queries with the Python BM25 class and saves
the rankings; site/test/retrieval.test.mjs checks the JavaScript BM25 returns
the same records with the same scores.
"""
import argparse
import datetime
import json
import os
import re
import sqlite3

import rag_advanced as ra

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "carehome.db")
OUT_DIR = os.path.join(ROOT, "site", "data")
REF_PATH = os.path.join(ROOT, "site", "test", "reference.json")

# Where each record type lives in carehome.db, so the demo can say exactly
# which table and row a citation came from.
TABLES = {
    "profile": "residents", "care_note": "care_notes", "incident": "incidents",
    "care_plan": "care_plans", "wellbeing": "wellbeing", "risk": "risk_assessments",
    "handover": "handovers", "family_comm": "family_comms", "medication": "medications",
}

REFERENCE_QUERIES = [
    "fall in the bathroom", "Ethel weight loss", "refused medication",
    "choking risk soft diet", "family complained about care", "pressure area redness",
    "low fluid intake dehydration", "agitated at night", "GP visit antibiotics",
    "sensor mat hourly checks", "DNACPR in place", "urinary tract infection",
    "walking frame supervised", "skin tear on arm", "poor appetite weight",
    "escalation required", "missed dose", "end of life preferences",
    "daughter visited", "pain in hip after fall",
]


def _tidy(text: str) -> str:
    """Remove the stray ' .' and '..' left by empty optional fields."""
    text = re.sub(r" \((?:None|none|)\)", "", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# Field labels written by rag_advanced.load_documents(). A label always starts a
# new line, even where the field before it had no closing full stop.
_FIELDS = [
    "Immediate actions", "Injuries", "Medical attention", "Outcome measures", "Outcome",
    "Lessons learned", "Preventative actions", "GP notified", "Family notified",
    "CQC notification", "Status", "Concerns for next shift", "Concerns", "Actions taken",
    "Handover", "Care completed", "Outstanding tasks", "Medication notes",
    "Escalation required", "Family response", "Follow-up", "Physical", "Mental health",
    "Social", "Goals progress", "Positive outcomes", "Resident voice", "Family feedback",
    "Risk factors", "Interventions", "Additional actions", "Administration notes",
    "Side effects to monitor", "Stopped because", "Indication", "Mood recorded as",
    "Appetite", "Fluid intake", "Weight", "Sleep quality", "Pain observed", "Skin check",
    "Falls this shift", "Activity", "Personal care", "Trigger",
]
_FIELD_RE = re.compile(r"\s+(?=(?:%s)\b:?)" % "|".join(
    re.escape(f) for f in sorted(_FIELDS, key=len, reverse=True)))
# A full stop ends a sentence unless it follows a title or a single initial.
_SENT_RE = re.compile(r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\b[A-Z]\.)(?<!\s\d\.)(?<!^\d\.)(?<=[.!?])\s+(?=[A-Z0-9])")


def split_lines(text: str) -> list[str]:
    """Split a record into citable lines: sentences, with each field on its own line."""
    lines = []
    for sentence in _SENT_RE.split(_tidy(text)):
        for part in _FIELD_RE.split(sentence):
            part = part.strip()
            # Only break before a label that is really a field (followed by ':'),
            # or that opens a structured observation such as "Fluid intake 1523 ml."
            if part:
                lines.append(part)
    # Re-join fragments that were split before a word that was not a field.
    merged = []
    for part in lines:
        is_field = bool(re.match(r"(?:%s)\b" % "|".join(map(re.escape, _FIELDS)), part))
        if merged and not is_field and not re.search(r"[.!?]$", merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged


def _row_ids(conn) -> dict[str, int]:
    """incident_id string -> table row id (other types carry the row id in their ID)."""
    return {iid: rid for rid, iid in conn.execute("SELECT id, incident_id FROM incidents")}


def build(db_path: str = DB) -> dict:
    if not os.path.exists(db_path):
        raise SystemExit("carehome.db not found. Run: python setup_project.py --skip-train")
    conn = sqlite3.connect(db_path)
    incident_rows = _row_ids(conn)
    residents = conn.execute(
        "SELECT resident_id, full_name, preferred_name FROM residents ORDER BY resident_id"
    ).fetchall()
    conn.close()
    res_index = {rid: i for i, (rid, _, _) in enumerate(residents)}

    docs, seen = [], {}
    for d in ra.load_documents(db_path):
        sid, stype = d["source_id"], d["source_type"]
        # Care plans are stored one document per section under one plan ID;
        # number the sections so every document has a unique key.
        n = seen.get(sid, 0) + 1
        seen[sid] = n
        key = f"{sid}/{n}" if stype == "care_plan" else sid

        if stype == "incident":
            row = incident_rows.get(sid)
        elif stype == "profile":
            row = res_index.get(d["resident_id"], -1) + 1
        else:
            m = re.search(r"-(\d+)$", sid)
            row = int(m.group(1)) if m else None

        lines = split_lines(d["text"])
        extra = {k: v for k, v in (d.get("extra") or {}).items() if v not in (None, "")}
        docs.append([key, stype, res_index.get(d["resident_id"], -1), d["date"],
                     d["section"], lines, TABLES[stype], row, extra])

    dates = sorted(x[3] for x in docs if x[3] and x[1] in ("care_note", "handover"))
    meta = {
        "built": datetime.date.today().isoformat(),
        "seed": 4242,
        "n_docs": len(docs),
        "n_lines": sum(len(x[5]) for x in docs),
        "date_from": dates[0] if dates else "",
        "date_to": dates[-1] if dates else "",
        "labels": ra.SOURCE_LABELS,
    }
    return {"meta": meta, "residents": [list(r) for r in residents], "docs": docs}


def doc_text(doc) -> str:
    """The exact text both engines index: the document's lines joined by spaces."""
    return " ".join(doc[5])


def reference(corpus: dict) -> dict:
    docs = corpus["docs"]
    bm = ra.BM25([ra.tokenize(doc_text(d)) for d in docs])
    out = []
    for q in REFERENCE_QUERIES:
        scores = bm.scores(ra.tokenize(q))
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        out.append({"query": q, "top": [[docs[i][0], round(float(s), 6)] for i, s in top]})
    return {"k1": bm.k1, "b": bm.b, "queries": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", action="store_true",
                    help="also write the Python BM25 rankings used by the JS parity test")
    args = ap.parse_args()

    corpus = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "corpus.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, separators=(",", ":"))
    m = corpus["meta"]
    print(f"  {m['n_docs']:,} records, {m['n_lines']:,} lines -> {os.path.relpath(path)} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")

    keys = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
    keys_path = os.path.join(ROOT, "site", "keys.js")
    with open(keys_path, "w", encoding="utf-8") as f:
        f.write("// Generated by build_rag_demo.py from GEMINI_API_KEYS. Do not commit.\n")
        f.write(f"export const GEMINI_KEYS = {json.dumps(keys)};\n")
    print(f"  {len(keys)} Gemini key(s) -> {os.path.relpath(keys_path)}"
          + ("" if keys else "  (none set: the demo will answer with quotes only)"))

    if args.reference:
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        with open(REF_PATH, "w", encoding="utf-8") as f:
            json.dump(reference(corpus), f, indent=1)
        print(f"  reference rankings -> {os.path.relpath(REF_PATH)}")


if __name__ == "__main__":
    main()
