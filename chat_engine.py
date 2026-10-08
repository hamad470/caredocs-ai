"""
chat_engine.py — Agentic conversational RAG over the care record
Ask anything about a resident or about the home as a whole, across the last
twelve months (or any window), and get a grounded, cited answer.

PIPELINE (each stage is logged and surfaced in the UI's "how this answer was
built" panel, so the system is inspectable rather than oracular):

    user question
        │
        ├─▶ 1. CONTEXTUALISE     follow-ups are rewritten into standalone
        │                        questions using the last few turns
        │                        ("and her weight?" → "What is Maggie's weight
        │                        trend over the last 12 months?")
        │
        ├─▶ 2. PLAN (router)     an LLM emits a JSON plan: intent, residents,
        │                        date window, which analytics tools to run and
        │                        which retrieval queries to issue.
        │                        A deterministic rule-based planner produces the
        │                        same structure when no API key is configured,
        │                        so the system degrades but never dies.
        │
        ├─▶ 3a. ANALYTICS        named, parameterised SQL tools compute exact
        │                        numbers (analytics_tools.py)
        ├─▶ 3b. RETRIEVAL        hybrid BM25 + dense + RRF + MMR search over
        │                        chunked records (rag_advanced.py)
        │
        ├─▶ 4. SYNTHESIS         the LLM answers using ONLY the evidence block,
        │                        citing chunks as [S1], [S2] and quoting tool
        │                        figures verbatim
        │
        └─▶ 5. AUDIT             question, plan, tools, chunk IDs, provider and
                                 latency are written to chat_messages and to
                                 rag_audit_log

WHY BOTH RETRIEVAL AND TOOLS: a top-k retriever cannot count. Asked "how many
falls last quarter", a retrieval-only system returns the five most fall-like
chunks and the model guesses a total — fluently and wrongly. Counting questions
go to SQL; "what did staff observe" questions go to the vector index; most real
questions ("is her hydration getting worse and what are staff saying about it")
need both, which is why the planner may select both in one turn.

SAFETY: when a resident scope is set the retriever is hard-filtered to that
resident before any search runs, and every tool call is parameterised. No
model-generated SQL is ever executed.
"""

from __future__ import annotations

import os
import re
import json
import time
import sqlite3
import datetime

import rag_advanced
import analytics_tools as tools
import llm_client

MAX_HISTORY_TURNS  = 12     # messages of conversation carried into the prompt
                            # (6 question/answer pairs — enough for real
                            #  follow-ups without crowding out the evidence)
MAX_TOOLS_PER_TURN = 3      # keeps latency and token cost bounded
DEFAULT_TOP_K      = 8
DEFAULT_MONTHS     = 12


# Storage — chat history and audit trail

def ensure_tables(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key   TEXT NOT NULL,
            username      TEXT,
            role          TEXT NOT NULL,          -- user | assistant
            content       TEXT NOT NULL,
            standalone_q  TEXT,
            intent        TEXT,
            residents     TEXT,                   -- JSON array of resident IDs
            date_from     TEXT,
            date_to       TEXT,
            tools_used    TEXT,                   -- JSON array of tool names
            chunk_ids     TEXT,                   -- JSON array of retrieved chunk IDs
            provider      TEXT,
            latency_ms    INTEGER,
            created       TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_chat_session
            ON chat_messages(session_key, id);

        -- One row per conversation. Kept separate from chat_messages so a
        -- conversation can exist (and be titled) before its first question,
        -- and so deleting a conversation is one explicit operation.
        CREATE TABLE IF NOT EXISTS chat_conversations (
            session_key TEXT PRIMARY KEY,
            username    TEXT,
            title       TEXT,
            created     TEXT DEFAULT (datetime('now')),
            updated     TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def save_message(db_path: str, session_key: str, username: str, role: str,
                 content: str, meta: dict | None = None) -> None:
    meta = meta or {}
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO chat_messages
              (session_key, username, role, content, standalone_q, intent,
               residents, date_from, date_to, tools_used, chunk_ids,
               provider, latency_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (session_key, username, role, content,
              meta.get("standalone_question"), meta.get("intent"),
              json.dumps(meta.get("residents", [])),
              meta.get("date_from"), meta.get("date_to"),
              json.dumps(meta.get("tools_used", [])),
              json.dumps(meta.get("chunk_ids", [])),
              meta.get("provider"), meta.get("latency_ms")))
        conn.commit()
        conn.close()
    except Exception:
        pass          # a logging failure must never break the conversation


# ── Conversations ─────────────────────────────────────────────────────────

def _title_from(question: str) -> str:
    q = " ".join((question or "").split())
    return (q[:60] + "…") if len(q) > 60 else (q or "New conversation")


def create_conversation(db_path: str, username: str, title: str = "") -> str:
    """Start a new conversation and return its key."""
    ensure_tables(db_path)
    import secrets
    key = f"{username or 'anon'}-{secrets.token_hex(4)}"
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT OR IGNORE INTO chat_conversations "
                     "(session_key, username, title) VALUES (?,?,?)",
                     (key, username, title or "New conversation"))
        conn.commit()
        conn.close()
    except Exception:
        pass
    return key


def touch_conversation(db_path: str, session_key: str, username: str,
                       first_question: str = "") -> None:
    """
    Register the conversation if new, title it from its first question, and
    stamp it as recently used so the sidebar sorts sensibly.
    """
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT title FROM chat_conversations WHERE session_key=?",
                           (session_key,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO chat_conversations (session_key, username, title) "
                         "VALUES (?,?,?)",
                         (session_key, username, _title_from(first_question)))
        else:
            if (not row[0]) or row[0] == "New conversation":
                conn.execute("UPDATE chat_conversations SET title=?, updated=datetime('now') "
                             "WHERE session_key=?", (_title_from(first_question), session_key))
            else:
                conn.execute("UPDATE chat_conversations SET updated=datetime('now') "
                             "WHERE session_key=?", (session_key,))
        conn.commit()
        conn.close()
    except Exception:
        pass


def list_conversations(db_path: str, username: str, limit: int = 50) -> list[dict]:
    """
    Conversations for one user, newest first, with message counts.

    Conversations are LEFT JOINed against their messages so a brand-new empty
    conversation still appears — otherwise clicking "New chat" would seem to do
    nothing until the first question was answered.
    """
    ensure_tables(db_path)
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT c.session_key, c.title, c.created, c.updated,
                   (SELECT COUNT(*) FROM chat_messages m
                     WHERE m.session_key = c.session_key AND m.role='user') AS questions,
                   (SELECT MAX(created) FROM chat_messages m
                     WHERE m.session_key = c.session_key) AS last_message_at
              FROM chat_conversations c
             WHERE c.username = ?
             ORDER BY COALESCE(
                 (SELECT MAX(created) FROM chat_messages m
                   WHERE m.session_key = c.session_key), c.updated) DESC
             LIMIT ?
        """, (username, limit)).fetchall()
        out = [dict(r) for r in rows]

        # Conversations created before this table existed still have messages;
        # surface them rather than orphaning the user's earlier history.
        known = {r["session_key"] for r in out}
        legacy = conn.execute("""
            SELECT m.session_key,
                   (SELECT content FROM chat_messages q
                     WHERE q.session_key = m.session_key AND q.role='user'
                     ORDER BY q.id LIMIT 1)              AS first_q,
                   MAX(m.created)                        AS last_message_at,
                   SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) AS questions
              FROM chat_messages m WHERE m.username=? GROUP BY m.session_key
             ORDER BY last_message_at DESC LIMIT ?
        """, (username, limit)).fetchall()
        for r in legacy:
            d = dict(r)
            if d["session_key"] in known:
                continue
            out.append({"session_key": d["session_key"],
                        "title": _title_from(d["first_q"] or ""),
                        "created": d["last_message_at"],
                        "updated": d["last_message_at"],
                        "questions": d["questions"] or 0,
                        "last_message_at": d["last_message_at"]})
        conn.close()
        out.sort(key=lambda d: d.get("last_message_at") or d.get("updated") or "",
                 reverse=True)
        return out[:limit]
    except Exception:
        return []


def conversation_owner(db_path: str, session_key: str) -> str | None:
    """Username that owns a conversation, or None if unknown/empty."""
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT username FROM chat_conversations WHERE session_key=?",
                           (session_key,)).fetchone()
        if row is None:
            row = conn.execute("SELECT username FROM chat_messages WHERE session_key=? "
                               "LIMIT 1", (session_key,)).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def rename_conversation(db_path: str, session_key: str, title: str) -> None:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE chat_conversations SET title=?, updated=datetime('now') "
                     "WHERE session_key=?", (title[:120], session_key))
        conn.commit()
        conn.close()
    except Exception:
        pass


def delete_conversation(db_path: str, session_key: str) -> None:
    """Remove a conversation and its messages."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM chat_messages WHERE session_key=?", (session_key,))
        conn.execute("DELETE FROM chat_conversations WHERE session_key=?", (session_key,))
        conn.commit()
        conn.close()
    except Exception:
        pass


def load_history(db_path: str, session_key: str, limit: int = 40) -> list[dict]:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, created FROM chat_messages "
            "WHERE session_key=? ORDER BY id DESC LIMIT ?", (session_key, limit)
        ).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]
    except Exception:
        return []


def clear_history(db_path: str, session_key: str) -> None:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM chat_messages WHERE session_key=?", (session_key,))
        conn.commit()
        conn.close()
    except Exception:
        pass


def log_rag_audit(db_path, resident_id, question, num_chunks, chunk_ids, username):
    """Reuse the existing GDPR retrieval audit table used by care-plan generation."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO rag_audit_log
              (resident_id, query_summary, section, num_chunks_retrieved, mode,
               chunk_ids_json, generated_by, retrieved_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (resident_id or "ALL", question[:200], "chat", num_chunks,
              "hybrid-rrf", json.dumps(chunk_ids), username,
              datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception:
        pass


# Resident and date resolution

def get_residents(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT resident_id, full_name, preferred_name, room_number, "
        "primary_diagnosis FROM residents WHERE active=1 ORDER BY resident_id")]
    conn.close()
    return rows


def match_residents(text: str, residents: list[dict]) -> list[str]:
    """Find residents referred to by ID, full name, surname or preferred name."""
    if not text:
        return []
    low = text.lower()
    found = []
    for r in residents:
        candidates = {str(r["resident_id"]).lower()}
        for field in ("full_name", "preferred_name"):
            val = (r.get(field) or "").strip().lower()
            if val:
                candidates.add(val)
                candidates.update(p for p in val.split() if len(p) > 3)
        if any(re.search(rf"\b{re.escape(c)}\b", low) for c in candidates if c):
            found.append(r["resident_id"])
    return found


_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def parse_time_window(text: str, months_default: int = DEFAULT_MONTHS) -> tuple[str, str, str]:
    """
    Extract a date window from natural language.
    Returns (date_from, date_to, human_label). Defaults to the trailing year,
    which matches how care-home managers actually think about the record.
    """
    today = datetime.date.today()
    low = (text or "").lower()

    def back(days):
        return (today - datetime.timedelta(days=days)).isoformat()

    m = re.search(r"(?:last|past|previous)\s+(\d{1,2})\s+(day|week|month|year)s?", low)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = {"day": 1, "week": 7, "month": 30.44, "year": 365.25}[unit] * n
        return back(int(days)), today.isoformat(), f"last {n} {unit}{'s' if n > 1 else ''}"

    if re.search(r"\b(last|past)\s+(year|12\s*months|twelve\s*months)\b", low):
        return back(365), today.isoformat(), "last 12 months"
    if re.search(r"\b(last|past)\s+(quarter|3\s*months|three\s*months)\b", low):
        return back(91), today.isoformat(), "last 3 months"
    if re.search(r"\b(last|past)\s+month\b", low):
        return back(30), today.isoformat(), "last month"
    if re.search(r"\b(last|past)\s+(week|7\s*days)\b", low):
        return back(7), today.isoformat(), "last 7 days"
    if re.search(r"\b(today|since yesterday)\b", low):
        return back(1), today.isoformat(), "last 24 hours"
    if re.search(r"\bthis year\b", low):
        return f"{today.year}-01-01", today.isoformat(), f"{today.year} to date"
    if re.search(r"\b(all time|ever|since admission|entire record|full history)\b", low):
        return "2000-01-01", today.isoformat(), "the entire record"

    # "in March 2026" / "March 2026" / "in March"
    m = re.search(r"\b(" + "|".join(_MONTHS) + r")\s*(\d{4})?\b", low)
    if m:
        mon = _MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else today.year
        start = datetime.date(year, mon, 1)
        end = (datetime.date(year + (mon == 12), (mon % 12) + 1, 1)
               - datetime.timedelta(days=1))
        return start.isoformat(), min(end, today).isoformat(), f"{m.group(1).title()} {year}"

    # explicit ISO range
    iso = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", low)
    if len(iso) >= 2:
        return iso[0], iso[1], f"{iso[0]} to {iso[1]}"
    if len(iso) == 1:
        return iso[0], today.isoformat(), f"since {iso[0]}"

    start = today - datetime.timedelta(days=int(30.44 * months_default))
    return start.isoformat(), today.isoformat(), f"last {months_default} months"


# Rule-based planner (offline fallback and safety net)

_TOOL_KEYWORDS = [
    (r"\bfall(s|en|ing)?\b|\btrip(ped|s)?\b|\bslip(ped|s)?\b", "falls_analysis"),
    (r"\bfluid|hydrat|drink|water intake|dehydrat", "fluid_intake_trend"),
    (r"\bweight|kilo|kg|nutrition|malnutri|losing weight|weight loss", "weight_trend"),
    (r"\bmedicat|\bmar\b|adherence|refus\w* (?:her |his )?(?:meds|medication)|drug|dose|tablet",
     "medication_adherence"),
    (r"\bprescri|\bwhat (?:medication|meds|drugs)|\bpill|controlled drug|\bprn\b",
     "medication_list"),
    (r"\bincident|accident|safeguard|injur|emergenc|cqc notif", "incident_summary"),
    (r"\bwellbeing|well-being|quality of life|overall score|assessment score", "wellbeing_trend"),
    (r"\bmood|appetite|eating|sleep|agitat|anxious|withdraw|depress|happy", "mood_and_appetite"),
    (r"\bpain|discomfort|analgesi|grimac", "pain_analysis"),
    (r"\brisk|pressure (sore|ulcer)|braden|waterlow|assessment overdue", "risk_register"),
    (r"\bcomplian|audit|inspect|overdue|cqc\b|regulat|governance", "compliance_overview"),
    (r"\bhandover|escalat|shift report", "handover_escalations"),
    (r"\bfamily|relative|next of kin|daughter|son|visit", "family_contact_summary"),
    (r"\bdocumentation|how many notes|note volume|staff activity|recording", "care_note_activity"),
    (r"\bwho (are|is) the residents|list (the )?residents|how many residents", "list_residents"),
    (r"\bprofile|diagnos|allerg|dnacpr|capacity|next of kin|gp\b|room number|admitted",
     "resident_profile"),
    (r"\btimeline|what (has )?happened|overview|summar(y|ise|ize)|catch me up|history",
     "resident_timeline"),
]

# Questions about the conversation rather than about the residents. These must
# never be routed to retrieval: searching the care record for "what did I just
# ask" returns clinical chunks, and the model then correctly reports that the
# evidence does not answer the question — which reads as amnesia.
_META_HINT = re.compile(
    r"\b("
    r"(my|our|the)\s+(last|previous|first|earlier|prior)\s+(question|message|query|ask)|"
    r"what\s+(did|have)\s+(i|we)\s+(ask|asked|say|said|discuss|discussed|cover|covered)|"
    r"what\s+was\s+(my|our|the)\s+(last|previous|first)|"
    r"(summar(y|ise|ize))\s+(of\s+)?(this|our|the)\s+(chat|conversation|thread|session)|"
    r"(recap|repeat)\s+(that|this|it|our|the\s+conversation)|"
    r"what\s+(are|were)\s+we\s+(talking|discussing)|"
    r"remind\s+me\s+what\s+(i|we)"
    r")\b", re.I)


def is_meta_question(text: str) -> bool:
    """True when the question is about the chat itself, not the care record."""
    return bool(_META_HINT.search(text or ""))


def transcript_block(history: list[dict], max_turns: int = 12,
                     max_chars: int = 700) -> str:
    """
    Render the conversation so far as explicit evidence.

    The model receives the history as chat turns anyway, but that is not enough:
    a system prompt that says "use only the evidence" makes it refuse to read
    its own context. Restating the transcript INSIDE the evidence block is what
    makes "what did I ask before?" answerable without weakening the
    no-invention rule that keeps clinical answers safe.
    """
    if not history:
        return ""
    turns = history[-max_turns:]
    lines, q_no = [], 0
    for m in turns:
        body = " ".join((m.get("content") or "").split())
        if len(body) > max_chars:
            body = body[:max_chars] + "…"
        if m.get("role") == "user":
            q_no += 1
            lines.append(f"[Q{q_no}] You asked: {body}")
        else:
            lines.append(f"[A{q_no or 1}] Assistant replied: {body}")
    return "\n".join(lines)


_ANALYTIC_HINT = re.compile(
    r"\bhow many|how much|count|number of|total|average|mean|trend|rate|percent|%|"
    r"compare|comparison|most|least|worst|best|increase|decrease|over time|per month|"
    r"statistic|breakdown|distribution|score\b", re.I)


def rule_plan(question: str, residents: list[dict], scope_ids: list[str] | None,
              months: int = DEFAULT_MONTHS) -> dict:
    """Deterministic planner. Also used to sanity-check the LLM's plan."""
    date_from, date_to, label = parse_time_window(question, months)
    ids = scope_ids or match_residents(question, residents)

    if is_meta_question(question):
        return {
            "standalone_question": question,
            "intent": "meta",
            "residents": ids,
            "date_from": date_from,
            "date_to": date_to,
            "window_label": label,
            "tools": [],
            "retrieval_queries": [],
            "needs_retrieval": False,
            "planner": "rules(meta)",
        }

    picked, seen = [], set()
    for pattern, tool in _TOOL_KEYWORDS:
        if re.search(pattern, question, re.I) and tool not in seen:
            seen.add(tool)
            picked.append(tool)
    if not picked:
        picked = ["resident_timeline"] if ids else ["compliance_overview"]

    picked = picked[:MAX_TOOLS_PER_TURN]
    args_common = {"date_from": date_from, "date_to": date_to}
    if ids:
        args_common["resident_id"] = ids

    plan_tools = []
    for t in picked:
        allowed = set(tools.TOOLS[t]["args"].keys())
        plan_tools.append({"name": t,
                           "args": {k: v for k, v in args_common.items() if k in allowed}})

    return {
        "standalone_question": question,
        "intent": "analytics" if _ANALYTIC_HINT.search(question) else "mixed",
        "residents": ids,
        "date_from": date_from,
        "date_to": date_to,
        "window_label": label,
        "tools": plan_tools,
        "retrieval_queries": [question],
        "needs_retrieval": True,
        "planner": "rules",
    }


# LLM planner

PLANNER_SYSTEM = """You are the query planner of a UK care-home records assistant.
You do NOT answer the question. You output a JSON plan describing how to gather evidence.

Return ONLY a JSON object with these keys:
  standalone_question : the user's question rewritten to stand alone without the chat history
  intent              : one of "analytics", "records", "mixed", "profile", "smalltalk"
  residents           : array of resident IDs the question is about (empty array = whole home)
  date_from, date_to  : ISO dates (YYYY-MM-DD) bounding the question
  tools               : array of {"name": <tool name>, "args": {...}} — at most 3, [] if none needed
  retrieval_queries   : 1-3 short search strings for the narrative record (free text, no dates)
  needs_retrieval     : true if written narrative evidence would help, false for pure counting

Rules:
- Counting, totals, averages, trends, comparisons and adherence rates MUST come from tools.
- Questions about what staff observed, wrote, or decided need retrieval.
- Most real questions need both.
- If no time period is stated, use the last 12 months.
- Only use tool names from the catalogue. Only pass arguments the tool declares.
- Never invent resident IDs; use only those listed."""


def llm_plan(question: str, history: list[dict], residents: list[dict],
             scope_ids: list[str] | None, months: int,
             provider: str | None = None) -> dict | None:
    roster = "\n".join(
        f"  {r['resident_id']}: {r['full_name']} (known as {r['preferred_name']}), "
        f"room {r['room_number']}, {r['primary_diagnosis']}" for r in residents)
    default_from, default_to, _label = parse_time_window("", months)

    convo = "\n".join(f"{m['role']}: {m['content'][:400]}"
                      for m in history[-MAX_HISTORY_TURNS:]) or "(no previous turns)"
    scope_note = (f"\nThe user has locked the conversation to resident(s): "
                  f"{', '.join(scope_ids)}. Use exactly these." if scope_ids else "")

    user_msg = f"""Today is {datetime.date.today().isoformat()}.
Default window if none is stated: {default_from} to {default_to}.

RESIDENTS ON THE REGISTER:
{roster}
{scope_note}

ANALYTICS TOOL CATALOGUE:
{tools.tool_catalogue()}

CONVERSATION SO FAR:
{convo}

NEW USER QUESTION: {question}

Output the JSON plan now."""

    obj, prov = llm_client.complete_json(
        [{"role": "user", "content": user_msg}],
        system=PLANNER_SYSTEM, max_tokens=800, temperature=0.0,
        provider=provider, timeout=25)
    if not obj:
        return None
    obj["planner"] = f"llm:{prov}"
    return obj


def _sanitise_plan(plan: dict, fallback: dict, residents: list[dict],
                   scope_ids: list[str] | None) -> dict:
    """
    Trust nothing the model returned. Every field is validated against the
    catalogue and the resident register before it can influence a query.
    """
    valid_ids = {r["resident_id"] for r in residents}
    out = dict(fallback)                       # start from the rule-based plan

    if isinstance(plan.get("standalone_question"), str) and plan["standalone_question"].strip():
        out["standalone_question"] = plan["standalone_question"].strip()[:400]

    if plan.get("intent") in ("analytics", "records", "mixed", "profile",
                              "smalltalk", "meta"):
        out["intent"] = plan["intent"]

    ids = plan.get("residents") or []
    if isinstance(ids, str):
        ids = [ids]
    ids = [i for i in ids if i in valid_ids]
    out["residents"] = scope_ids if scope_ids else ids

    for key in ("date_from", "date_to"):
        val = plan.get(key)
        if isinstance(val, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", val.strip()):
            out[key] = val.strip()
    if out["date_from"] > out["date_to"]:
        out["date_from"], out["date_to"] = out["date_to"], out["date_from"]

    plan_tools = []
    for t in (plan.get("tools") or [])[:MAX_TOOLS_PER_TURN]:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if name not in tools.TOOLS:
            continue
        declared = set(tools.TOOLS[name]["args"].keys())
        args = {k: v for k, v in (t.get("args") or {}).items() if k in declared}
        # Scope and window are authoritative from the validated plan, not the model.
        if "resident_id" in declared and out["residents"]:
            args["resident_id"] = out["residents"]
        # The window is forced rather than defaulted: if a tool computed over a
        # different period from the one shown to the user, the answer would
        # quote a figure that does not match its own stated date range.
        if "date_from" in declared:
            args["date_from"] = out["date_from"]
        if "date_to" in declared:
            args["date_to"] = out["date_to"]
        plan_tools.append({"name": name, "args": args})
    if plan_tools:
        out["tools"] = plan_tools

    queries = [q for q in (plan.get("retrieval_queries") or [])
               if isinstance(q, str) and q.strip()][:3]
    if queries:
        out["retrieval_queries"] = [q.strip()[:200] for q in queries]

    if isinstance(plan.get("needs_retrieval"), bool):
        out["needs_retrieval"] = plan["needs_retrieval"]
    out["planner"] = plan.get("planner", out.get("planner", "rules"))
    return out


# Answer synthesis

ANSWER_SYSTEM = """You are a care-home records analyst supporting UK care staff and managers.
You answer questions about residents using ONLY the evidence supplied to you.

Rules:
1. Use only the evidence supplied: the ANALYTICS RESULTS, the RETRIEVED RECORDS, and the CONVERSATION SO FAR. Never invent a fact, a number, a date or a name.
1b. The CONVERSATION SO FAR is a true record of this chat. Questions about the conversation itself — "what did I just ask", "what have we covered", "summarise this chat", "repeat that" — are answered from it directly. Never claim you have no access to the conversation history when a transcript is supplied.
2. Quote analytics figures exactly as computed. Do not recalculate or round them differently.
3. Cite narrative evidence inline as [S1], [S2] etc., matching the evidence labels. Cite the specific records that support each claim.
4. If the evidence does not answer the question, say so plainly and state what is missing. Never guess.
5. Be concise and clinical: short paragraphs or bullets, UK English, care-home vocabulary.
6. Flag anything that looks clinically urgent (rapid weight loss, repeated falls, unmanaged pain, safeguarding, overdue reviews) in a short line beginning "⚠ Clinical flag:".
7. You are a decision-support tool, not a clinician. Do not give diagnoses, prescriptions or dosage advice; when asked for those, point to the GP or prescriber.
8. Never reveal information about a resident other than the one(s) in scope."""


def _offline_reason() -> str:
    """
    Explain *why* no model answered. A bare "offline" tells a user nothing and
    costs them a support conversation; naming the key and the HTTP status turns
    it into something they can act on in thirty seconds.
    """
    st = llm_client.status()
    if not st["providers"].get("gemini"):
        return "no Gemini API key is saved — add one at Settings → AI Settings"
    if (os.environ.get("AI_PROVIDER") or "auto").lower() == "template":
        return "generation is pinned to the offline template in AI Settings"

    keys = st.get("keys") or []
    if keys and not any(k.get("usable") for k in keys):
        cooling = [k for k in keys if k.get("cooling_down")]
        if len(cooling) == len(keys):
            wait = max(k.get("cooldown_seconds", 0) for k in cooling)
            return (f"all {len(keys)} Gemini key(s) are out of free-tier quota; "
                    f"the first retries in about {wait}s")
        dead = [k["id"] for k in keys if not k.get("usable") and not k.get("cooling_down")]
        if dead:
            return (f"key(s) {', '.join(dead)} were rejected by the API — "
                    f"replace them at Settings → AI Settings")

    errors = st.get("errors") or {}
    if errors:
        detail = "; ".join(f"{p} → {msg[:90]}" for p, msg in errors.items())
        return f"Gemini returned an error: {detail}"
    return ("Gemini returned nothing — use the 'Test AI' button to see the "
            "exact error against each key")


def _meta_answer(question: str, history: list[dict]) -> str:
    """
    Deterministic answer for questions about the conversation itself.
    Computed from the stored transcript, so it is correct even with no AI
    provider configured — and it can never hallucinate a question you did not
    actually ask.
    """
    questions = [" ".join((m.get("content") or "").split())
                 for m in history if m.get("role") == "user"]
    if not questions:
        return ("This conversation has no earlier questions yet — this is the first one. "
                "Earlier threads are listed under **Conversations** on the right.")

    low = (question or "").lower()
    if re.search(r"\b(first|earliest)\b", low):
        return f"Your first question in this conversation was:\n\n> {questions[0]}"
    if re.search(r"summar|recap|what have (i|we)|what did we (discuss|cover)|"
                 r"what (are|were) we (talking|discussing)", low):
        lines = [f"This conversation has {len(questions)} question(s) so far:", ""]
        lines += [f"{i}. {q}" for i, q in enumerate(questions, 1)]
        return "\n".join(lines)
    return (f"Your last question was:\n\n> {questions[-1]}\n\n"
            f"That is question {len(questions)} of {len(questions)} in this conversation.")


def _fallback_answer(question: str, tool_results: list[dict], chunks: list[dict],
                     window_label: str) -> str:
    """
    Deterministic answer used when no LLM backend is available.
    It is a genuine, if blunt, answer — every figure below is computed by SQL —
    so the system remains demonstrable with zero API keys configured.
    """
    parts = [f"**Offline evidence mode** — {_offline_reason()}. "
             f"Retrieval and analytics still ran; what is missing is only the "
             f"written synthesis. Evidence for *{question}* ({window_label}):\n"]
    for res in tool_results:
        if res.get("error"):
            continue
        parts.append(f"**{res.get('title')}**  \n{res.get('summary')}")
        cols, rows = res.get("columns") or [], res.get("rows") or []
        if cols and rows:
            parts.append("| " + " | ".join(map(str, cols)) + " |")
            parts.append("|" + "|".join("---" for _ in cols) + "|")
            for row in rows[:8]:
                parts.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
        parts.append("")
    if chunks:
        parts.append("**Most relevant records:**")
        for i, c in enumerate(chunks[:5], 1):
            parts.append(f"- [S{i}] *{c['source_label']}, {c['date']}, "
                         f"{c['resident_name']}* — {c['text'][:260]}…")
    if len(parts) <= 1:
        parts.append("No matching records were found for that question in this window.")
    parts.append("\n_Fix the provider (Settings → AI Settings, or the "
                 "'Test AI connection' button above) to get a synthesised, "
                 "cited answer instead of this evidence dump._")
    return "\n".join(parts)


def synthesise(question: str, plan: dict, tool_results: list[dict],
               chunks: list[dict], history: list[dict], residents: list[dict],
               provider: str | None = None) -> tuple[str, str | None]:
    evidence = []
    transcript = transcript_block(history)
    if transcript:
        evidence.append("CONVERSATION SO FAR (this chat, oldest first — "
                        "use for follow-ups and for questions about the chat itself):")
        evidence.append(transcript)
        evidence.append("")
    if tool_results:
        evidence.append("ANALYTICS RESULTS (computed directly from the database — exact figures):")
        evidence.extend(tools.render_tool_result(r) for r in tool_results)
    if chunks:
        evidence.append("\nRETRIEVED RECORDS (cite these as [S1], [S2] …):")
        evidence.append(rag_advanced.build_context_block(chunks))
    if not evidence:
        evidence.append("(no evidence could be gathered for this question)")

    scope = ", ".join(
        next((f"{r['full_name']} ({rid})" for r in residents if r["resident_id"] == rid), rid)
        for rid in plan["residents"]) or "the whole home"

    # Prior turns are also passed as real chat turns. Long assistant answers are
    # trimmed here: full ones (with their tables) would crowd out the evidence.
    convo = []
    for m in history[-MAX_HISTORY_TURNS:]:
        body = m.get("content") or ""
        if m.get("role") != "user" and len(body) > 900:
            body = body[:900] + "…"
        convo.append({"role": m.get("role", "user"), "content": body})
    # The Gemini API requires the first turn to be from the user.
    while convo and convo[0]["role"] != "user":
        convo.pop(0)

    convo.append({"role": "user", "content":
                  f"""QUESTION: {plan['standalone_question']}

SCOPE: {scope}
PERIOD: {plan['date_from']} to {plan['date_to']} ({plan.get('window_label', '')})

EVIDENCE
========
{chr(10).join(evidence)}
========

Answer the question using only the evidence above. Cite records as [S1], [S2].
If this is a question about our conversation, answer it from CONVERSATION SO FAR.
If the evidence is insufficient, say exactly what is missing."""})

    text, prov = llm_client.complete(convo, system=ANSWER_SYSTEM, max_tokens=1400,
                                     temperature=0.15, provider=provider, timeout=60)
    if not text:
        # No provider: a meta question still has a correct deterministic answer.
        if plan.get("intent") == "meta":
            return _meta_answer(question, history), None
        return _fallback_answer(question, tool_results, chunks,
                                plan.get("window_label", "")), None
    return text, prov


# Orchestration

def suggest_followups(plan: dict, residents: list[dict]) -> list[str]:
    """Context-aware next questions (deterministic — no extra API call)."""
    if plan.get("intent") == "meta":
        return ["Summarise this conversation",
                "Which residents need attention this week?",
                "Show all incidents in the last 3 months",
                "Which risk assessments are overdue?"]
    name = None
    if plan["residents"]:
        rid = plan["residents"][0]
        name = next((r["preferred_name"] or r["full_name"]
                     for r in residents if r["resident_id"] == rid), rid)
    who = name or "the home"
    used = {t["name"] for t in plan.get("tools", [])}
    pool = [
        (f"How has {who}'s weight changed over the last 12 months?", "weight_trend"),
        (f"Show all incidents for {who} in the last 6 months", "incident_summary"),
        (f"What is the medication adherence rate for {who}?", "medication_adherence"),
        (f"Is {who}'s hydration below target?", "fluid_intake_trend"),
        (f"What are the current risk assessments and are any overdue?", "risk_register"),
        (f"Summarise the wellbeing trend for {who}", "wellbeing_trend"),
        (f"What has happened with {who} in the last 3 months?", "resident_timeline"),
        ("Which residents need attention this week?", "compliance_overview"),
    ]
    out = [q for q, t in pool if t not in used][:4]
    return out


def answer(db_path: str,
           question: str,
           session_key: str = "default",
           username: str = "unknown",
           scope_resident_ids: list[str] | None = None,
           months: int = DEFAULT_MONTHS,
           top_k: int = DEFAULT_TOP_K,
           provider: str | None = None,
           use_llm_planner: bool = True,
           persist: bool = True,
           retrieval_opts: dict | None = None) -> dict:
    """
    Answer one conversational question. Returns a dict containing the answer,
    its citations, the analytics tables, and a full execution trace.
    """
    t0 = time.time()
    ensure_tables(db_path)
    residents = get_residents(db_path)
    history = load_history(db_path, session_key) if persist else []
    retrieval_opts = retrieval_opts or {}

    if not question or not question.strip():
        return {"answer": "Please type a question.", "sources": [], "tools": [],
                "trace": {}, "suggestions": []}

    # ── 1-2 ── plan ──────────────────────────────────────────────────────
    base_plan = rule_plan(question, residents, scope_resident_ids, months)
    plan = base_plan
    planner_error = None
    if use_llm_planner:
        try:
            raw = llm_plan(question, history, residents, scope_resident_ids,
                           months, provider)
            if raw:
                plan = _sanitise_plan(raw, base_plan, residents, scope_resident_ids)
        except Exception as e:
            planner_error = f"{type(e).__name__}: {e}"

    # A question about the conversation is settled here, not by the planner
    # model: the LLM planner happily rewrites "what was my last question?" into
    # a clinical query and sends it to retrieval, which is exactly the failure
    # that makes the assistant look as though it has no memory.
    if is_meta_question(question):
        plan["intent"] = "meta"
        plan["standalone_question"] = question
        plan["tools"] = []
        plan["retrieval_queries"] = []
        plan["needs_retrieval"] = False

    # The window label follows whichever dates survived validation.
    if plan["date_from"] != base_plan["date_from"] or plan["date_to"] != base_plan["date_to"]:
        plan["window_label"] = f"{plan['date_from']} to {plan['date_to']}"

    # ── 3a ── analytics tools ────────────────────────────────────────────
    tool_results = []
    for spec in plan.get("tools", [])[:MAX_TOOLS_PER_TURN]:
        tool_results.append(tools.run_tool(db_path, spec["name"], spec.get("args")))

    # ── 3b ── hybrid retrieval ───────────────────────────────────────────
    chunks, retrieval_trace = [], {}
    if plan.get("needs_retrieval", True):
        seen = set()
        per_query_k = max(3, top_k // max(len(plan["retrieval_queries"]), 1) + 2)
        for q in plan["retrieval_queries"]:
            tr = {}
            hits = rag_advanced.retrieve(
                query=q,
                resident_ids=plan["residents"] or None,
                top_k=per_query_k,
                date_from=plan["date_from"],
                date_to=plan["date_to"],
                source_types=retrieval_opts.get("source_types"),
                use_recency=retrieval_opts.get("use_recency", True),
                use_mmr=retrieval_opts.get("use_mmr", True),
                use_hybrid=retrieval_opts.get("use_hybrid", True),
                trace=tr)
            for h in hits:
                if h["chunk_id"] in seen:
                    continue
                seen.add(h["chunk_id"])
                chunks.append(h)
            retrieval_trace[q] = tr
        chunks.sort(key=lambda c: -c["score"])
        chunks = chunks[:top_k]

        # If a resident-scoped, date-bounded search found nothing, widen the
        # window once rather than reporting "no records" for a record that
        # exists just outside an arbitrary boundary.
        if not chunks and plan["residents"]:
            tr = {}
            chunks = rag_advanced.retrieve(
                query=plan["standalone_question"],
                resident_ids=plan["residents"], top_k=top_k, trace=tr)
            retrieval_trace["(widened: no date filter)"] = tr

    # ── 4 ── synthesis ───────────────────────────────────────────────────
    answer_text, used_provider = synthesise(question, plan, tool_results, chunks,
                                            history, residents, provider)

    # ── citations actually used ──────────────────────────────────────────
    cited = {int(n) for n in re.findall(r"\[S(\d+)\]", answer_text)}
    sources = []
    for i, c in enumerate(chunks, 1):
        sources.append({**c, "label": f"S{i}", "cited": i in cited})

    latency_ms = int((time.time() - t0) * 1000)
    trace = {
        "planner": plan.get("planner"),
        "planner_error": planner_error,
        "intent": plan.get("intent"),
        "standalone_question": plan.get("standalone_question"),
        "residents": plan["residents"],
        "date_from": plan["date_from"],
        "date_to": plan["date_to"],
        "window_label": plan.get("window_label"),
        "tools_called": [{"name": t["name"], "args": t.get("args", {})}
                         for t in plan.get("tools", [])],
        "retrieval_queries": plan.get("retrieval_queries", []),
        "retrieval": retrieval_trace,
        "chunks_retrieved": len(chunks),
        "chunks_cited": len(cited),
        "answer_provider": used_provider or "offline-template",
        "offline_reason": None if used_provider else _offline_reason(),
        "llm": llm_client.last_call(),
        "latency_ms": latency_ms,
        "index": rag_advanced.get_status(),
    }

    if persist:
        meta = {"standalone_question": plan.get("standalone_question"),
                "intent": plan.get("intent"), "residents": plan["residents"],
                "date_from": plan["date_from"], "date_to": plan["date_to"],
                "tools_used": [t["name"] for t in plan.get("tools", [])],
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "provider": used_provider, "latency_ms": latency_ms}
        touch_conversation(db_path, session_key, username, question)
        save_message(db_path, session_key, username, "user", question, meta)
        save_message(db_path, session_key, username, "assistant", answer_text, meta)
        log_rag_audit(db_path, ",".join(plan["residents"]) or "ALL",
                      plan.get("standalone_question", question), len(chunks),
                      [c["chunk_id"] for c in chunks], username)

    return {
        "answer": answer_text,
        "sources": sources,
        "tools": tool_results,
        "trace": trace,
        "plan": plan,
        "suggestions": suggest_followups(plan, residents),
        "provider": used_provider or "offline-template",
        "latency_ms": latency_ms,
    }


# CLI harness:  python chat_engine.py "how many falls last 6 months?"

if __name__ == "__main__":
    import sys
    try:
        import ai_config          # loads saved API keys into the environment
    except Exception:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    db = os.path.join(here, "carehome.db")
    q = " ".join(sys.argv[1:]) or "What has happened with Maggie in the last 12 months?"

    if not rag_advanced.get_status().get("built"):
        print("[*] Building index …")
        rag_advanced.build_index(db)

    res = answer(db, q, session_key="cli", username="cli", persist=False)
    print("\n" + "=" * 78)
    print("Q:", q)
    print("=" * 78)
    print(res["answer"])
    print("\n--- trace ---")
    print(json.dumps({k: v for k, v in res["trace"].items()
                      if k not in ("retrieval", "index")}, indent=2, default=str))
    print("--- sources ---")
    for s in res["sources"]:
        print(f" {s['label']}{'*' if s['cited'] else ' '} {s['source_label']:22s} "
              f"{s['date']} {s['resident_name']:16s} {s['text'][:70]}")
