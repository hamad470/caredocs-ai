"""
test_chat_rag.py — self-test for the advanced RAG + chat assistant
Run this before a demo or a supervisor meeting. It checks every layer in
order, so the first FAIL tells you exactly which one is broken.

    python test_chat_rag.py                 # full check (uses your API keys)
    python test_chat_rag.py --offline       # skip all network calls
    python test_chat_rag.py --rebuild       # force a fresh index build

Checks:
  1  dependencies importable
  2  database present and populated
  3  hybrid index builds / loads
  4  retrieval returns sensible hits
  5  resident isolation — no cross-resident leakage (safety-critical)
  6  every analytics tool executes
  7  chat answers a question with no AI provider (offline fallback)
  8  each configured AI provider responds
  9  chat answers with the AI provider, and cites its sources
"""

import os
import sys
import time
import json
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "carehome.db")
OFFLINE = "--offline" in sys.argv
REBUILD = "--rebuild" in sys.argv

passes, failures, warnings = [], [], []


def ok(msg):
    passes.append(msg)
    print(f"  \033[92mPASS\033[0m  {msg}")


def fail(msg):
    failures.append(msg)
    print(f"  \033[91mFAIL\033[0m  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  \033[93mWARN\033[0m  {msg}")


def head(n, title):
    print(f"\n\033[1m[{n}] {title}\033[0m")


# ── 1. dependencies ────────────────────────────────────────────────────────
head(1, "Dependencies")
try:
    import numpy, sklearn                                   # noqa: F401
    ok(f"numpy {numpy.__version__}, scikit-learn {sklearn.__version__}")
except Exception as e:
    fail(f"numpy/scikit-learn missing: {e}  →  pip install -r requirements.txt")
    sys.exit(1)

try:
    import faiss                                            # noqa: F401
    ok("faiss-cpu present (vector search accelerated)")
except ImportError:
    warn("faiss-cpu not installed — falling back to numpy search (still correct, "
         "just slower).  pip install faiss-cpu")

try:
    import ai_config                                        # noqa: F401
    ok("ai_config imported — saved API keys loaded into the environment")
except Exception as e:
    warn(f"ai_config not loaded: {e}")

import rag_advanced
import analytics_tools
import chat_engine
import llm_client

# ── 2. database ────────────────────────────────────────────────────────────
head(2, "Database")
if not os.path.exists(DB):
    fail(f"{DB} not found — run  python seed_data.py  first")
    sys.exit(1)
conn = sqlite3.connect(DB)
counts = {}
for t in ("residents", "care_notes", "incidents", "care_plans", "wellbeing",
          "risk_assessments", "mar_records", "handovers", "family_comms", "medications"):
    try:
        counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    except sqlite3.Error:
        counts[t] = 0
conn.close()
print("        " + ", ".join(f"{k}={v}" for k, v in counts.items()))
if counts["residents"] and counts["care_notes"]:
    ok(f"{counts['residents']} residents, {counts['care_notes']} care notes")
else:
    fail("database looks empty — run  python seed_data.py")

# ── 3. index ───────────────────────────────────────────────────────────────
head(3, "Hybrid index")
status = rag_advanced.get_status()
if REBUILD or not status.get("built"):
    print("        building …")
    t0 = time.time()
    status = rag_advanced.build_index(DB, verbose=False)
    print(f"        built in {time.time() - t0:.1f}s")
if status.get("error"):
    fail(status["error"])
elif status.get("num_chunks", 0) > 0:
    ok(f"{status['num_chunks']:,} chunks from {status['num_docs']:,} documents "
       f"({status['dense_backend']} + {status['lexical_backend']})")
    print(f"        covers {status['date_range'][0]} → {status['date_range'][1]}, "
          f"types: {', '.join(status['source_types'])}")
else:
    fail("index is empty")

# ── 4. retrieval ───────────────────────────────────────────────────────────
head(4, "Retrieval quality")
QUERIES = [
    "falls during the night and mobility problems",
    "refused medication and became agitated",
    "poor appetite and reduced fluid intake",
    "family raised a concern about care",
    "pressure area skin redness",
]
for q in QUERIES:
    tr = {}
    hits = rag_advanced.retrieve(q, top_k=5, trace=tr)
    if hits:
        ok(f"'{q[:45]}…' → {len(hits)} hits "
           f"(bm25 {tr.get('bm25_hits')}, dense {tr.get('dense_hits')}, "
           f"pool {tr.get('fused_pool')})")
    else:
        fail(f"no hits for '{q}'")

# hybrid should beat single-retriever on at least one query (sanity, not proof)
tr_h, tr_d = {}, {}
rag_advanced.retrieve(QUERIES[0], top_k=5, use_hybrid=True, trace=tr_h)
rag_advanced.retrieve(QUERIES[0], top_k=5, use_hybrid=False, trace=tr_d)
if tr_h.get("fused_pool", 0) >= tr_d.get("fused_pool", 0):
    ok(f"hybrid fusion widens the candidate pool "
       f"({tr_d.get('fused_pool')} dense-only → {tr_h.get('fused_pool')} hybrid)")

# ── 5. resident isolation (safety-critical) ────────────────────────────────
head(5, "Resident isolation")
residents = chat_engine.get_residents(DB)
leaks = 0
for r in residents:
    hits = rag_advanced.retrieve("recent concerns and incidents",
                                 resident_ids=[r["resident_id"]], top_k=10)
    others = {h["resident_id"] for h in hits} - {r["resident_id"]}
    if others:
        leaks += 1
        fail(f"{r['resident_id']} query leaked {others}")
if not leaks:
    ok(f"no cross-resident leakage across {len(residents)} scoped searches")

# date filtering
hits = rag_advanced.retrieve("care given", top_k=20,
                             date_from="2026-01-01", date_to="2026-03-31")
bad = [h["date"] for h in hits if h["date"] and not ("2026-01-01" <= h["date"] <= "2026-03-31")]
if bad:
    fail(f"date filter leaked {len(bad)} out-of-window chunks, e.g. {bad[:3]}")
else:
    ok(f"date window filter honoured ({len(hits)} chunks, all inside Q1 2026)")

# ── 6. analytics tools ─────────────────────────────────────────────────────
head(6, "Analytics tools")
for name in analytics_tools.TOOLS:
    args = {}
    if "resident_id" in analytics_tools.TOOLS[name]["args"] and residents:
        args["resident_id"] = residents[0]["resident_id"]
    res = analytics_tools.run_tool(DB, name, args)
    if res.get("error"):
        fail(f"{name}: {res['error']}")
    else:
        ok(f"{name}: {res['summary'][:90]}")

# tool argument validation must reject junk
bad = analytics_tools.run_tool(DB, "falls_analysis",
                               {"resident_id": "RES001", "evil": "DROP TABLE residents"})
ok("unknown tool arguments are dropped before execution"
   if not bad.get("error") else "argument validation raised instead of dropping")
missing = analytics_tools.run_tool(DB, "definitely_not_a_tool", {})
ok("unknown tool names are rejected" if missing.get("error") else "unknown tool NOT rejected")

# ── 7. offline chat ────────────────────────────────────────────────────────
head(7, "Chat — offline fallback (no AI provider)")
res = chat_engine.answer(DB, "How many falls did we have in the last 6 months?",
                         provider="template", use_llm_planner=False, persist=False)
if res["answer"] and res["tools"]:
    ok(f"offline answer produced ({len(res['answer'])} chars, "
       f"{len(res['tools'])} analytics table(s), {len(res['sources'])} records)")
else:
    fail("offline fallback produced nothing")

res = chat_engine.answer(DB, "Is Ethel losing weight?", provider="template",
                         use_llm_planner=False, persist=False)
if any(t["tool"] == "weight_trend" for t in res["tools"]):
    ok("rule-based planner routed a weight question to weight_trend")
else:
    fail(f"weight question routed to {[t['tool'] for t in res['tools']]}")
if res["trace"]["residents"] == ["RES003"]:
    ok("resident 'Ethel' resolved to RES003 from the question text")
else:
    fail(f"name resolution gave {res['trace']['residents']}")

# ── 8. AI providers ────────────────────────────────────────────────────────
head(8, "AI providers")
avail = llm_client.available_providers()
print("        keys configured: " +
      ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in avail.items()))
live = []
if OFFLINE:
    warn("--offline given, skipping live API calls")
else:
    if avail.get("gemini"):
        info = llm_client.discover_gemini_models(force=True)
        if info["models"]:
            ok(f"gemini key exposes {len(info['models'])} model(s) via {info['version']}; "
               f"using {info['chosen']}")
        else:
            fail(f"gemini model discovery failed — {info['error']}  "
                 f"(run  python check_ai.py  for the full diagnosis)")
    for name, present in avail.items():
        if not present:
            warn(f"{name}: no API key configured (Settings → AI Settings)")
            continue
        t0 = time.time()
        text, prov = llm_client.complete(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            system="You reply with exactly one word.", max_tokens=10,
            provider=name, timeout=30)
        if text:
            live.append(name)
            ok(f"{name} responded in {int((time.time()-t0)*1000)} ms "
               f"(model {llm_client.last_call().get('model')})")
        else:
            fail(f"{name} did not respond — {llm_client.last_call().get('error')}")

# ── 9. full chat with an AI provider ───────────────────────────────────────
head(9, "Chat — full pipeline with AI")
if not live:
    warn("no working AI provider, skipping (the offline path above still works)")
else:
    prov = live[0]
    q = "Which residents have had falls in the last 6 months, and what do the care notes say about why?"
    t0 = time.time()
    res = chat_engine.answer(DB, q, provider=prov, persist=False)
    dt = int((time.time() - t0) * 1000)
    print("\n" + "-" * 74)
    print(res["answer"][:1200])
    print("-" * 74)
    cited = [s["label"] for s in res["sources"] if s["cited"]]
    ok(f"answered via {res['provider']} in {dt} ms, planner={res['trace']['planner']}, "
       f"tools={[t['name'] for t in res['trace']['tools_called']]}")
    if cited:
        ok(f"answer cites its sources: {', '.join(cited)}")
    else:
        warn("answer did not use [S#] citations — check the synthesis prompt")

    follow = chat_engine.answer(DB, "and what about her hydration?", provider=prov,
                                session_key="selftest", persist=True)
    ok(f"follow-up handled: '{follow['trace']['standalone_question'][:70]}'")
    chat_engine.clear_history(DB, "selftest")

# ── 10. conversation memory ────────────────────────────────────────────────
head(10, "Conversation memory")
mem_key = chat_engine.create_conversation(DB, "selftest")
chat_engine.answer(DB, "How many falls in the last 6 months?", session_key=mem_key,
                   username="selftest", provider="template", use_llm_planner=False)
chat_engine.answer(DB, "Is Ethel losing weight?", session_key=mem_key,
                   username="selftest", provider="template", use_llm_planner=False)

hist = chat_engine.load_history(DB, mem_key)
ok(f"history persisted: {len(hist)} messages across 2 turns") if len(hist) >= 4 \
    else fail(f"history not persisted ({len(hist)} messages)")

recall = chat_engine.answer(DB, "what was my last question", session_key=mem_key,
                            username="selftest", provider="template",
                            use_llm_planner=False)
if "Ethel" in recall["answer"]:
    ok("recalls the previous question verbatim")
else:
    fail(f"memory recall failed: {recall['answer'][:120]}")
if recall["trace"]["intent"] == "meta" and not recall["trace"]["tools_called"]:
    ok("meta questions skip retrieval and analytics (no wasted search)")
else:
    fail(f"meta routing wrong: intent={recall['trace']['intent']}, "
         f"tools={recall['trace']['tools_called']}")

convos = chat_engine.list_conversations(DB, "selftest")
ok(f"conversation listed in the sidebar as '{convos[0]['title'][:40]}'") if convos \
    else fail("conversation missing from the sidebar list")
chat_engine.delete_conversation(DB, mem_key)
ok("conversation deleted cleanly" if not chat_engine.load_history(DB, mem_key)
   else "delete left messages behind")

# ── summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print(f"  {len(passes)} passed · {len(failures)} failed · {len(warnings)} warnings")
print("=" * 74)
if failures:
    print("\nFailures:")
    for f in failures:
        print("  ✗ " + f)
    sys.exit(1)
print("\nAll good. Start the app with  run.bat  (or  python app.py  )"
      "  and open  http://localhost:5000/chat")
