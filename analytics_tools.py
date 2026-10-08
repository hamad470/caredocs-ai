"""
analytics_tools.py — Safe, parameterised analytics layer for the chat assistant
Vector retrieval answers "what was said about X"; it cannot answer "how many
falls did we have last quarter" — counting requires arithmetic over the whole
table, and a top-k retriever by definition sees only k rows. A RAG system that
tries to count from retrieved chunks does not fail loudly, it fails plausibly,
which is worse.

This module therefore provides the second half of the hybrid: a fixed registry
of read-only, hand-written SQL analytics functions. The language model does not
write SQL. It selects a tool by name and supplies typed arguments, which are
validated here and passed as bound parameters. Consequences:

  • SQL injection is impossible — no model-generated string ever reaches SQLite.
  • Results are deterministic and reproducible, so the same question asked
    twice gives the same number (a requirement for clinical audit).
  • Every number the assistant states can be traced to one named function and
    one parameter set, which is recorded in the chat audit log.

This is the "tool-calling / function-calling" pattern (Schick et al., 2023,
Toolformer; OpenAI function calling, 2023) applied with a closed tool set —
the safety-critical variant, as opposed to open text-to-SQL.

Every tool returns the same envelope:
    {tool, title, columns, rows, summary, chart|None, params}
"""

from __future__ import annotations

import sqlite3
import datetime
from collections import defaultdict, Counter


# Helpers

def _conn(db_path: str) -> sqlite3.Connection:
    """Read-only-by-convention connection (the app never writes from here)."""
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def _window(date_from: str | None, date_to: str | None, months_back: int = 12):
    """Default analysis window: the trailing `months_back` months."""
    today = datetime.date.today()
    if not date_to:
        date_to = today.isoformat()
    if not date_from:
        start = today - datetime.timedelta(days=int(30.44 * months_back))
        date_from = start.isoformat()
    return str(date_from)[:10], str(date_to)[:10]


def _resident_filter(resident_id, params: list, column: str = "resident_id") -> str:
    """Build an IN (...) clause with bound parameters — never interpolation."""
    if not resident_id:
        return ""
    ids = resident_id if isinstance(resident_id, (list, tuple, set)) else [resident_id]
    ids = [str(i) for i in ids if i]
    if not ids:
        return ""
    params.extend(ids)
    return f" AND {column} IN ({','.join('?' for _ in ids)})"


def _names(conn) -> dict:
    return {r["resident_id"]: (r["preferred_name"] or r["full_name"])
            for r in conn.execute("SELECT resident_id, full_name, preferred_name FROM residents")}


def _month(d: str) -> str:
    return (d or "")[:7]


def _envelope(tool, title, columns, rows, summary, params, chart=None) -> dict:
    return {"tool": tool, "title": title, "columns": columns, "rows": rows,
            "summary": summary, "chart": chart, "params": params,
            "row_count": len(rows)}


def _pct(n, d) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


# Tools

def list_residents(db_path, **kw):
    conn = _conn(db_path)
    rows = [[r["resident_id"], r["full_name"], r["preferred_name"], r["age"],
             r["room_number"], r["care_type"], r["primary_diagnosis"],
             r["falls_risk"], r["mobility_level"]]
            for r in conn.execute(
                "SELECT * FROM residents WHERE active=1 ORDER BY resident_id")]
    conn.close()
    return _envelope("list_residents", "Current residents",
                     ["ID", "Name", "Known as", "Age", "Room", "Care type",
                      "Primary diagnosis", "Falls risk", "Mobility"],
                     rows, f"{len(rows)} active residents on the register.", kw)


def resident_profile(db_path, resident_id=None, **kw):
    conn = _conn(db_path)
    params = []
    where = _resident_filter(resident_id, params)
    rows, summary_bits = [], []
    for r in conn.execute(f"SELECT * FROM residents WHERE 1=1{where}", params):
        d = dict(r)
        rows.append([d["resident_id"], d["full_name"], d["age"], d["room_number"],
                     d["admission_date"], d["primary_diagnosis"],
                     d["secondary_diagnoses"], d["allergies"], d["mobility_level"],
                     d["continence_needs"], d["nutrition_texture"], d["falls_risk"],
                     d["pressure_sore_risk"], d["dnacpr_status"], d["mental_capacity"],
                     d["gp_name"], f"{d['nok_name']} ({d['nok_relationship']})",
                     d["key_worker"]])
        summary_bits.append(
            f"{d['full_name']} ({d['resident_id']}), age {d['age']}, "
            f"{d['primary_diagnosis']}, falls risk {d['falls_risk']}, "
            f"mobility {d['mobility_level']}, DNACPR {d['dnacpr_status']}")
    conn.close()
    return _envelope("resident_profile", "Resident profile",
                     ["ID", "Name", "Age", "Room", "Admitted", "Primary diagnosis",
                      "Secondary", "Allergies", "Mobility", "Continence", "Diet",
                      "Falls risk", "Pressure risk", "DNACPR", "Capacity", "GP",
                      "Next of kin", "Key worker"],
                     rows, "; ".join(summary_bits) or "No matching resident.", kw)


def incident_summary(db_path, resident_id=None, date_from=None, date_to=None,
                     incident_type=None, severity=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params)
    if incident_type:
        where += " AND LOWER(incident_type) LIKE ?"
        params.append(f"%{str(incident_type).lower()}%")
    if severity:
        where += " AND LOWER(severity)=?"
        params.append(str(severity).lower())

    recs = list(conn.execute(
        f"SELECT * FROM incidents WHERE date BETWEEN ? AND ?{where} ORDER BY date DESC",
        params))
    conn.close()

    by_type = Counter(r["incident_type"] for r in recs)
    by_sev = Counter(r["severity"] for r in recs)
    by_month = Counter(_month(r["date"]) for r in recs)
    open_n = sum(1 for r in recs if (r["status"] or "").lower() == "open")

    rows = [[r["incident_id"] or r["id"], r["date"], names.get(r["resident_id"], r["resident_id"]),
             r["incident_type"], r["severity"], r["location"], r["status"],
             (r["description"] or "")[:160]] for r in recs[:40]]

    months = sorted(by_month)
    chart = {"type": "bar", "labels": months,
             "series": [{"name": "Incidents", "data": [by_month[m] for m in months]}]} if months else None

    summary = (f"{len(recs)} incident(s) between {date_from} and {date_to}. "
               f"By type: {', '.join(f'{k} {v}' for k, v in by_type.most_common()) or 'none'}. "
               f"By severity: {', '.join(f'{k} {v}' for k, v in by_sev.most_common()) or 'none'}. "
               f"{open_n} still open.")
    return _envelope("incident_summary", f"Incidents {date_from} → {date_to}",
                     ["Ref", "Date", "Resident", "Type", "Severity", "Location",
                      "Status", "Description"], rows, summary,
                     {"date_from": date_from, "date_to": date_to,
                      "resident_id": resident_id, "incident_type": incident_type,
                      "severity": severity}, chart)


def falls_analysis(db_path, resident_id=None, date_from=None, date_to=None, **kw):
    """Falls from BOTH sources: formal incident reports and care-note counters."""
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)

    p1 = [date_from, date_to]
    w1 = _resident_filter(resident_id, p1)
    inc = list(conn.execute(
        f"SELECT * FROM incidents WHERE date BETWEEN ? AND ? "
        f"AND LOWER(incident_type) LIKE '%fall%'{w1} ORDER BY date", p1))

    p2 = [date_from, date_to]
    w2 = _resident_filter(resident_id, p2)
    notes = list(conn.execute(
        f"SELECT resident_id, date, falls_this_shift FROM care_notes "
        f"WHERE date BETWEEN ? AND ? AND falls_this_shift > 0{w2}", p2))
    conn.close()

    per_res = defaultdict(lambda: {"incidents": 0, "note_falls": 0, "severities": Counter()})
    by_month = Counter()
    for r in inc:
        per_res[r["resident_id"]]["incidents"] += 1
        per_res[r["resident_id"]]["severities"][r["severity"]] += 1
        by_month[_month(r["date"])] += 1
    for r in notes:
        per_res[r["resident_id"]]["note_falls"] += (r["falls_this_shift"] or 0)

    rows = [[names.get(rid, rid), v["incidents"], v["note_falls"],
             ", ".join(f"{k}:{c}" for k, c in v["severities"].most_common()) or "—"]
            for rid, v in sorted(per_res.items(), key=lambda kv: -kv[1]["incidents"])]

    months = sorted(by_month)
    chart = {"type": "line", "labels": months,
             "series": [{"name": "Fall incidents", "data": [by_month[m] for m in months]}]} if months else None

    total_i = sum(v["incidents"] for v in per_res.values())
    total_n = sum(v["note_falls"] for v in per_res.values())
    peak = max(by_month.items(), key=lambda kv: kv[1]) if by_month else None
    summary = (f"{total_i} fall incident report(s) and {total_n} fall(s) logged in care notes "
               f"between {date_from} and {date_to}. "
               + (f"Busiest month {peak[0]} with {peak[1]}. " if peak else "")
               + (f"Most affected: {rows[0][0]} ({rows[0][1]} reports)." if rows else "No falls recorded."))
    return _envelope("falls_analysis", f"Falls {date_from} → {date_to}",
                     ["Resident", "Incident reports", "Falls in care notes", "Severity mix"],
                     rows, summary,
                     {"date_from": date_from, "date_to": date_to, "resident_id": resident_id},
                     chart)


def fluid_intake_trend(db_path, resident_id=None, date_from=None, date_to=None,
                       target_ml=1500, **kw):
    date_from, date_to = _window(date_from, date_to)
    try:
        target_ml = int(target_ml)
    except Exception:
        target_ml = 1500
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params)
    recs = list(conn.execute(
        f"SELECT resident_id, date, fluid_intake_ml FROM care_notes "
        f"WHERE date BETWEEN ? AND ? AND fluid_intake_ml IS NOT NULL{where}", params))
    conn.close()

    per_res = defaultdict(list)
    per_month = defaultdict(list)
    for r in recs:
        per_res[r["resident_id"]].append(r["fluid_intake_ml"])
        per_month[_month(r["date"])].append(r["fluid_intake_ml"])

    rows = []
    for rid, vals in sorted(per_res.items()):
        below = sum(1 for v in vals if v < target_ml)
        rows.append([names.get(rid, rid), len(vals), round(sum(vals) / len(vals)),
                     min(vals), max(vals), below, _pct(below, len(vals))])

    months = sorted(per_month)
    chart = {"type": "line", "labels": months,
             "series": [{"name": "Mean fluid intake (ml)",
                         "data": [round(sum(per_month[m]) / len(per_month[m])) for m in months]}]} if months else None

    all_vals = [v for vals in per_res.values() for v in vals]
    overall = round(sum(all_vals) / len(all_vals)) if all_vals else 0
    worst = min(rows, key=lambda r: r[2]) if rows else None
    summary = (f"Mean recorded fluid intake {overall} ml/shift across {len(all_vals)} "
               f"records ({date_from} → {date_to}); target {target_ml} ml. "
               + (f"Lowest average: {worst[0]} at {worst[2]} ml with {worst[6]}% of records below target."
                  if worst else "No fluid records in this window."))
    return _envelope("fluid_intake_trend", f"Fluid intake {date_from} → {date_to}",
                     ["Resident", "Records", "Mean ml", "Min", "Max",
                      f"Below {target_ml}ml", "% below"],
                     rows, summary,
                     {"date_from": date_from, "date_to": date_to,
                      "resident_id": resident_id, "target_ml": target_ml}, chart)


def weight_trend(db_path, resident_id=None, date_from=None, date_to=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params)
    recs = list(conn.execute(
        f"SELECT resident_id, date, weight_kg FROM care_notes "
        f"WHERE date BETWEEN ? AND ? AND weight_kg IS NOT NULL{where} ORDER BY date", params))
    conn.close()

    per_res = defaultdict(list)
    for r in recs:
        per_res[r["resident_id"]].append((r["date"], r["weight_kg"]))

    rows, alerts = [], []
    for rid, series in sorted(per_res.items()):
        first, last = series[0], series[-1]
        change = round(last[1] - first[1], 1)
        pct = round(100 * change / first[1], 1) if first[1] else 0
        rows.append([names.get(rid, rid), len(series), first[0], first[1],
                     last[0], last[1], change, f"{pct}%"])
        # ≥5% unintentional loss is the standard malnutrition red flag (MUST tool)
        if pct <= -5:
            alerts.append(f"{names.get(rid, rid)} down {abs(pct)}% ({change} kg)")

    # One resident → the actual trajectory. Several → the comparison that
    # matters clinically, which is percentage change, not absolute weight.
    chart = None
    if len(per_res) == 1:
        rid = next(iter(per_res))
        chart = {"type": "line",
                 "labels": [d for d, _ in per_res[rid]],
                 "series": [{"name": f"{names.get(rid, rid)} weight (kg)",
                             "data": [w for _, w in per_res[rid]]}]}
    elif rows:
        chart = {"type": "bar",
                 "labels": [r[0] for r in rows],
                 "series": [{"name": "Weight change (%)",
                             "data": [float(str(r[7]).rstrip("%")) for r in rows]}]}

    summary = (f"Weight tracked for {len(rows)} resident(s) between {date_from} and {date_to}. "
               + (f"Significant loss (≥5%): {'; '.join(alerts)}." if alerts
                  else "No resident shows a ≥5% weight loss in this window."))
    return _envelope("weight_trend", f"Weight trajectory {date_from} → {date_to}",
                     ["Resident", "Readings", "First date", "First kg",
                      "Latest date", "Latest kg", "Change kg", "Change %"],
                     rows, summary,
                     {"date_from": date_from, "date_to": date_to, "resident_id": resident_id},
                     chart)


def medication_adherence(db_path, resident_id=None, date_from=None, date_to=None,
                         medication_name=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params, "m.resident_id")
    if medication_name:
        where += " AND LOWER(md.medication_name) LIKE ?"
        params.append(f"%{str(medication_name).lower()}%")

    recs = list(conn.execute(
        f"SELECT m.*, md.medication_name, md.is_prn FROM mar_records m "
        f"LEFT JOIN medications md ON md.id = m.medication_id "
        f"WHERE m.date BETWEEN ? AND ?{where}", params))
    conn.close()

    per_res = defaultdict(Counter)
    per_med = defaultdict(Counter)
    reasons = Counter()
    for r in recs:
        status = (r["administered"] or "unknown").lower()
        per_res[r["resident_id"]][status] += 1
        per_med[r["medication_name"] or "unknown"][status] += 1
        if status in ("refused", "no", "not_available") and r["refusal_reason"]:
            reasons[r["refusal_reason"]] += 1

    rows = []
    for rid, c in sorted(per_res.items()):
        total = sum(c.values())
        rows.append([names.get(rid, rid), total, c.get("yes", 0), c.get("refused", 0),
                     c.get("not_available", 0), c.get("no", 0), _pct(c.get("yes", 0), total)])

    med_rows = sorted(per_med.items(), key=lambda kv: -sum(kv[1].values()))[:10]
    chart = {"type": "bar",
             "labels": [m for m, _ in med_rows],
             "series": [{"name": "Adherence %",
                         "data": [_pct(c.get("yes", 0), sum(c.values())) for _, c in med_rows]}]} if med_rows else None

    total_all = sum(sum(c.values()) for c in per_res.values())
    given_all = sum(c.get("yes", 0) for c in per_res.values())
    worst = min(rows, key=lambda r: r[6]) if rows else None
    summary = (f"{total_all} MAR entries between {date_from} and {date_to}; "
               f"overall adherence {_pct(given_all, total_all)}%. "
               + (f"Lowest: {worst[0]} at {worst[6]}%. " if worst else "")
               + (f"Top refusal reasons: {', '.join(f'{k} ({v})' for k, v in reasons.most_common(3))}."
                  if reasons else ""))
    return _envelope("medication_adherence", f"Medication adherence {date_from} → {date_to}",
                     ["Resident", "MAR entries", "Given", "Refused",
                      "Not available", "Not given / omitted", "Adherence %"],
                     rows, summary,
                     {"date_from": date_from, "date_to": date_to,
                      "resident_id": resident_id, "medication_name": medication_name},
                     chart)


def medication_list(db_path, resident_id=None, status="active", **kw):
    conn = _conn(db_path)
    names = _names(conn)
    params = []
    where = _resident_filter(resident_id, params)
    if status and str(status).lower() != "all":
        where += " AND LOWER(status)=?"
        params.append(str(status).lower())
    recs = list(conn.execute(
        f"SELECT * FROM medications WHERE 1=1{where} ORDER BY resident_id, medication_name",
        params))
    conn.close()
    rows = [[names.get(r["resident_id"], r["resident_id"]), r["medication_name"],
             r["dose"], r["route"], r["frequency"], r["indication"],
             "Yes" if r["is_prn"] else "No", "Yes" if r["is_controlled"] else "No",
             r["review_date"], r["status"]] for r in recs]
    prn = sum(1 for r in recs if r["is_prn"])
    ctrl = sum(1 for r in recs if r["is_controlled"])
    return _envelope("medication_list", "Prescribed medications",
                     ["Resident", "Medication", "Dose", "Route", "Frequency",
                      "Indication", "PRN", "Controlled", "Review due", "Status"],
                     rows,
                     f"{len(rows)} prescription(s) ({prn} PRN, {ctrl} controlled drugs).",
                     {"resident_id": resident_id, "status": status})


def wellbeing_trend(db_path, resident_id=None, date_from=None, date_to=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params)
    recs = list(conn.execute(
        f"SELECT * FROM wellbeing WHERE assessment_date BETWEEN ? AND ?{where} "
        f"ORDER BY assessment_date", params))
    conn.close()

    rows = [[names.get(r["resident_id"], r["resident_id"]), r["assessment_date"],
             r["overall_score"], r["physical_health_score"], r["mental_health_score"],
             r["social_engagement_score"], r["personal_care_score"],
             r["nutrition_score"], r["pain_management_score"],
             (r["summary"] or "")[:120]] for r in recs]

    per_month = defaultdict(list)
    for r in recs:
        if r["overall_score"] is not None:
            per_month[_month(r["assessment_date"])].append(r["overall_score"])
    months = sorted(per_month)
    chart = {"type": "line", "labels": months,
             "series": [{"name": "Mean overall wellbeing",
                         "data": [round(sum(per_month[m]) / len(per_month[m]), 1) for m in months]}]} if months else None

    scores = [r["overall_score"] for r in recs if r["overall_score"] is not None]
    trend = ""
    if len(scores) >= 4:
        half = len(scores) // 2
        early, late = sum(scores[:half]) / half, sum(scores[half:]) / (len(scores) - half)
        direction = "improving" if late > early + 0.2 else ("declining" if late < early - 0.2 else "stable")
        trend = f" Trend is {direction} ({round(early,1)} → {round(late,1)})."
    summary = (f"{len(recs)} wellbeing assessment(s) between {date_from} and {date_to}; "
               f"mean overall score {round(sum(scores)/len(scores),1) if scores else 'n/a'}/10.{trend}")
    return _envelope("wellbeing_trend", f"Wellbeing {date_from} → {date_to}",
                     ["Resident", "Date", "Overall", "Physical", "Mental", "Social",
                      "Personal care", "Nutrition", "Pain mgmt", "Summary"],
                     rows, summary,
                     {"date_from": date_from, "date_to": date_to, "resident_id": resident_id},
                     chart)


def mood_and_appetite(db_path, resident_id=None, date_from=None, date_to=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params)
    recs = list(conn.execute(
        f"SELECT resident_id, date, mood, appetite, sleep_quality FROM care_notes "
        f"WHERE date BETWEEN ? AND ?{where}", params))
    conn.close()

    mood = Counter(r["mood"] for r in recs if r["mood"])
    appetite = Counter(r["appetite"] for r in recs if r["appetite"])
    sleep = Counter(r["sleep_quality"] for r in recs if r["sleep_quality"])
    per_res_mood = defaultdict(Counter)
    for r in recs:
        if r["mood"]:
            per_res_mood[r["resident_id"]][r["mood"]] += 1

    rows = []
    for rid, c in sorted(per_res_mood.items()):
        total = sum(c.values())
        low = c.get("low", 0) + c.get("agitated", 0) + c.get("withdrawn", 0)
        rows.append([names.get(rid, rid), total,
                     ", ".join(f"{k} {v}" for k, v in c.most_common(4)),
                     _pct(low, total)])

    labels = [k for k, _ in mood.most_common()]
    chart = {"type": "bar", "labels": labels,
             "series": [{"name": "Mood records", "data": [mood[k] for k in labels]}]} if labels else None
    summary = (f"Across {len(recs)} care notes ({date_from} → {date_to}): "
               f"mood {', '.join(f'{k} {v}' for k, v in mood.most_common(5)) or 'not recorded'}; "
               f"appetite {', '.join(f'{k} {v}' for k, v in appetite.most_common(4)) or 'not recorded'}; "
               f"sleep {', '.join(f'{k} {v}' for k, v in sleep.most_common(4)) or 'not recorded'}.")
    return _envelope("mood_and_appetite", f"Mood, appetite and sleep {date_from} → {date_to}",
                     ["Resident", "Mood records", "Mood mix", "% low/agitated/withdrawn"],
                     rows, summary,
                     {"date_from": date_from, "date_to": date_to, "resident_id": resident_id},
                     chart)


def pain_analysis(db_path, resident_id=None, date_from=None, date_to=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params)
    recs = list(conn.execute(
        f"SELECT resident_id, date, pain_observed, pain_location FROM care_notes "
        f"WHERE date BETWEEN ? AND ?{where}", params))
    conn.close()

    per_res = defaultdict(lambda: {"total": 0, "pain": 0, "sites": Counter()})
    for r in recs:
        d = per_res[r["resident_id"]]
        d["total"] += 1
        # The field is free-ish text ("Mild", "Grimacing", "Verbal complaint",
        # "None observed"), so classify by exclusion rather than by whitelist —
        # a whitelist silently under-counts every vocabulary the seed data adds.
        val = (r["pain_observed"] or "").strip().lower()
        if val and val not in ("no", "none", "none observed", "nil", "n/a", "not observed"):
            d["pain"] += 1
            if r["pain_location"]:
                d["sites"][r["pain_location"]] += 1

    rows = [[names.get(rid, rid), v["total"], v["pain"], _pct(v["pain"], v["total"]),
             ", ".join(f"{k} ({c})" for k, c in v["sites"].most_common(3)) or "—"]
            for rid, v in sorted(per_res.items(), key=lambda kv: -_pct(kv[1]["pain"], kv[1]["total"]))]
    tot = sum(v["total"] for v in per_res.values())
    pain = sum(v["pain"] for v in per_res.values())
    summary = (f"Pain observed in {pain} of {tot} care notes ({_pct(pain, tot)}%) "
               f"between {date_from} and {date_to}."
               + (f" Highest rate: {rows[0][0]} at {rows[0][3]}%." if rows else ""))
    chart = {"type": "bar", "labels": [r[0] for r in rows],
             "series": [{"name": "% of notes recording pain",
                         "data": [r[3] for r in rows]}]} if rows else None
    return _envelope("pain_analysis", f"Pain observations {date_from} → {date_to}",
                     ["Resident", "Notes", "With pain", "% with pain", "Common sites"],
                     rows, summary,
                     {"date_from": date_from, "date_to": date_to, "resident_id": resident_id},
                     chart)


def risk_register(db_path, resident_id=None, risk_level=None, assessment_type=None, **kw):
    """Latest risk assessment per resident per type, with review-overdue flags."""
    conn = _conn(db_path)
    names = _names(conn)
    params = []
    where = _resident_filter(resident_id, params)
    if risk_level:
        where += " AND LOWER(risk_level) LIKE ?"
        params.append(f"%{str(risk_level).lower()}%")
    if assessment_type:
        where += " AND LOWER(assessment_type) LIKE ?"
        params.append(f"%{str(assessment_type).lower()}%")
    recs = list(conn.execute(
        f"SELECT * FROM risk_assessments WHERE 1=1{where} ORDER BY date_assessed DESC",
        params))
    conn.close()

    latest = {}
    for r in recs:
        key = (r["resident_id"], r["assessment_type"])
        if key not in latest:
            latest[key] = r

    today = datetime.date.today().isoformat()
    rows, overdue = [], 0
    for (rid, atype), r in sorted(latest.items()):
        is_overdue = bool(r["review_date"] and r["review_date"] < today)
        overdue += is_overdue
        rows.append([names.get(rid, rid), atype, r["date_assessed"], r["score"],
                     r["risk_level"], r["review_date"],
                     "OVERDUE" if is_overdue else "in date", r["assessed_by"]])

    high = [r for r in rows if str(r[4]).lower() in ("high", "very_high", "very high")]
    summary = (f"{len(rows)} current risk assessment(s) across {len({k[0] for k in latest})} "
               f"resident(s); {len(high)} at high or very high risk; {overdue} review(s) overdue.")
    level_counts = Counter(str(r[4] or "unknown").replace("_", " ").title() for r in rows)
    chart = {"type": "bar", "labels": list(level_counts.keys()),
             "series": [{"name": "Current assessments",
                         "data": list(level_counts.values())}]} if level_counts else None
    return _envelope("risk_register", "Risk register (latest per resident and type)",
                     ["Resident", "Type", "Assessed", "Score", "Level",
                      "Review due", "Review status", "Assessed by"],
                     rows, summary,
                     {"resident_id": resident_id, "risk_level": risk_level,
                      "assessment_type": assessment_type}, chart)


def care_note_activity(db_path, resident_id=None, date_from=None, date_to=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params)
    recs = list(conn.execute(
        f"SELECT * FROM care_notes WHERE date BETWEEN ? AND ?{where}", params))
    conn.close()

    by_month = Counter(_month(r["date"]) for r in recs)
    by_shift = Counter(r["shift"] for r in recs)
    by_status = Counter(r["status"] for r in recs)
    ai_n = sum(1 for r in recs if r["ai_generated"])
    per_res = Counter(r["resident_id"] for r in recs)

    rows = [[names.get(rid, rid), n, round(n / max(len(by_month), 1), 1)]
            for rid, n in per_res.most_common()]
    months = sorted(by_month)
    chart = {"type": "bar", "labels": months,
             "series": [{"name": "Care notes", "data": [by_month[m] for m in months]}]} if months else None
    summary = (f"{len(recs)} care notes between {date_from} and {date_to} "
               f"({ai_n} AI-assisted, {_pct(ai_n, len(recs))}%). "
               f"Shifts: {', '.join(f'{k} {v}' for k, v in by_shift.most_common())}. "
               f"Status: {', '.join(f'{k} {v}' for k, v in by_status.most_common())}.")
    return _envelope("care_note_activity", f"Documentation activity {date_from} → {date_to}",
                     ["Resident", "Notes", "Avg per month"], rows, summary,
                     {"date_from": date_from, "date_to": date_to, "resident_id": resident_id},
                     chart)


def handover_escalations(db_path, resident_id=None, date_from=None, date_to=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params)
    try:
        recs = list(conn.execute(
            f"SELECT * FROM handovers WHERE date BETWEEN ? AND ?{where} ORDER BY date DESC",
            params))
    except sqlite3.Error:
        recs = []
    conn.close()

    esc = [r for r in recs if str(r["escalation_required"] or "").lower() in ("yes", "y", "true", "1")]
    rows = [[r["date"], names.get(r["resident_id"], r["resident_id"]),
             f"{r['shift_ending']}→{r['shift_starting']}", r["compiled_by"],
             (r["escalation_details"] or r["concerns_next_shift"] or "")[:160]]
            for r in esc[:40]]
    summary = (f"{len(esc)} of {len(recs)} handovers between {date_from} and {date_to} "
               f"required escalation ({_pct(len(esc), len(recs))}%).")
    per_month = Counter(_month(r["date"]) for r in esc)
    months = sorted(per_month)
    chart = {"type": "bar", "labels": months,
             "series": [{"name": "Escalations",
                         "data": [per_month[m] for m in months]}]} if months else None
    return _envelope("handover_escalations", f"Handover escalations {date_from} → {date_to}",
                     ["Date", "Resident", "Shift", "Compiled by", "Detail"],
                     rows, summary,
                     {"date_from": date_from, "date_to": date_to, "resident_id": resident_id},
                     chart)


def family_contact_summary(db_path, resident_id=None, date_from=None, date_to=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    names = _names(conn)
    params = [date_from, date_to]
    where = _resident_filter(resident_id, params)
    try:
        recs = list(conn.execute(
            f"SELECT * FROM family_comms WHERE date BETWEEN ? AND ?{where} ORDER BY date DESC",
            params))
    except sqlite3.Error:
        recs = []
    conn.close()

    by_type = Counter(r["comm_type"] for r in recs)
    rows = [[r["date"], names.get(r["resident_id"], r["resident_id"]), r["comm_type"],
             r["direction"], r["family_contact"], r["staff_member"],
             (r["subject"] or "")[:80], (r["follow_up"] or "")[:40]] for r in recs[:40]]
    summary = (f"{len(recs)} family communication(s) between {date_from} and {date_to}: "
               f"{', '.join(f'{k} {v}' for k, v in by_type.most_common()) or 'none'}.")
    chart = {"type": "bar", "labels": [k for k, _ in by_type.most_common()],
             "series": [{"name": "Contacts",
                         "data": [v for _, v in by_type.most_common()]}]} if by_type else None
    return _envelope("family_contact_summary", f"Family contact {date_from} → {date_to}",
                     ["Date", "Resident", "Type", "Direction", "Contact",
                      "Staff", "Subject", "Follow-up"], rows, summary,
                     {"date_from": date_from, "date_to": date_to, "resident_id": resident_id},
                     chart)


def compliance_overview(db_path, date_from=None, date_to=None, **kw):
    date_from, date_to = _window(date_from, date_to)
    conn = _conn(db_path)
    today = datetime.date.today().isoformat()
    rows = []
    flags = []
    for r in conn.execute("SELECT * FROM residents WHERE active=1 ORDER BY resident_id"):
        rid = r["resident_id"]
        notes = conn.execute(
            "SELECT COUNT(*) FROM care_notes WHERE resident_id=? AND date BETWEEN ? AND ?",
            (rid, date_from, date_to)).fetchone()[0]
        open_inc = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE resident_id=? AND status='open'",
            (rid,)).fetchone()[0]
        last_wb = conn.execute(
            "SELECT MAX(assessment_date) FROM wellbeing WHERE resident_id=?",
            (rid,)).fetchone()[0]
        overdue = conn.execute(
            "SELECT COUNT(*) FROM risk_assessments WHERE resident_id=? AND review_date < ?",
            (rid, today)).fetchone()[0]
        plan = conn.execute(
            "SELECT MAX(effective_from) FROM care_plans WHERE resident_id=? "
            "AND LOWER(COALESCE(status,'')) IN ('active','approved','current')",
            (rid,)).fetchone()[0]
        status = "green"
        if open_inc or overdue:
            status = "red"
        elif notes < 10:
            status = "amber"
        if status == "red":
            flags.append(f"{r['preferred_name'] or r['full_name']} "
                         f"({open_inc} open incident(s), {overdue} overdue risk review(s))")
        rows.append([r["preferred_name"] or r["full_name"], notes, open_inc,
                     last_wb or "never", overdue, plan or "none", status.upper()])
    conn.close()
    summary = (f"Compliance snapshot {date_from} → {date_to}. "
               + (f"Attention needed: {'; '.join(flags)}." if flags
                  else "No open incidents or overdue risk reviews."))
    chart = {"type": "bar", "labels": [r[0] for r in rows],
             "series": [{"name": "Care notes in period",
                         "data": [r[1] for r in rows]}]} if rows else None
    return _envelope("compliance_overview", f"Compliance overview {date_from} → {date_to}",
                     ["Resident", "Care notes", "Open incidents", "Last wellbeing",
                      "Overdue risk reviews", "Care plan from", "RAG"],
                     rows, summary, {"date_from": date_from, "date_to": date_to}, chart)


def resident_timeline(db_path, resident_id=None, date_from=None, date_to=None,
                      limit=40, **kw):
    """Chronological key events — the backbone for 'what has happened with X'."""
    date_from, date_to = _window(date_from, date_to)
    try:
        limit = max(5, min(int(limit), 120))
    except Exception:
        limit = 40
    conn = _conn(db_path)
    names = _names(conn)
    events = []

    p = [date_from, date_to]; w = _resident_filter(resident_id, p)
    for r in conn.execute(f"SELECT * FROM incidents WHERE date BETWEEN ? AND ?{w}", p):
        events.append([r["date"], names.get(r["resident_id"], r["resident_id"]), "Incident",
                       f"{r['incident_type']} ({r['severity']}) — {(r['description'] or '')[:120]}"])

    p = [date_from, date_to]; w = _resident_filter(resident_id, p)
    for r in conn.execute(f"SELECT * FROM risk_assessments WHERE date_assessed BETWEEN ? AND ?{w}", p):
        events.append([r["date_assessed"], names.get(r["resident_id"], r["resident_id"]),
                       "Risk assessment",
                       f"{r['assessment_type']} — level {r['risk_level']} (score {r['score']})"])

    p = [date_from, date_to]; w = _resident_filter(resident_id, p)
    for r in conn.execute(f"SELECT * FROM wellbeing WHERE assessment_date BETWEEN ? AND ?{w}", p):
        events.append([r["assessment_date"], names.get(r["resident_id"], r["resident_id"]),
                       "Wellbeing review",
                       f"Overall {r['overall_score']}/10 — {(r['summary'] or '')[:110]}"])

    p = [date_from, date_to]; w = _resident_filter(resident_id, p)
    for r in conn.execute(f"SELECT * FROM care_plans WHERE effective_from BETWEEN ? AND ?{w}", p):
        events.append([r["effective_from"], names.get(r["resident_id"], r["resident_id"]),
                       "Care plan", f"Version {r['version']} ({r['status']})"])

    p = [date_from, date_to]; w = _resident_filter(resident_id, p)
    for r in conn.execute(f"SELECT * FROM family_comms WHERE date BETWEEN ? AND ?{w}", p):
        events.append([r["date"], names.get(r["resident_id"], r["resident_id"]),
                       "Family contact", f"{r['comm_type']} — {(r['subject'] or '')[:110]}"])
    conn.close()

    events.sort(key=lambda e: e[0], reverse=True)
    rows = events[:limit]
    kinds = Counter(e[2] for e in events)
    summary = (f"{len(events)} key event(s) between {date_from} and {date_to}: "
               f"{', '.join(f'{k} {v}' for k, v in kinds.most_common())}."
               + (f" Showing the {len(rows)} most recent." if len(events) > len(rows) else ""))
    per_month = Counter(_month(e[0]) for e in events)
    months = sorted(per_month)
    chart = {"type": "line", "labels": months,
             "series": [{"name": "Key events",
                         "data": [per_month[m] for m in months]}]} if months else None
    return _envelope("resident_timeline", f"Event timeline {date_from} → {date_to}",
                     ["Date", "Resident", "Event", "Detail"], rows, summary,
                     {"date_from": date_from, "date_to": date_to,
                      "resident_id": resident_id, "limit": limit}, chart)


# Registry — this is what the planner model sees

_COMMON = {
    "resident_id": "Resident ID such as RES001, or a list of IDs. Omit for the whole home.",
    "date_from": "ISO start date (YYYY-MM-DD). Defaults to 12 months ago.",
    "date_to": "ISO end date (YYYY-MM-DD). Defaults to today.",
}

TOOLS: dict[str, dict] = {
    "list_residents": {
        "fn": list_residents,
        "description": "List all active residents with room, care type, diagnosis and falls risk.",
        "args": {},
    },
    "resident_profile": {
        "fn": resident_profile,
        "description": "Full profile for one or more residents: diagnoses, allergies, mobility, "
                       "diet, DNACPR, capacity, GP, next of kin, key worker.",
        "args": {"resident_id": _COMMON["resident_id"]},
    },
    "incident_summary": {
        "fn": incident_summary,
        "description": "Counts and details of incidents by type, severity and month. "
                       "Use for 'how many incidents', 'what incidents happened', safeguarding questions.",
        "args": {**_COMMON,
                 "incident_type": "Optional filter, e.g. fall, medication error, behaviour, skin.",
                 "severity": "Optional filter: minor, moderate, major, severe."},
    },
    "falls_analysis": {
        "fn": falls_analysis,
        "description": "Falls specifically, combining incident reports and care-note fall counters, "
                       "broken down per resident and per month.",
        "args": _COMMON,
    },
    "fluid_intake_trend": {
        "fn": fluid_intake_trend,
        "description": "Hydration: mean/min/max fluid intake per resident and per month, and how "
                       "often intake fell below target. Use for dehydration questions.",
        "args": {**_COMMON, "target_ml": "Daily target in ml (default 1500)."},
    },
    "weight_trend": {
        "fn": weight_trend,
        "description": "Weight trajectory per resident with change in kg and %, flagging ≥5% loss "
                       "(malnutrition red flag). Use for weight loss / nutrition questions.",
        "args": _COMMON,
    },
    "medication_adherence": {
        "fn": medication_adherence,
        "description": "MAR adherence: given / refused / not available counts and adherence %, per "
                       "resident and per medication, with refusal reasons.",
        "args": {**_COMMON, "medication_name": "Optional medication name filter."},
    },
    "medication_list": {
        "fn": medication_list,
        "description": "Current prescriptions: dose, route, frequency, indication, PRN and "
                       "controlled-drug flags, review dates.",
        "args": {"resident_id": _COMMON["resident_id"],
                 "status": "active (default), stopped, or all."},
    },
    "wellbeing_trend": {
        "fn": wellbeing_trend,
        "description": "Wellbeing assessment scores over time (overall, physical, mental, social, "
                       "personal care, nutrition, pain) with direction of travel.",
        "args": _COMMON,
    },
    "mood_and_appetite": {
        "fn": mood_and_appetite,
        "description": "Distribution of recorded mood, appetite and sleep quality from care notes, "
                       "including the share of low/agitated/withdrawn days.",
        "args": _COMMON,
    },
    "pain_analysis": {
        "fn": pain_analysis,
        "description": "How often pain was observed, per resident, with the most common pain sites.",
        "args": _COMMON,
    },
    "risk_register": {
        "fn": risk_register,
        "description": "Latest risk assessment per resident per type (falls, pressure, nutrition, "
                       "behaviour), with scores, levels and overdue review flags.",
        "args": {"resident_id": _COMMON["resident_id"],
                 "risk_level": "Optional: low, medium, high, very_high.",
                 "assessment_type": "Optional: falls, pressure, nutrition, behaviour, environment."},
    },
    "care_note_activity": {
        "fn": care_note_activity,
        "description": "Documentation volume: care notes per resident and per month, shift and "
                       "status breakdown, share that was AI-assisted.",
        "args": _COMMON,
    },
    "handover_escalations": {
        "fn": handover_escalations,
        "description": "Shift handovers that required escalation, with the reason given.",
        "args": _COMMON,
    },
    "family_contact_summary": {
        "fn": family_contact_summary,
        "description": "Family communications: calls, visits, emails, meetings, and follow-ups owed.",
        "args": _COMMON,
    },
    "compliance_overview": {
        "fn": compliance_overview,
        "description": "Home-wide compliance snapshot per resident: note volume, open incidents, "
                       "last wellbeing review, overdue risk reviews, active care plan, RAG status.",
        "args": {"date_from": _COMMON["date_from"], "date_to": _COMMON["date_to"]},
    },
    "resident_timeline": {
        "fn": resident_timeline,
        "description": "Chronological list of key events (incidents, risk assessments, wellbeing "
                       "reviews, care plan versions, family contact). Best for open questions such "
                       "as 'what has happened with Maggie over the last year'.",
        "args": {**_COMMON, "limit": "Max events to return (default 40)."},
    },
}


def tool_catalogue() -> str:
    """Compact catalogue string injected into the planner prompt."""
    lines = []
    for name, spec in TOOLS.items():
        arg_desc = "; ".join(f"{k}: {v}" for k, v in spec["args"].items()) or "no arguments"
        lines.append(f"- {name}: {spec['description']} ARGS → {arg_desc}")
    return "\n".join(lines)


def run_tool(db_path: str, name: str, args: dict | None = None) -> dict:
    """Execute a registered tool with validated arguments."""
    spec = TOOLS.get(name)
    if not spec:
        return {"tool": name, "error": f"Unknown analytics tool '{name}'.",
                "columns": [], "rows": [], "summary": "", "chart": None, "params": args or {}}
    args = {k: v for k, v in (args or {}).items() if v not in (None, "", [], {})}
    # Drop any argument the tool does not declare — the model is not trusted to
    # stay inside the schema, so the schema is enforced here.
    allowed = set(spec["args"].keys()) | {"resident_id", "date_from", "date_to"}
    clean = {k: v for k, v in args.items() if k in allowed}
    try:
        return spec["fn"](db_path, **clean)
    except Exception as e:                                   # never break the chat
        return {"tool": name, "error": f"{type(e).__name__}: {e}", "columns": [],
                "rows": [], "summary": f"The {name} analysis could not be completed.",
                "chart": None, "params": clean}


def render_tool_result(result: dict, max_rows: int = 25) -> str:
    """Markdown rendering of a tool result for injection into the LLM prompt."""
    if result.get("error"):
        return f"[{result.get('tool')}] ERROR: {result['error']}"
    lines = [f"### {result.get('title', result.get('tool'))}",
             f"Summary: {result.get('summary', '')}"]
    cols, rows = result.get("columns", []), result.get("rows", [])
    if cols and rows:
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for row in rows[:max_rows]:
            lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
        if len(rows) > max_rows:
            lines.append(f"_({len(rows) - max_rows} further rows omitted)_")
    return "\n".join(lines)
