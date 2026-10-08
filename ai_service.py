"""
AI Service — generates care narrative text from structured form data.

One cloud provider is supported, plus an always-available offline fallback.
Keys may be supplied EITHER as environment variables (e.g. via run.bat) OR
saved inside the app at Settings → AI Settings, which writes them to
ai_config.json and injects them into the environment. Both routes end in the
same place, so the two methods are interchangeable.

    Backend    Env var              Model                     Cost
    ─────────  ───────────────────  ────────────────────────  ─────────────────
    Gemini     GEMINI_API_KEYS      gemini flash (discovered) generous free tier
    Template   (none)               deterministic strings     free, works offline

Why one provider
An earlier version of this module could call Anthropic, Google or Groq and
fell through from one to the next when a call failed. For a demonstration that
is convenient. For a dissertation it is a defect, for two reasons.

The first is attribution. A care note is a legal record, and `ai_generated` is
recorded against every generated field. If three vendors can answer the same
request, the record says a model wrote the text but not which model, and a
reader auditing a resident's file six months later cannot reconstruct what
produced it.

The second is reproducibility. Chapter 7 of the dissertation compares
generation with and without retrieval on identical prompts. Silent failover
means an arm of that comparison could be served by a different model family
partway through, because one free tier hit its daily quota — so a difference
between arms might be a difference between vendors rather than a difference
between conditions, and nothing in the output would reveal it.

Several Gemini keys may still be configured, and they are tried in order. That
is a different kind of variation: every key reaches the same models, so
rotation changes whether the system can answer, never what it answers.
AI_PROVIDER=template pins the offline path, which the evaluation harness uses
to establish the floor the system produces with no model at all.

PRODUCTION DEPLOYMENT — DATA PROTECTION WARNING (UK GDPR / DPA 2018)
This module transmits resident data (name, diagnosis, medications, DNACPR
status, mental capacity, care notes, family contact details) to a third-party
AI API (Google) as part of prompt construction.

In a RESEARCH / SYNTHETIC DATA context (current use):
  → Data is entirely fictional — no UK GDPR Article 4(1) personal data
    is transmitted. No regulatory obligation applies.

In a PRODUCTION context with REAL resident data:
  1. A signed Data Processing Agreement (DPA) under UK GDPR Article 28
     with the API provider is MANDATORY before any data is transmitted.
  2. A Data Protection Impact Assessment (DPIA) under Article 35 must be
     completed — this system qualifies as high-risk AI processing of
     special category health data (Article 9).
  3. Residents must be informed of AI-assisted documentation via a clear
     Privacy Notice meeting UK GDPR Articles 13–14.
  4. Consider replacing the cloud API with a locally-hosted LLM (e.g. Llama
     via Ollama) to eliminate cross-border data transfer entirely.
  5. All AI-generated text must remain flagged (ai_generated=True) and
     subject to mandatory human review before entering the clinical record.

  6. FREE TIERS ARE NOT SUITABLE FOR REAL DATA. Providers commonly reserve
     the right to retain and train on free-tier traffic, which would be an
     unlawful onward processing of Article 9 health data. Any production
     deployment must use a paid tier whose terms explicitly exclude training,
     under a signed Article 28 DPA — or, preferably, a locally-hosted model.
     This applies with particular force here, because the multi-key rotation
     described above exists precisely to stay inside free-tier quotas.

This warning must NOT be removed prior to production deployment.
DO NOT send real resident data to any API without completing steps 1–3.
"""
import os
import json
import urllib.request
import urllib.error


# Sent on every request: the default "Python-urllib/3.x" User-Agent is rejected
# by some CDN edges with HTTP 403 before the API key is ever checked.
USER_AGENT = os.environ.get("CAREHOME_USER_AGENT",
                            "CareHomeMVP/1.0 (+dissertation-research; python-urllib)")

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# One system prompt, sent verbatim on every call. Fixing it here rather than
# per-call is what makes the with-and-without-retrieval comparison in Chapter 7
# a comparison of conditions rather than of instructions.
SYSTEM_PROMPT = (
    "You are a professional care home documentation assistant. "
    "Write clear, factual, person-centred care documentation in the "
    "first-person professional style used in UK care homes. "
    "Use 2–4 sentences unless told otherwise. "
    "Never invent clinical information not provided."
)

MAX_TOKENS  = 400
TEMPERATURE = 0.4   # low — factual consistency matters more than variety

# Records which backend produced the most recent narrative, so the UI and the
# evaluation harness can report it. Set by generate_narrative().
_LAST_PROVIDER = None

# Google retires model names, and a retired name answers HTTP 404 rather than
# anything more helpful. Rather than hard-coding one name that may stop working
# mid-project, these are tried in order and the first that responds is cached in
# _GEMINI_WORKING_MODEL for the rest of the process. llm_client discovers names
# per key and is preferred; this list is the fallback when it is unavailable.
# Override the whole list with the GEMINI_MODEL environment variable.
GEMINI_MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]

_GEMINI_WORKING_MODEL = None


def _gemini_request(model: str, key: str, prompt: str) -> str | None:
    """Single Gemini generateContent call. Returns None on any failure."""
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_TOKENS,
        },
    }).encode()
    req = urllib.request.Request(
        f"{GEMINI_API_BASE}/{model}:generateContent",
        data=payload,
        headers={
            # Header auth rather than ?key=... so the API key never appears in
            # a URL, where it could be captured by proxy or server logs.
            "x-goog-api-key": key,
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _configured_gemini_keys() -> list[str]:
    """
    Every configured key, in preference order.

    Reads GEMINI_API_KEYS (comma-separated) first and falls back to the single
    GEMINI_API_KEY / GOOGLE_API_KEY variables, so a one-key setup written by an
    earlier version still works. Duplicates are dropped, because pasting the
    same key into two slots is an easy mistake and trying it twice only doubles
    the time spent discovering it is out of quota.
    """
    raw = os.environ.get("GEMINI_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    for single in (os.environ.get("GEMINI_API_KEY", ""),
                   os.environ.get("GOOGLE_API_KEY", "")):
        if single and single not in keys:
            keys.append(single)
    out = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def _call_gemini(prompt: str) -> str | None:
    """
    Google Gemini via the Generative Language REST API.

    Uses only urllib from the standard library — deliberately no
    google-generativeai SDK, so the dependency list stays short.

    Several keys may be configured. They are tried in order until one answers,
    which matters because free-tier quotas are small and a full evaluation run
    makes hundreds of calls. Rotation is a quota device and nothing more: every
    key reaches the same models, so which one answered cannot change what the
    model said.
    """
    global _GEMINI_WORKING_MODEL

    keys = _configured_gemini_keys()
    if not keys:
        return None

    # Preferred path: llm_client already implements key rotation with per-key
    # cooldown after a quota rejection, and per-key model discovery across both
    # API versions. Hard-coded model lists are the usual cause of "404 ... is
    # not found for API version v1beta", and maintaining two of them in one
    # codebase means fixing that bug twice.
    try:
        import llm_client
        text = llm_client.gemini_generate(prompt, SYSTEM_PROMPT,
                                          MAX_TOKENS, TEMPERATURE, timeout=25)
        if text:
            chosen = llm_client._GEMINI_CACHE.get("chosen")
            if chosen:
                _GEMINI_WORKING_MODEL = chosen
            return text
    except Exception:
        pass          # fall through to the direct path below

    # Direct fallback, used only if llm_client is unavailable or errored. It
    # rotates keys itself rather than reading only the first one, so a quota
    # exhaustion on key one does not silently disable the whole fallback.
    pinned = os.environ.get("GEMINI_MODEL", "") or _GEMINI_WORKING_MODEL
    if pinned:
        candidates = [pinned]
    else:
        candidates = list(GEMINI_MODEL_CANDIDATES)
        try:
            import llm_client
            discovered = llm_client.discover_gemini_models().get("models") or []
            if discovered:
                candidates = discovered[:4]
        except Exception:
            pass

    for key in keys:
        for model in candidates:
            try:
                text = _gemini_request(model, key, prompt)
                if text:
                    _GEMINI_WORKING_MODEL = model
                    return text
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 403):
                    break          # this key is bad — no model name will help
                continue           # 404/429 — next model, then the next key
            except Exception:
                continue
    return None


def get_gemini_model() -> str:
    """The Gemini model actually in use, for display and for the audit trail."""
    if os.environ.get("GEMINI_MODEL", ""):
        return os.environ["GEMINI_MODEL"]
    if _GEMINI_WORKING_MODEL:
        return _GEMINI_WORKING_MODEL
    try:
        import llm_client
        chosen = llm_client.discover_gemini_models().get("chosen")
        if chosen:
            return chosen
    except Exception:
        pass
    return GEMINI_MODEL_CANDIDATES[0]


def _template_care_note(data: dict) -> str:
    name = data.get("preferred_name", "The resident")
    mood = data.get("mood", "").lower()
    appetite = data.get("appetite", "").lower()
    fluid = data.get("fluid_intake_ml", "")
    personal_care = data.get("personal_care", "")
    activity = data.get("activity_description", "")
    concerns = data.get("concerns", "")
    pain = data.get("pain_observed", "No")

    parts = [f"{name} was attended to during the {data.get('shift','').lower()} shift."]
    if personal_care:
        parts.append(f"Personal care was {personal_care.lower()}.")
    if mood:
        parts.append(f"{name} appeared {mood} in mood.")
    if appetite:
        parts.append(f"Appetite was {appetite}.")
    if fluid:
        parts.append(f"Fluid intake recorded as {fluid}ml.")
    if activity:
        parts.append(f"Activities: {activity}.")
    if pain and pain.lower() not in ("no", "none", ""):
        parts.append(f"Pain was observed ({data.get('pain_location','unspecified')}); appropriate action taken.")
    if concerns:
        parts.append(f"Concerns noted: {concerns}.")
    return " ".join(parts)


def _template_incident(data: dict) -> str:
    name = data.get("preferred_name", "The resident")
    inc_type = data.get("incident_type", "incident")
    location = data.get("location", "the care home")
    time = data.get("time", "")
    severity = data.get("severity", "")
    immediate = data.get("immediate_actions", "")
    injuries = data.get("injuries", "No injuries noted")

    parts = [
        f"At {time}, a {severity.lower()} {inc_type.lower()} occurred involving {name} in {location}.",
        injuries + ".",
    ]
    if immediate:
        parts.append(f"Immediate actions taken: {immediate}.")
    return " ".join(parts)


def _template_handover(data: dict) -> str:
    name = data.get("preferred_name", "The resident")
    shift = data.get("shift_ending", "")
    concerns = data.get("concerns_next_shift", "")
    outstanding = data.get("outstanding_tasks", "")
    escalation = data.get("escalation_required", "No")
    fluid = data.get("fluid_target_met", "")

    parts = [f"Handover summary for {name} following the {shift} shift."]
    if fluid:
        parts.append(f"Fluid target {'met' if fluid.lower()=='yes' else 'not met'}.")
    if concerns:
        parts.append(f"Concerns for next shift: {concerns}.")
    if outstanding:
        parts.append(f"Outstanding tasks: {outstanding}.")
    if escalation and escalation.lower() not in ("no",""):
        parts.append("Escalation required — see details above.")
    return " ".join(parts)


def _template_wellbeing(data: dict) -> str:
    """
    Offline fallback. It is deterministic and it is deliberately plainer than
    the model output, but it still reports movement between reviews rather than
    listing this period's numbers, because a score with no comparison tells a
    reader nothing about whether the resident is doing better or worse.
    """
    name = data.get("preferred_name", "The resident")
    period = data.get("period_covered", "this period")
    labels = {
        "physical_health_score":   "physical health",
        "mental_health_score":     "mental health",
        "social_engagement_score": "social engagement",
        "personal_care_score":     "personal care",
        "nutrition_score":         "nutrition and hydration",
        "pain_management_score":   "pain management",
    }
    prev = data.get("previous_scores") or {}

    current, improved, declined = [], [], []
    for key, label in labels.items():
        val = data.get(key)
        if val in (None, "", "None"):
            continue
        current.append(f"{label} {val}/10")
        try:
            diff = int(val) - int(prev.get(key))
        except (TypeError, ValueError):
            continue
        if diff > 0:
            improved.append(f"{label} (+{diff})")
        elif diff < 0:
            declined.append(f"{label} ({diff})")

    parts = [f"Wellbeing review completed for {name} covering {period}."]
    overall = data.get("overall_score", "")
    prev_overall = data.get("previous_overall", "")
    if overall and prev_overall:
        try:
            d = int(overall) - int(prev_overall)
            direction = "an improvement on" if d > 0 else ("a decline from" if d < 0 else "unchanged from")
            parts.append(f"The overall score is {overall}/10, {direction} {prev_overall}/10 at the previous review.")
        except (TypeError, ValueError):
            parts.append(f"The overall score is {overall}/10.")
    elif overall:
        parts.append(f"The overall score is {overall}/10, the first recorded for this resident.")
    if current:
        parts.append("Domain scores this period were " + ", ".join(current) + ".")
    if improved:
        parts.append("Improvement was recorded in " + ", ".join(improved) + ".")
    if declined:
        parts.append("A fall in score was recorded in " + ", ".join(declined) + ".")

    for field, lead in (("concerns", "Concerns raised this period"),
                        ("positive_outcomes", "Positive outcomes noted"),
                        ("resident_voice", f"In {name}'s own words"),
                        ("family_feedback", "Family feedback"),
                        ("actions_next_period", "Planned for the next period")):
        val = (data.get(field) or "").strip()
        if val:
            parts.append(f"{lead}: {val.rstrip('.')}.")
    return " ".join(parts)


def _template_risk(data: dict) -> str:
    name = data.get("preferred_name", "The resident")
    a_type = data.get("assessment_type", "risk")
    risk_level = data.get("risk_level", "")
    factors = data.get("risk_factors", "")
    interventions = data.get("interventions", "")

    parts = [f"A {a_type} risk assessment was completed for {name}."]
    if risk_level:
        parts.append(f"Risk level identified as {risk_level}.")
    if factors:
        parts.append(f"Key risk factors: {factors}.")
    if interventions:
        parts.append(f"Interventions in place: {interventions}.")
    return " ".join(parts)


def _template_care_plan_section(data: dict) -> str:
    name = data.get("preferred_name", "The resident")
    section = data.get("section", "care")
    diag = data.get("primary_diagnosis", "")
    return (f"{name}'s {section} plan takes into account their diagnosis of {diag}. "
            f"Care is delivered in line with their individual needs and preferences. "
            f"Staff should refer to the full care plan for detailed guidance.")


def _template_family_comm(data: dict) -> str:
    name = data.get("preferred_name", "the resident")
    contact = data.get("family_contact", "the family")
    subject = data.get("subject", "a routine update")
    staff = data.get("staff_member", "staff")
    return (f"Dear {contact},\n\nI hope this message finds you well. I am writing to provide you with "
            f"an update regarding {name}.\n\n{subject}.\n\n"
            f"Please do not hesitate to contact us should you have any questions.\n\n"
            f"Kind regards,\n{staff}\nSunrise Care Home")


def _build_prompt(report_type: str, data: dict) -> str:
    name = data.get("preferred_name", data.get("full_name", "the resident"))
    base = f"Resident: {name}. "
    if report_type == "care_note":
        return (
            base + f"Shift: {data.get('shift')}. "
            f"Personal care: {data.get('personal_care')}. "
            f"Mood: {data.get('mood')}. Appetite: {data.get('appetite')}. "
            f"Fluid intake: {data.get('fluid_intake_ml')}ml. "
            f"Pain observed: {data.get('pain_observed')} ({data.get('pain_location','')}). "
            f"Activities: {data.get('activity_description','')}. "
            f"Concerns: {data.get('concerns','')}. "
            f"Actions taken: {data.get('actions_taken','')}. "
            "Write a 3–4 sentence professional care note narrative for this UK care home shift."
        )
    elif report_type == "incident":
        return (
            base + f"Incident type: {data.get('incident_type')}. "
            f"Severity: {data.get('severity')}. Time: {data.get('time')}. "
            f"Location: {data.get('location')}. "
            f"Witness: {data.get('witness_name','')}. "
            f"Description: {data.get('description','')}. "
            f"Injuries: {data.get('injuries','')}. "
            f"Immediate actions: {data.get('immediate_actions','')}. "
            "Write a 3–4 sentence factual incident report narrative for a UK care home."
        )
    elif report_type == "handover":
        return (
            base + f"Shift ending: {data.get('shift_ending')}. "
            f"Care completed: {data.get('care_completed','')}. "
            f"Concerns for next shift: {data.get('concerns_next_shift','')}. "
            f"Outstanding tasks: {data.get('outstanding_tasks','')}. "
            f"Fluid target met: {data.get('fluid_target_met','')}. "
            f"Incidents: {data.get('incidents_this_shift',0)}. "
            "Write a professional handover summary for a UK care home."
        )
    elif report_type == "wellbeing":
        # Six domain scores, the free-text notes attached to each, and the
        # previous assessment where one exists. The comparison matters: a
        # wellbeing review is a periodic instrument, so the reader wants the
        # direction of travel, not a restatement of this month's numbers.
        domains = [
            ("Physical health",   "physical_health_score",   "physical_notes"),
            ("Mental health",     "mental_health_score",     "mental_notes"),
            ("Social engagement", "social_engagement_score", "social_notes"),
            ("Personal care",     "personal_care_score",     None),
            ("Nutrition and hydration", "nutrition_score",   None),
            ("Pain management",   "pain_management_score",   None),
        ]
        rows = []
        for label, score_key, note_key in domains:
            val = data.get(score_key)
            if val in (None, "", "None"):
                continue
            prev = (data.get("previous_scores") or {}).get(score_key)
            move = ""
            if prev not in (None, "", "None"):
                try:
                    diff = int(val) - int(prev)
                    move = (f" (was {prev}/10, {'up' if diff > 0 else 'down'} {abs(diff)})"
                            if diff else f" (unchanged from {prev}/10)")
                except (TypeError, ValueError):
                    move = ""
            note = (data.get(note_key) or "").strip() if note_key else ""
            rows.append(f"- {label}: {val}/10{move}." + (f" Staff note: {note}" if note else ""))

        overall = data.get("overall_score") or ""
        prev_overall = data.get("previous_overall") or ""
        optional = [
            ("Goals progress",   data.get("goals_progress", "")),
            ("Concerns raised",  data.get("concerns", "")),
            ("Positive outcomes", data.get("positive_outcomes", "")),
            ("Planned actions",  data.get("actions_next_period", "")),
            ("Resident's own words", data.get("resident_voice", "")),
            ("Family feedback",  data.get("family_feedback", "")),
        ]
        extras = "\n".join(f"- {k}: {v.strip()}" for k, v in optional if (v or "").strip())

        return (
            f"You are a senior carer writing the summary section of a periodic "
            f"wellbeing review in a UK care home.\n\n"
            f"RESIDENT: {name}\n"
            f"PERIOD REVIEWED: {data.get('period_covered', 'this period')}\n"
            + (f"OVERALL SCORE: {overall}/10"
               + (f" (previous review: {prev_overall}/10)\n" if prev_overall else "\n")
               if overall else "")
            + "\nDOMAIN SCORES (1 = very poor, 10 = excellent):\n"
            + ("\n".join(rows) if rows else "- No domain scores recorded.")
            + (f"\n\nADDITIONAL INFORMATION:\n{extras}" if extras else "")
            + (f"\n\nPREVIOUS REVIEW SUMMARY:\n{data['previous_summary'].strip()}"
               if (data.get("previous_summary") or "").strip() else "")
            + "\n\nWrite four to six sentences of continuous prose. Open with the "
              "resident's overall position this period. Name the domains that moved "
              "and say in which direction, and where the record explains a movement, "
              "give that explanation. Quote the resident's own words if they are "
              "provided. Close with what the team will do next period. Use the "
              "resident's preferred name and UK spelling. State only what the data "
              "above supports — do not invent observations, diagnoses or dates. "
              "Write the summary text only, with no heading and no preamble."
        )
    elif report_type == "risk":
        return (
            base + f"Assessment type: {data.get('assessment_type')}. "
            f"Risk level: {data.get('risk_level')}. Score: {data.get('score')}. "
            f"Risk factors: {data.get('risk_factors','')}. "
            f"Interventions: {data.get('interventions','')}. "
            "Write a 3–4 sentence professional risk assessment narrative for a UK care home."
        )
    elif report_type == "care_plan_section":
        # Rich context: pull in medications, incidents, wellbeing, risk history
        meds_str   = data.get("medications_list", "")
        inc_str    = data.get("recent_incidents", "")
        risk_str   = data.get("risk_summary_context", "")
        wb_str     = data.get("wellbeing_summary", "")
        notes_str  = data.get("recent_care_notes", "")
        prev_plan  = data.get("previous_plan_section", "")
        allergies  = data.get("allergies", "")
        rag_context = data.get("rag_context", "")   # RAG-retrieved similar records
        return (
            # RAG context is injected BEFORE the task instruction (Lewis et al., 2020)
            (f"{rag_context}\n\n" if rag_context else "")
            + f"You are writing a personalised care plan section for a UK care home.\n\n"
            f"RESIDENT PROFILE:\n"
            f"Name: {name} | Age: {data.get('age','')} | Gender: {data.get('gender','')}\n"
            f"Primary diagnosis: {data.get('primary_diagnosis','')}\n"
            f"Secondary diagnoses: {data.get('secondary_diagnoses','')}\n"
            f"Allergies: {allergies if allergies else 'None known'}\n"
            f"DNACPR: {data.get('dnacpr_status','')} | Mental capacity: {data.get('mental_capacity','')}\n"
            f"Mobility: {data.get('mobility_level','')} | Continence: {data.get('continence_needs','')}\n"
            f"Diet/texture: {data.get('nutrition_texture','')} | Dietary restrictions: {data.get('dietary_restrictions','')}\n"
            f"Falls risk: {data.get('falls_risk','')} | Pressure sore risk: {data.get('pressure_sore_risk','')}\n\n"
            + (f"ACTIVE MEDICATIONS:\n{meds_str}\n\n" if meds_str else "")
            + (f"RECENT INCIDENTS:\n{inc_str}\n\n" if inc_str else "")
            + (f"RISK ASSESSMENT SUMMARY:\n{risk_str}\n\n" if risk_str else "")
            + (f"RECENT WELLBEING SCORES:\n{wb_str}\n\n" if wb_str else "")
            + (f"RECENT CARE NOTES SUMMARY:\n{notes_str}\n\n" if notes_str else "")
            + (f"PREVIOUS VERSION OF THIS SECTION:\n{prev_plan}\n\n" if prev_plan else "")
            + f"NOW write the '{data.get('section','')}' section of the care plan.\n"
            f"Requirements: 4-6 sentences, person-centred UK care home language, "
            f"include specific cautions relevant to this resident's diagnoses and medications, "
            f"reference any known allergies or risks where relevant."
        )
    elif report_type == "high_risk_alert_email":
        trigger    = data.get("trigger_type", "incident")
        res_name   = data.get("resident_name", name)
        severity   = data.get("severity", "")
        risk_level = data.get("risk_level", "")
        details    = data.get("details", "")
        actions    = data.get("immediate_actions", "")
        reporter   = data.get("reported_by", "")
        home       = data.get("home_name", "Sunrise Care Home")
        return (
            f"Write a professional, concise alert email from a UK care home to the registered manager.\n\n"
            f"Context:\n"
            f"- Resident: {res_name}\n"
            f"- Trigger: {'Major/Critical incident' if 'incident' in trigger else 'Very High / High risk assessment'}\n"
            + (f"- Severity: {severity}\n" if severity else "")
            + (f"- Risk level: {risk_level}\n" if risk_level else "")
            + f"- Details: {details}\n"
            + (f"- Immediate actions taken: {actions}\n" if actions else "")
            + f"- Reported by: {reporter}\n"
            f"- Care home: {home}\n\n"
            f"Write a professional 3-paragraph email:\n"
            f"Para 1: What happened (factual, no speculation)\n"
            f"Para 2: What actions have been taken so far\n"
            f"Para 3: What follow-up / decisions are required from the manager\n"
            f"Sign off as '{reporter}, {home}'. No salutation needed — it will be added separately."
        )
    elif report_type == "family_comm":
        return (
            f"Write a professional, warm family communication letter/email for UK care home staff. "
            f"Resident: {name}. Family contact: {data.get('family_contact','')}. "
            f"Communication type: {data.get('comm_type','')}. "
            f"Subject/reason: {data.get('subject','')}. "
            f"Trigger event: {data.get('trigger_event','')}. "
            f"Staff member: {data.get('staff_member','')}. "
            f"Write a complete, empathetic 3-4 paragraph message including salutation and sign-off."
        )
    return base + json.dumps(data) + ". Summarise this care home record professionally in 3 sentences."


def _template_high_risk_alert_email(data: dict) -> str:
    name    = data.get("resident_name", data.get("preferred_name", "the resident"))
    trigger = data.get("trigger_type", "")
    details = data.get("details", "")
    actions = data.get("immediate_actions", "")
    reporter = data.get("reported_by", "Staff on duty")
    severity = data.get("severity", data.get("risk_level", ""))
    home    = data.get("home_name", "Sunrise Care Home")
    event   = "incident" if "incident" in trigger else "risk assessment"
    return (
        f"This email is to notify you of a {severity} {event} involving {name} at {home}.\n\n"
        f"{details}\n\n"
        + (f"Immediate actions taken: {actions}\n\n" if actions else "")
        + f"Please review this matter and advise on any further actions required. "
        f"All relevant documentation has been recorded in the care management system.\n\n"
        f"Kind regards,\n{reporter}\n{home}"
    )


_TEMPLATE_MAP = {
    "care_note":              _template_care_note,
    "incident":               _template_incident,
    "handover":               _template_handover,
    "wellbeing":              _template_wellbeing,
    "risk":                   _template_risk,
    "care_plan_section":      _template_care_plan_section,
    "family_comm":            _template_family_comm,
    "high_risk_alert_email":  _template_high_risk_alert_email,
}


_PROVIDERS = {"gemini": _call_gemini}

# Retained as a list of one so the resolution code below reads the same as it
# did with three backends, and so a second provider could be reintroduced
# without restructuring anything. The list being length one is the point.
_AUTO_ORDER = ["gemini"]


def get_auto_order() -> list[str]:
    """Backends tried when AI_PROVIDER is "auto". Gemini, or nothing."""
    return list(_AUTO_ORDER)


def get_preferred_provider() -> str:
    """'auto' | 'gemini' | 'template'."""
    val = (os.environ.get("AI_PROVIDER", "") or "auto").strip().lower()
    return val if val in ("auto", "template", *_PROVIDERS) else "auto"


def generate_narrative(report_type: str, data: dict) -> tuple[str, bool]:
    """
    Returns (narrative_text, ai_used).

    Resolution:
      AI_PROVIDER=auto (default) → Gemini, then the offline template
      AI_PROVIDER=template       → the offline template only

    Within Gemini, several keys may be tried in order; llm_client handles that
    and puts a quota-exhausted key into a short cooldown. Key rotation cannot
    change what the model says, only whether it can answer, so it does not
    affect the attribution recorded against the note.

    The template fallback is unconditional: if no key works, or none is
    configured, a deterministic narrative is still produced. Care documentation
    is a legal record, so "the API was down, so no note exists" is never an
    acceptable outcome.
    """
    global _LAST_PROVIDER

    prompt = _build_prompt(report_type, data)
    preferred = get_preferred_provider()

    if preferred == "template":
        order = []
    elif preferred == "auto":
        order = get_auto_order()
    else:
        order = [preferred]

    for name in order:
        text = _PROVIDERS[name](prompt)
        if text:
            _LAST_PROVIDER = name
            return text, True

    fn = _TEMPLATE_MAP.get(report_type)
    text = fn(data) if fn else f"Record created for {data.get('preferred_name','resident')}."
    _LAST_PROVIDER = "template"
    return text, False


def get_last_provider() -> str | None:
    """Which backend produced the most recent narrative ('template' if none)."""
    return _LAST_PROVIDER


def get_ai_status() -> dict:
    """Return which AI backends are configured and which one will be used."""
    try:
        import llm_client
        keys = llm_client.gemini_keys()
    except Exception:
        keys = [k for k in [os.environ.get("GEMINI_API_KEY"),
                            os.environ.get("GOOGLE_API_KEY")] if k]
    configured = {"gemini": bool(keys)}
    preferred = get_preferred_provider()

    # Resolve what an actual call would do right now, so the UI can state it
    # plainly instead of making the user infer it from the priority rules.
    if preferred == "template":
        active = "template"
    elif preferred == "auto":
        active = next((p for p in get_auto_order() if configured[p]), "template")
    else:
        active = preferred if configured.get(preferred) else "template"

    return {
        **configured,
        "fallback":     True,
        "preferred":    preferred,
        "active":       active,
        "last_used":    _LAST_PROVIDER,
        "gemini_model": get_gemini_model(),
        "n_keys":       len(keys),
        "labels": {
            "gemini":   f"Gemini ({get_gemini_model()})",
            "template": "Offline template (no AI)",
        },
    }
