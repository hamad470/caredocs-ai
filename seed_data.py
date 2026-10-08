"""
seed_data.py — generate 12 months of realistic synthetic data for all 5 residents.
Run once:  python seed_data.py

Generates (July 2025 → June 2026):
  • Care notes      — every 2–3 days per resident  (~180 total)
  • Incidents       — 1–3 per month across home     (~28 total)
  • Wellbeing       — monthly per resident           (60 total)
  • Risk assessments— quarterly per resident         (20 total)
  • MAR records     — daily for each medication      (~3 000 total)
  • Medications     — full med list per resident
  • Handovers       — every 2 days per resident      (~90 total)
  • Family comms    — monthly per resident            (60 total)
  • Care plans      — quarterly per resident          (20 total)
"""

import sqlite3, random, datetime, os

DB = os.path.join(os.path.dirname(__file__), "carehome.db")

random.seed(42)          # reproducible

# ── helpers ────────────────────────────────────────────────────────────────

def date_range(start_str, end_str):
    s = datetime.date.fromisoformat(start_str)
    e = datetime.date.fromisoformat(end_str)
    return [s + datetime.timedelta(days=i) for i in range((e - s).days + 1)]

START = "2025-07-01"
END   = "2026-06-30"
DATES = date_range(START, END)

STAFF = [
    ("Sarah Mitchell",  "senior_carer"),
    ("James O'Brien",   "care_worker"),
    ("Priya Sharma",    "care_worker"),
    ("Karen Hughes",    "care_worker"),
    ("David Lee",       "care_worker"),
    ("Ruth Okafor",     "senior_carer"),
]

SHIFTS = ["Morning", "Afternoon", "Night"]

RESIDENTS = {
    "RES001": {
        "name": "Margaret Brown", "preferred": "Maggie",
        "diagnosis": "Alzheimer's Dementia",
        "falls_risk": "High", "pressure_risk": "High",
        "dnacpr": True, "mobility": "Requires 2-person assist",
        "mood_baseline": 5,   # out of 10, lower = more confused/low
        "incidents_per_month": 1.5,
    },
    "RES002": {
        "name": "Albert Singh", "preferred": "Albert",
        "diagnosis": "Parkinson's Disease",
        "falls_risk": "High", "pressure_risk": "Medium",
        "dnacpr": False, "mobility": "Walking frame, supervised",
        "mood_baseline": 6,
        "incidents_per_month": 1.0,
    },
    "RES003": {
        "name": "Dorothy Clarke", "preferred": "Dorothy",
        "diagnosis": "Vascular Dementia",
        "falls_risk": "Very High", "pressure_risk": "High",
        "dnacpr": True, "mobility": "Wheelchair dependent",
        "mood_baseline": 4,
        "incidents_per_month": 2.0,
    },
    "RES004": {
        "name": "George Patel", "preferred": "George",
        "diagnosis": "COPD",
        "falls_risk": "Low", "pressure_risk": "Low",
        "dnacpr": False, "mobility": "Independent with stick",
        "mood_baseline": 7,
        "incidents_per_month": 0.5,
    },
    "RES005": {
        "name": "Ethel Davies", "preferred": "Ethel",
        "diagnosis": "Frailty Syndrome",
        "falls_risk": "Very High", "pressure_risk": "Very High",
        "dnacpr": True, "mobility": "Hoist transfer only",
        "mood_baseline": 5,
        "incidents_per_month": 2.5,
    },
}

MEDICATIONS_DB = {
    "RES001": [
        # NOTE: GP names use the "-Fictional" suffix to prevent association with real practitioners
        ("Donepezil",     "Donepezil HCl",  "10mg",   "Oral", "Once daily at night",          "Alzheimer's dementia",    "Dr A. Kapoor-Fictional", False),
        ("Amlodipine",    "Amlodipine",     "5mg",    "Oral", "Once daily in the morning",    "Hypertension",            "Dr A. Kapoor-Fictional", False),
        ("Lorazepam",     "Lorazepam",      "0.5mg",  "Oral", "PRN — max twice daily",        "Anxiety / agitation",     "Dr A. Kapoor-Fictional", True),
    ],
    "RES002": [
        ("Co-careldopa",  "Levodopa/Carbidopa", "25/100mg tablet", "Oral", "Three times daily — 7am 1pm 7pm", "Parkinson's Disease", "Dr M. Patel-Fictional", False),
        ("Omeprazole",    "Omeprazole",     "20mg",   "Oral", "Once daily — 30 min before breakfast", "Gastro-oesophageal reflux", "Dr M. Patel-Fictional", False),
        ("Rivastigmine",  "Rivastigmine",   "1.5mg",  "Oral", "Twice daily with meals",       "Parkinson's dementia",    "Dr M. Patel-Fictional", False),
    ],
    "RES003": [
        ("Aspirin",       "Aspirin",        "75mg",   "Oral", "Once daily with food",         "Vascular dementia prevention", "Dr J. Clark-Fictional", False),
        ("Atorvastatin",  "Atorvastatin",   "40mg",   "Oral", "Once daily at night",          "Hypercholesterolaemia",   "Dr J. Clark-Fictional", False),
        ("Haloperidol",   "Haloperidol",    "0.5mg",  "Oral", "Once daily at night PRN",      "Behavioural symptoms dementia", "Dr J. Clark-Fictional", True),
        ("Furosemide",    "Furosemide",     "40mg",   "Oral", "Once daily in the morning",    "Fluid retention",         "Dr J. Clark-Fictional", False),
    ],
    "RES004": [
        ("Salbutamol",    "Salbutamol",     "100mcg 2 puffs", "Inhaled", "PRN up to QDS",    "COPD — acute breathlessness", "Dr S. Khan-Fictional", False),
        ("Tiotropium",    "Tiotropium",     "18mcg",  "Inhaled", "Once daily in the morning", "COPD — maintenance",      "Dr S. Khan-Fictional", False),
        ("Prednisolone",  "Prednisolone",   "5mg",    "Oral", "Once daily — during exacerbation", "COPD exacerbation",  "Dr S. Khan-Fictional", False),
        ("Ramipril",      "Ramipril",       "5mg",    "Oral", "Once daily in the morning",    "Hypertension",            "Dr S. Khan-Fictional", False),
    ],
    "RES005": [
        ("Morphine SR",   "Morphine sulfate", "10mg", "Oral", "Twice daily — 8am and 8pm",   "Chronic pain — frailty",  "Dr R. Owens-Fictional", True),
        ("Morphine IR",   "Morphine sulfate", "2.5mg","Oral", "PRN — max 4 hourly",          "Breakthrough pain",       "Dr R. Owens-Fictional", True),
        ("Lactulose",     "Lactulose",      "10ml",   "Oral", "Twice daily",                  "Opioid-induced constipation", "Dr R. Owens-Fictional", False),
        ("Midazolam",     "Midazolam",      "2.5mg",  "SC",   "PRN — max 4 hourly",          "Agitation / distress",    "Dr R. Owens-Fictional", True),
    ],
}

MOOD_OPTIONS   = ["Happy", "Content", "Settled", "Anxious", "Confused", "Agitated", "Low", "Tired", "Bright", "Calm"]
APPETITE_OPTIONS = ["Good", "Fair", "Poor", "Excellent", "Reduced", "Refused meals"]
SLEEP_OPTIONS  = ["Good", "Fair", "Poor", "Disturbed", "Excellent"]
PAIN_OPTIONS   = ["None observed", "Mild", "Moderate", "Grimacing", "Verbal complaint"]
ACTIVITY_OPTIONS = ["Seated activity", "Music therapy", "TV", "Reminiscence", "Physiotherapy", "Occupational therapy", "Bingo", "Art group", "Garden walk", "Family visit", "Hair salon", "Reading", "Religious service", "Chair exercises"]

INCIDENT_TYPES = ["Fall", "Medication error", "Skin concern", "Behaviour", "Medical emergency", "Safeguarding", "Choking", "Property loss"]
SEVERITY_OPTS  = ["Minor", "Moderate", "Major", "Critical"]
LOCATIONS      = ["Bedroom", "Bathroom", "Lounge", "Corridor", "Dining room", "Garden"]

COMM_TYPES     = ["Phone call", "Email", "In-person visit", "Letter", "Video call"]
COMM_SUBJECTS  = ["Monthly welfare update", "Incident notification", "Care plan review", "Health update", "Family visit coordination", "Medication change notification"]

WELLBEING_NOTES = {
    "physical": ["Physically stable this period.", "Some decline in physical health noted.", "Weight stable. Appetite fair.", "COPD exacerbation managed well.", "Skin integrity maintained with repositioning."],
    "mental":   ["Settled and content overall.", "Some episodes of confusion noted.", "Anxiety episodes increasing.", "Positive engagement with reminiscence therapy.", "Good response to music therapy."],
    "social":   ["Engaged well with group activities.", "Prefers one-to-one contact.", "Family visited regularly this period.", "Withdrawn from group activities.", "Participated in bingo and chair exercises."],
}


def weighted_mood(baseline, date):
    """Mood fluctuates seasonally and randomly around a per-resident baseline."""
    seasonal = 1 if date.month in [6, 7, 8] else 0   # better in summer
    val = baseline + seasonal + random.randint(-2, 2)
    return max(1, min(10, val))


def pick_staff():
    s = random.choice(STAFF)
    return s[0], s[1]


def fmt(d):
    return d.isoformat() if isinstance(d, datetime.date) else d


# ── main seed ──────────────────────────────────────────────────────────────

def seed():
    conn = sqlite3.connect(DB)
    c    = conn.cursor()

    # ── 1. medications ──────────────────────────────────────────────────────
    print("Seeding medications...")
    c.execute("DELETE FROM medications")  # wipe all so GP name updates always apply cleanly
    for res_id, meds in MEDICATIONS_DB.items():
        for (name, generic, dose, route, freq, indication, gp, controlled) in meds:
            c.execute("SELECT id FROM medications WHERE resident_id=? AND medication_name=?", (res_id, name))
            if not c.fetchone():
                c.execute("""INSERT INTO medications
                    (resident_id,medication_name,generic_name,dose,route,frequency,indication,
                     prescribing_gp,start_date,review_date,is_controlled,is_prn,prn_instructions,
                     admin_notes,side_effects,status,created)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (res_id, name, generic, dose, route, freq, indication, gp,
                     "2025-07-01", "2026-01-01",
                     1 if controlled else 0,
                     1 if "PRN" in freq else 0,
                     freq if "PRN" in freq else "",
                     "DO NOT crush" if name == "Co-careldopa" else
                     "Administer with food" if name in ("Rivastigmine","Prednisolone","Aspirin") else "",
                     "Monitor for sedation" if controlled else "Monitor BP" if "amlodipine" in name.lower() else "",
                     "active",
                     datetime.datetime.now().isoformat()))
    conn.commit()

    # fetch medication ids for MAR
    c.execute("SELECT id, resident_id, medication_name, is_prn FROM medications WHERE status='active'")
    all_meds = c.fetchall()
    print(f"  Medications: {len(all_meds)} active")

    # ── 2. care notes ───────────────────────────────────────────────────────
    print("Seeding care notes...")
    care_note_count = 0
    c.execute("DELETE FROM care_notes WHERE date >= '2025-07-01'")
    for res_id, rdata in RESIDENTS.items():
        note_dates = [d for d in DATES if d.toordinal() % 3 == hash(res_id) % 3]  # every ~3 days
        for d in note_dates:
            shift = random.choice(SHIFTS)
            staff_name, staff_role = pick_staff()
            mood_score = weighted_mood(rdata["mood_baseline"], d)
            mood_label = MOOD_OPTIONS[min(len(MOOD_OPTIONS)-1, max(0, 10 - mood_score))]
            appetite   = random.choices(APPETITE_OPTIONS, weights=[30,25,15,10,15,5])[0]
            fluid      = random.randint(800, 1800)
            weight     = round(random.uniform(48 if res_id=="RES005" else 55, 85), 1)
            skin_ok    = random.random() > 0.15
            pain       = random.choices(PAIN_OPTIONS, weights=[60,20,10,5,5])[0]
            falls      = 1 if random.random() < rdata["incidents_per_month"] / 30 else 0
            activity   = random.choice(ACTIVITY_OPTIONS)
            narratives = [
                f"{rdata['preferred']} was {mood_label.lower()} during the {shift.lower()} shift. "
                f"Appetite was {appetite.lower()}; fluid intake recorded at {fluid}ml. "
                f"{'No concerns noted.' if skin_ok else 'Red area noted on sacrum — repositioning increased.'} "
                f"{'No pain reported.' if pain=='None observed' else f'Pain noted: {pain.lower()} — analgesia reviewed.'}",

                f"Supported {rdata['preferred']} with personal care this {shift.lower()} shift. "
                f"Mood observed as {mood_label.lower()}. {activity} enjoyed. "
                f"Fluid intake {fluid}ml. {'No incidents this shift.' if not falls else 'Fall noted — incident report completed.'}",
            ]
            c.execute("""INSERT INTO care_notes
                (resident_id,date,shift,staff_name,staff_role,note_type,personal_care,
                 mood,appetite,fluid_intake_ml,weight_kg,skin_checked,skin_concern,
                 repositioned,activity,activity_description,sleep_quality,pain_observed,
                 pain_location,falls_this_shift,care_narrative,concerns,actions_taken,
                 handover_notes,ai_generated,status,created)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                # 27 values
                (res_id, fmt(d), shift, staff_name, staff_role, "Routine",
                 "Assisted with all personal care" if rdata["mobility"] != "Independent with stick" else "Independent, supervised",
                 mood_label, appetite, fluid, weight,
                 "Yes", "" if skin_ok else "Red area noted — pressure relief increased",
                 "Yes" if rdata["pressure_risk"] in ("High","Very High") else "No",
                 activity, f"Participated in {activity.lower()}.",
                 random.choice(SLEEP_OPTIONS), pain,
                 "Sacrum" if not skin_ok else "",
                 falls, random.choice(narratives),
                 "" if skin_ok else "Monitor skin integrity.",
                 "Repositioning chart updated." if not skin_ok else "Continue monitoring.",
                 "No urgent concerns." if not falls else "Fall occurred — see incident report.",
                 1, "approved",
                 datetime.datetime.combine(d, datetime.time(8,0)).isoformat()))
            care_note_count += 1
    conn.commit()
    print(f"  Care notes: {care_note_count}")

    # ── 3. incidents ─────────────────────────────────────────────────────────
    print("Seeding incidents...")
    c.execute("DELETE FROM incidents WHERE date >= '2025-07-01'")
    incident_count = 0
    inc_types_by_res = {
        "RES001": ["Fall","Behaviour","Skin concern"],
        "RES002": ["Fall","Medication error","Skin concern"],
        "RES003": ["Fall","Behaviour","Medical emergency"],
        "RES004": ["Medical emergency","Medication error","Property loss"],
        "RES005": ["Fall","Medical emergency","Skin concern"],
    }
    for res_id, rdata in RESIDENTS.items():
        # probability per month
        num_months = 12
        for month_offset in range(num_months):
            month_start = datetime.date(2025, 7, 1) + datetime.timedelta(days=30 * month_offset)
            n_incidents = int(rdata["incidents_per_month"] * random.uniform(0.5, 1.5))
            for _ in range(n_incidents):
                inc_day  = month_start + datetime.timedelta(days=random.randint(0, 28))
                if inc_day > datetime.date(2026, 6, 30):
                    continue
                itype    = random.choice(inc_types_by_res[res_id])
                severity = random.choices(SEVERITY_OPTS, weights=[50,30,15,5])[0]
                staff_n, _ = pick_staff()
                inc_id   = f"INC-{inc_day.strftime('%Y%m')}-{incident_count+1:03d}"
                c.execute("""INSERT INTO incidents
                    (incident_id,resident_id,date,time,shift,incident_type,severity,location,
                     witnessed,witness_name,staff_first_on_scene,description,immediate_actions,
                     injuries,medical_attention,outcome,gp_notified,family_notified,
                     cqc_notification,risk_assessment_updated,care_plan_updated,
                     investigation_required,lessons_learned,preventative_actions,
                     manager_sign_off,status,ai_generated,created)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (inc_id, res_id, fmt(inc_day),
                     f"{random.randint(7,21):02d}:{random.choice(['00','15','30','45'])}",
                     random.choice(SHIFTS), itype, severity,
                     random.choice(LOCATIONS),
                     "Yes", staff_n, staff_n,
                     f"{itype} involving {rdata['preferred']}. Staff responded immediately.",
                     "Observed and assessed. Senior notified.",
                     "None" if severity in ("Minor","Moderate") else "Minor bruising",
                     "No" if severity == "Minor" else "GP informed",
                     "Monitored. No lasting harm." if severity in ("Minor","Moderate") else "GP review completed.",
                     "Yes" if severity in ("Major","Critical") else "No",
                     "Yes" if severity in ("Major","Critical") else "No",
                     "Yes" if severity == "Critical" else "No",
                     "Yes", "Yes" if severity in ("Major","Critical") else "No",
                     "Yes" if severity in ("Major","Critical") else "No",
                     f"Staff to {('increase supervision' if itype=='Fall' else 'review medication timing' if itype=='Medication error' else 'monitor closely')}.",
                     f"Additional {'fall prevention' if itype=='Fall' else 'clinical'} review scheduled.",
                     "Ruth Okafor",
                     "closed", 0,
                     datetime.datetime.combine(inc_day, datetime.time(10,0)).isoformat()))
                incident_count += 1
    conn.commit()
    print(f"  Incidents: {incident_count}")

    # ── 4. wellbeing assessments ─────────────────────────────────────────────
    print("Seeding wellbeing assessments...")
    c.execute("DELETE FROM wellbeing WHERE assessment_date >= '2025-07-01'")
    wb_count = 0
    for res_id, rdata in RESIDENTS.items():
        for month_offset in range(12):
            assess_date = datetime.date(2025, 7, 1) + datetime.timedelta(days=30 * month_offset + 14)
            if assess_date > datetime.date(2026, 6, 30):
                continue
            staff_n, _ = pick_staff()
            mood_score = weighted_mood(rdata["mood_baseline"], assess_date)
            base = rdata["mood_baseline"]
            phys  = max(1, min(10, base + random.randint(-1,1)))
            ment  = max(1, min(10, mood_score))
            social= max(1, min(10, base - 1 + random.randint(-1,2)))
            pc    = max(1, min(10, base + random.randint(-2,1)))
            nutr  = max(1, min(10, base + random.randint(-1,2)))
            pain  = max(1, min(10, base + random.randint(-2,2)))
            overall = round((phys+ment+social+pc+nutr+pain)/6, 1)
            c.execute("""INSERT INTO wellbeing
                (resident_id,assessment_date,assessed_by,period_covered,
                 physical_health_score,mental_health_score,social_engagement_score,
                 personal_care_score,nutrition_score,pain_management_score,overall_score,
                 physical_notes,mental_notes,social_notes,goals_progress,concerns,
                 positive_outcomes,actions_next_period,family_feedback,resident_voice,
                 summary,ai_generated,status,created)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (res_id, fmt(assess_date), staff_n,
                 assess_date.strftime("%B %Y"),
                 phys, ment, social, pc, nutr, pain, int(overall*10)/10,
                 random.choice(WELLBEING_NOTES["physical"]),
                 random.choice(WELLBEING_NOTES["mental"]),
                 random.choice(WELLBEING_NOTES["social"]),
                 "Goals on track." if overall >= 5 else "Goals require review.",
                 "" if phys >= 5 else "Physical health declining — GP review requested.",
                 "Positive engagement with care team." if overall >= 6 else "Some improvement in personal care.",
                 "Continue current care plan." if overall >= 6 else "Review care plan next month.",
                 "Family report satisfaction with care.",
                 f"{rdata['preferred']} indicates feeling {'settled' if ment>=6 else 'unsettled'} this period.",
                 f"Overall wellbeing score {overall}/10 for {assess_date.strftime('%B %Y')}.",
                 0, "approved",
                 datetime.datetime.combine(assess_date, datetime.time(10,0)).isoformat()))
            wb_count += 1
    conn.commit()
    print(f"  Wellbeing: {wb_count}")

    # ── 5. risk assessments ──────────────────────────────────────────────────
    print("Seeding risk assessments...")
    c.execute("DELETE FROM risk_assessments WHERE date_assessed >= '2025-07-01'")
    ra_count = 0
    RISK_TYPES = ["Falls", "Pressure Sore", "Nutrition", "Moving and Handling", "Medication"]
    risk_scores_base = {
        "RES001": {"Falls":16,"Pressure Sore":17,"Nutrition":14,"Moving and Handling":12,"Medication":10},
        "RES002": {"Falls":15,"Pressure Sore":12,"Nutrition":12,"Moving and Handling":13,"Medication":14},
        "RES003": {"Falls":19,"Pressure Sore":17,"Nutrition":15,"Moving and Handling":15,"Medication":12},
        "RES004": {"Falls":8, "Pressure Sore":6, "Nutrition":10,"Moving and Handling":7, "Medication":9},
        "RES005": {"Falls":19,"Pressure Sore":20,"Nutrition":16,"Moving and Handling":16,"Medication":15},
    }
    risk_level = lambda s: "Very High" if s>=18 else "High" if s>=14 else "Medium" if s>=10 else "Low"
    for res_id in RESIDENTS:
        for quarter in range(4):
            q_date = datetime.date(2025, 7, 1) + datetime.timedelta(days=90*quarter+7)
            if q_date > datetime.date(2026, 6, 30): continue
            staff_n, _ = pick_staff()
            for rtype in RISK_TYPES:
                base_score = risk_scores_base[res_id][rtype]
                score = max(1, min(20, base_score + random.randint(-2, 2)))
                rl    = risk_level(score)
                c.execute("""INSERT INTO risk_assessments
                    (resident_id,assessment_type,date_assessed,assessed_by,review_date,
                     score,risk_level,risk_factors,interventions,additional_actions,
                     outcome_measures,narrative,ai_generated,status,created)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (res_id, rtype, fmt(q_date), staff_n,
                     fmt(q_date + datetime.timedelta(days=90)),
                     score, rl,
                     f"Known history of {rtype.lower()} risk. Current score {score}/20.",
                     f"{'Two-hourly repositioning, pressure mattress.' if rtype=='Pressure Sore' else 'Non-slip footwear, call bell within reach.' if rtype=='Falls' else 'Regular monitoring and review.'}",
                     "Review at next care plan meeting.",
                     f"{rtype} risk to remain {'reduced' if rl in ('Low','Medium') else 'closely monitored'}.",
                     f"{RESIDENTS[res_id]['preferred']} assessed as {rl.lower()} risk for {rtype.lower()} on {q_date}.",
                     0, "active",
                     datetime.datetime.combine(q_date, datetime.time(9,0)).isoformat()))
                ra_count += 1
    conn.commit()
    print(f"  Risk assessments: {ra_count}")

    # ── 6. MAR records ───────────────────────────────────────────────────────
    print("Seeding MAR records (this may take a moment)...")
    c.execute("DELETE FROM mar_records WHERE date >= '2025-07-01'")
    mar_count = 0
    c.execute("SELECT id, resident_id, frequency, is_prn FROM medications WHERE status='active'")
    meds_list = c.fetchall()
    # sample 1 in 3 days for performance; PRN less often
    sample_dates = [d for d in DATES if d.toordinal() % 2 == 0]  # every other day
    for med_id, res_id, freq, is_prn in meds_list:
        for d in sample_dates:
            if is_prn:
                if random.random() > 0.25:  # PRN used 25% of days
                    continue
            staff_n, _ = pick_staff()
            admin = "Yes" if random.random() > 0.05 else "No"
            refusal = "" if admin == "Yes" else random.choice(["Resident refused", "Asleep", "Vomiting"])
            doses_per_day = 3 if "Three" in freq else 2 if "Twice" in freq else 1
            for dose_n in range(doses_per_day):
                hour = [8, 13, 18][dose_n] if doses_per_day == 3 else [8, 20][dose_n] if doses_per_day == 2 else 8
                c.execute("""INSERT INTO mar_records
                    (medication_id,resident_id,date,time_given,shift,given_by,
                     administered,refusal_reason,notes,created)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (med_id, res_id, fmt(d),
                     f"{hour:02d}:{random.randint(0,59):02d}",
                     "Morning" if hour < 12 else "Afternoon" if hour < 17 else "Night",
                     staff_n, admin, refusal,
                     "" if admin == "Yes" else f"Not administered — {refusal}",
                     datetime.datetime.combine(d, datetime.time(hour,0)).isoformat()))
                mar_count += 1
    conn.commit()
    print(f"  MAR records: {mar_count}")

    # ── 7. handovers ─────────────────────────────────────────────────────────
    print("Seeding handovers...")
    c.execute("DELETE FROM handovers WHERE date >= '2025-07-01'")
    ho_count = 0
    ho_dates = [d for d in DATES if d.toordinal() % 4 == 0]
    for d in ho_dates:
        for res_id, rdata in RESIDENTS.items():
            staff_n, _ = pick_staff()
            shift_end   = random.choice(["Morning","Afternoon"])
            shift_start = "Afternoon" if shift_end=="Morning" else "Night"
            c.execute("""INSERT INTO handovers
                (date,shift_ending,shift_starting,compiled_by,resident_id,
                 overall_summary,care_completed,concerns_next_shift,outstanding_tasks,
                 medication_notes,fluid_target_met,incidents_this_shift,incident_reference,
                 family_contact,family_contact_notes,escalation_required,
                 ai_generated,status,created)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (fmt(d), shift_end, shift_start, staff_n, res_id,
                 f"{rdata['preferred']} had a {'settled' if random.random()>0.3 else 'unsettled'} shift. No significant concerns.",
                 "Personal care completed. Meals offered. Repositioning done.",
                 "Continue monitoring." if random.random()>0.2 else "Monitor skin integrity closely.",
                 "Fluid chart to be updated.",
                 "All medications administered as prescribed.",
                 "Yes" if random.random()>0.2 else "No",
                 0, "",
                 "Yes" if random.random()>0.6 else "No",
                 "Family updated by phone." if random.random()>0.6 else "",
                 "No", 0, "approved",
                 datetime.datetime.combine(d, datetime.time(14,30)).isoformat()))
            ho_count += 1
    conn.commit()
    print(f"  Handovers: {ho_count}")

    # ── 8. family comms ───────────────────────────────────────────────────────
    print("Seeding family communications...")
    c.execute("DELETE FROM family_comms WHERE date >= '2025-07-01'")
    fc_count = 0
    for res_id, rdata in RESIDENTS.items():
        for month_offset in range(12):
            comm_date = datetime.date(2025, 7, 1) + datetime.timedelta(days=30*month_offset + random.randint(5,25))
            if comm_date > datetime.date(2026, 6, 30): continue
            staff_n, _ = pick_staff()
            subject = random.choice(COMM_SUBJECTS)
            c.execute("""INSERT INTO family_comms
                (resident_id,date,comm_type,direction,staff_member,family_contact,
                 subject,trigger_event,body,family_response,follow_up,
                 follow_up_actions,ai_drafted,approved_by,created)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (res_id, fmt(comm_date), random.choice(COMM_TYPES), "Outgoing",
                 staff_n, "Next of kin",
                 subject, "Routine monthly contact",
                 f"Dear family of {rdata['preferred']}, we are writing to update you on {rdata['preferred']}'s care. "
                 f"Overall {rdata['preferred']} has been {('settled and well' if rdata['mood_baseline']>=6 else 'receiving close monitoring')} this month. "
                 f"Please do not hesitate to contact us if you have any questions.",
                 "Family acknowledged and expressed satisfaction with care.",
                 "No follow-up required.", "",
                 0, "Care Manager",
                 datetime.datetime.combine(comm_date, datetime.time(10,0)).isoformat()))
            fc_count += 1
    conn.commit()
    print(f"  Family comms: {fc_count}")

    # ── 9. care plans ─────────────────────────────────────────────────────────
    print("Seeding care plans...")
    c.execute("DELETE FROM care_plans WHERE effective_from >= '2025-07-01'")
    cp_count = 0
    for res_id, rdata in RESIDENTS.items():
        for quarter in range(4):
            eff_date = datetime.date(2025, 7, 1) + datetime.timedelta(days=90*quarter)
            rev_date = eff_date + datetime.timedelta(days=90)
            if eff_date > datetime.date(2026, 6, 30): continue
            c.execute("""INSERT INTO care_plans
                (resident_id,version,effective_from,review_date,reviewed_by,status,
                 personal_identity_summary,mobility_care_plan,personal_care_plan,
                 continence_care_plan,nutrition_hydration_plan,medication_management_plan,
                 cognitive_support_plan,emotional_wellbeing_plan,social_activity_plan,
                 end_of_life_preferences,risk_summary,goals_of_care,family_involvement_plan,
                 ai_generated,created,generation_method)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (res_id, quarter+1, fmt(eff_date), fmt(rev_date), "Care Manager",
                 "approved",
                 f"{rdata['preferred']} is a {rdata['diagnosis']} resident who values dignity and person-centred care.",
                 f"Mobility support: {rdata['mobility']}. Falls risk assessed as {rdata['falls_risk']}.",
                 "Supported with personal care as required, respecting dignity and preferences.",
                 "Continence care provided with dignity. Continence aids used as required.",
                 "Ensure adequate fluid intake (min 1500ml). Texture-modified diet as prescribed.",
                 "All medications administered as prescribed. Controlled drugs double-checked.",
                 f"{'Cognitive stimulation activities daily.' if 'Dementia' in rdata['diagnosis'] else 'Cognitive abilities monitored.'}",
                 f"Emotional wellbeing supported. {'DNACPR in place and discussed with family.' if rdata['dnacpr'] else 'Resident engaged in decision-making.'}",
                 "Social activities offered daily. Participation encouraged but not forced.",
                 f"{'DNACPR in place. Comfort measures prioritised.' if rdata['dnacpr'] else 'Full treatment to be pursued if required.'}",
                 f"Falls risk {rdata['falls_risk']}. Pressure sore risk {rdata['pressure_risk']}.",
                 "Maintain dignity, comfort and quality of life.",
                 "Family to be kept informed. Monthly welfare calls arranged.",
                 0, datetime.datetime.combine(eff_date, datetime.time(10,0)).isoformat(),
                 "manual"))
            cp_count += 1
    conn.commit()
    print(f"  Care plans: {cp_count}")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n✓ Seed complete. Database row counts:")
    for t in ["medications","care_notes","incidents","wellbeing","risk_assessments","mar_records","handovers","family_comms","care_plans"]:
        c.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t:25s}: {c.fetchone()[0]}")
    conn.close()


if __name__ == "__main__":
    seed()
