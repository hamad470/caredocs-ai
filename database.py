"""
Database initialisation and seed data for CareDocs AI.
SQLite, no external DB server needed.
"""
import sqlite3
import hashlib
import os
from datetime import date, datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "carehome.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def init_db():
    conn = get_db()
    c = conn.cursor()

    # ── Users & Roles ──────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT UNIQUE NOT NULL,
        password  TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role      TEXT NOT NULL,  -- manager | senior_carer | care_worker | readonly
        email     TEXT,
        active    INTEGER DEFAULT 1,
        created   TEXT DEFAULT (datetime('now'))
    )""")
    # Add email column to existing DBs (migration-safe)
    try:
        c.execute("ALTER TABLE users ADD COLUMN email TEXT")
    except Exception:
        pass

    # ── Residents ──────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS residents (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id         TEXT UNIQUE NOT NULL,
        full_name           TEXT NOT NULL,
        preferred_name      TEXT,
        date_of_birth       TEXT,
        age                 INTEGER,
        gender              TEXT,
        room_number         TEXT,
        admission_date      TEXT,
        care_type           TEXT,
        primary_diagnosis   TEXT,
        secondary_diagnoses TEXT,
        allergies           TEXT,
        medications_summary TEXT,
        gp_name             TEXT,
        gp_phone            TEXT,
        dnacpr_status       TEXT,
        mental_capacity     TEXT,
        mobility_level      TEXT,
        continence_needs    TEXT,
        nutrition_texture   TEXT,
        dietary_restrictions TEXT,
        falls_risk          TEXT,
        pressure_sore_risk  TEXT,
        nok_name            TEXT,
        nok_relationship    TEXT,
        nok_phone           TEXT,
        nok_email           TEXT,
        key_worker          TEXT,
        active              INTEGER DEFAULT 1
    )""")

    # ── Care Notes ─────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS care_notes (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id           TEXT NOT NULL,
        date                  TEXT NOT NULL,
        shift                 TEXT NOT NULL,
        staff_name            TEXT NOT NULL,
        staff_role            TEXT NOT NULL,
        note_type             TEXT NOT NULL,
        personal_care         TEXT,
        mood                  TEXT,
        appetite              TEXT,
        fluid_intake_ml       INTEGER,
        weight_kg             REAL,
        skin_checked          TEXT,
        skin_concern          TEXT,
        repositioned          TEXT,
        activity              TEXT,
        activity_description  TEXT,
        sleep_quality         TEXT,
        pain_observed         TEXT,
        pain_location         TEXT,
        falls_this_shift      INTEGER DEFAULT 0,
        care_narrative        TEXT,
        concerns              TEXT,
        actions_taken         TEXT,
        handover_notes        TEXT,
        ai_generated          INTEGER DEFAULT 0,
        status                TEXT DEFAULT 'draft',  -- draft | pending_approval | approved
        approved_by           TEXT,
        approved_at           TEXT,
        created               TEXT DEFAULT (datetime('now'))
    )""")

    # ── Incidents ──────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS incidents (
        id                       INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id              TEXT UNIQUE,
        resident_id              TEXT NOT NULL,
        date                     TEXT NOT NULL,
        time                     TEXT,
        shift                    TEXT,
        incident_type            TEXT NOT NULL,
        severity                 TEXT NOT NULL,
        location                 TEXT,
        witnessed                TEXT,
        witness_name             TEXT,
        staff_first_on_scene     TEXT NOT NULL,
        description              TEXT NOT NULL,
        immediate_actions        TEXT,
        injuries                 TEXT,
        medical_attention        TEXT,
        outcome                  TEXT,
        gp_notified              TEXT,
        family_notified          TEXT,
        cqc_notification         TEXT,
        risk_assessment_updated  TEXT,
        care_plan_updated        TEXT,
        investigation_required   TEXT,
        investigation_summary    TEXT,
        lessons_learned          TEXT,
        preventative_actions     TEXT,
        manager_sign_off         TEXT,
        manager_sign_off_date    TEXT,
        status                   TEXT DEFAULT 'open',
        ai_generated             INTEGER DEFAULT 0,
        created                  TEXT DEFAULT (datetime('now'))
    )""")

    # ── Handovers ──────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS handovers (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        date                  TEXT NOT NULL,
        shift_ending          TEXT NOT NULL,
        shift_starting        TEXT NOT NULL,
        compiled_by           TEXT NOT NULL,
        resident_id           TEXT NOT NULL,
        overall_summary       TEXT,
        care_completed        TEXT,
        concerns_next_shift   TEXT,
        outstanding_tasks     TEXT,
        medication_notes      TEXT,
        fluid_target_met      TEXT,
        incidents_this_shift  INTEGER DEFAULT 0,
        incident_reference    TEXT,
        family_contact        TEXT,
        family_contact_notes  TEXT,
        escalation_required   TEXT,
        escalation_details    TEXT,
        ai_generated          INTEGER DEFAULT 0,
        status                TEXT DEFAULT 'draft',
        created               TEXT DEFAULT (datetime('now'))
    )""")

    # ── Medications ────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS medications (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id         TEXT NOT NULL,
        medication_name     TEXT NOT NULL,
        generic_name        TEXT,
        dose                TEXT NOT NULL,
        route               TEXT NOT NULL,
        frequency           TEXT NOT NULL,
        indication          TEXT,
        prescribing_gp      TEXT,
        start_date          TEXT,
        review_date         TEXT,
        is_controlled       INTEGER DEFAULT 0,
        is_prn              INTEGER DEFAULT 0,
        prn_instructions    TEXT,
        admin_notes         TEXT,
        side_effects        TEXT,
        status              TEXT DEFAULT 'active',
        stopped_reason      TEXT,
        created             TEXT DEFAULT (datetime('now'))
    )""")

    # ── Medication Administration Records (MAR) ─────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS mar_records (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        medication_id   INTEGER NOT NULL,
        resident_id     TEXT NOT NULL,
        date            TEXT NOT NULL,
        time_given      TEXT,
        shift           TEXT,
        given_by        TEXT NOT NULL,
        administered    TEXT NOT NULL,  -- yes | no | refused | not_available
        refusal_reason  TEXT,
        notes           TEXT,
        created         TEXT DEFAULT (datetime('now'))
    )""")

    # ── Wellbeing Assessments ──────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS wellbeing (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id             TEXT NOT NULL,
        assessment_date         TEXT NOT NULL,
        assessed_by             TEXT NOT NULL,
        period_covered          TEXT,
        physical_health_score   INTEGER,
        mental_health_score     INTEGER,
        social_engagement_score INTEGER,
        personal_care_score     INTEGER,
        nutrition_score         INTEGER,
        pain_management_score   INTEGER,
        overall_score           INTEGER,
        physical_notes          TEXT,
        mental_notes            TEXT,
        social_notes            TEXT,
        goals_progress          TEXT,
        concerns                TEXT,
        positive_outcomes       TEXT,
        actions_next_period     TEXT,
        family_feedback         TEXT,
        resident_voice          TEXT,
        summary                 TEXT,
        ai_generated            INTEGER DEFAULT 0,
        status                  TEXT DEFAULT 'draft',
        created                 TEXT DEFAULT (datetime('now'))
    )""")

    # ── Risk Assessments ───────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS risk_assessments (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id         TEXT NOT NULL,
        assessment_type     TEXT NOT NULL,  -- falls | pressure | nutrition | behaviour | environment
        date_assessed       TEXT NOT NULL,
        assessed_by         TEXT NOT NULL,
        review_date         TEXT,
        score               INTEGER,
        risk_level          TEXT,           -- low | medium | high | very_high
        risk_factors        TEXT,
        interventions       TEXT,
        additional_actions  TEXT,
        outcome_measures    TEXT,
        narrative           TEXT,
        ai_generated        INTEGER DEFAULT 0,
        status              TEXT DEFAULT 'draft',
        created             TEXT DEFAULT (datetime('now'))
    )""")

    # ── Family Communications ──────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS family_comms (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id      TEXT NOT NULL,
        date             TEXT NOT NULL,
        comm_type        TEXT NOT NULL,   -- phone_call | visit | email | letter | meeting
        direction        TEXT NOT NULL,   -- inbound | outbound
        staff_member     TEXT NOT NULL,
        family_contact   TEXT NOT NULL,
        subject          TEXT,
        trigger_event    TEXT,
        body             TEXT NOT NULL,
        family_response  TEXT,
        follow_up        TEXT,
        follow_up_actions TEXT,
        ai_drafted       INTEGER DEFAULT 0,
        approved_by      TEXT,
        created          TEXT DEFAULT (datetime('now'))
    )""")

    # ── Care Plans ────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS care_plans (
        id                          INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id                 TEXT NOT NULL,
        version                     INTEGER DEFAULT 1,
        effective_from              TEXT,
        review_date                 TEXT,
        reviewed_by                 TEXT,
        status                      TEXT DEFAULT 'Active',
        personal_identity_summary   TEXT,
        mobility_care_plan          TEXT,
        personal_care_plan          TEXT,
        continence_care_plan        TEXT,
        nutrition_hydration_plan    TEXT,
        medication_management_plan  TEXT,
        cognitive_support_plan      TEXT,
        emotional_wellbeing_plan    TEXT,
        social_activity_plan        TEXT,
        end_of_life_preferences     TEXT,
        risk_summary                TEXT,
        goals_of_care               TEXT,
        family_involvement_plan     TEXT,
        ai_generated                INTEGER DEFAULT 0,
        created                     TEXT DEFAULT (datetime('now'))
    )""")

    # ── Family Communications ─────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS family_comms (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id      TEXT NOT NULL,
        date             TEXT NOT NULL,
        comm_type        TEXT NOT NULL,
        direction        TEXT NOT NULL,
        staff_member     TEXT NOT NULL,
        family_contact   TEXT NOT NULL,
        subject          TEXT,
        trigger_event    TEXT,
        body             TEXT NOT NULL,
        family_response  TEXT,
        follow_up        TEXT DEFAULT 'No',
        follow_up_actions TEXT,
        ai_drafted       INTEGER DEFAULT 0,
        approved_by      TEXT,
        created          TEXT DEFAULT (datetime('now'))
    )""")

    # ── Email Queue ────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS email_queue (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id     TEXT NOT NULL,
        trigger_type    TEXT NOT NULL,   -- incident_high | risk_very_high | risk_high
        trigger_ref     TEXT,            -- incident_id or risk assessment id
        to_email        TEXT NOT NULL,
        to_name         TEXT,
        subject         TEXT NOT NULL,
        body            TEXT NOT NULL,
        generated_by    TEXT NOT NULL,   -- staff who triggered it
        status          TEXT DEFAULT 'pending_approval',  -- pending_approval | approved | sent | rejected
        approved_by     TEXT,
        approved_at     TEXT,
        sent_at         TEXT,
        ai_drafted      INTEGER DEFAULT 1,
        created         TEXT DEFAULT (datetime('now'))
    )""")

    # ── RAG Audit Log ─────────────────────────────────────────────────────────
    # GDPR compliance: every retrieval event is logged (3-year minimum retention)
    c.execute("""CREATE TABLE IF NOT EXISTS rag_audit_log (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        resident_id           TEXT NOT NULL,
        query_summary         TEXT,
        section               TEXT,
        num_chunks_retrieved  INTEGER DEFAULT 0,
        mode                  TEXT,              -- tfidf | semantic
        chunk_ids_json        TEXT,              -- JSON array of chunk IDs used
        generated_by          TEXT,              -- staff username
        retrieved_at          TEXT DEFAULT (datetime('now'))
    )""")

    # Add RAG tracking columns to care_plans (migration-safe)
    for col_sql in [
        "ALTER TABLE care_plans ADD COLUMN generation_method TEXT DEFAULT 'direct'",
        "ALTER TABLE care_plans ADD COLUMN rag_chunks_used TEXT",
        "ALTER TABLE care_plans ADD COLUMN rag_mode TEXT",
    ]:
        try:
            c.execute(col_sql)
        except Exception:
            pass  # Column already exists

    # ── Audit Log ──────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER,
        username    TEXT,
        action      TEXT NOT NULL,
        table_name  TEXT,
        record_id   INTEGER,
        details     TEXT,
        ip_address  TEXT,
        created     TEXT DEFAULT (datetime('now'))
    )""")

    conn.commit()
    _seed_demo_data(conn)
    conn.close()
    print("✅  Database initialised.")


def _seed_demo_data(conn):
    c = conn.cursor()

    # Check if already seeded
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] > 0:
        return

    # ── Demo Users ─────────────────────────────────────────────────────────────
    users = [
        ("manager1",  hash_pw("manager123"),  "Sarah Johnson",  "manager",      "manager@sunrisecare.example.com"),
        ("senior1",   hash_pw("senior123"),   "David Patel",    "senior_carer", "senior1@sunrisecare.example.com"),
        ("carer1",    hash_pw("carer123"),    "Emma Williams",  "care_worker",  "carer1@sunrisecare.example.com"),
        ("carer2",    hash_pw("carer123"),    "James Thompson", "care_worker",  "carer2@sunrisecare.example.com"),
        ("readonly1", hash_pw("readonly123"), "Dr. A. Khan",    "readonly",     "gp@practice.example.com"),
    ]
    c.executemany(
        "INSERT INTO users (username,password,full_name,role,email) VALUES (?,?,?,?,?)",
        users
    )

    # ── Demo Residents ──────────────────────────────────────────────────────────
    residents = [
        ("RES001","Margaret Brown","Maggie","1939-03-15",85,"Female","Room 1","2022-01-10",
         "Residential","Dementia (Alzheimer's type)","Hypertension, Osteoporosis",
         "Penicillin","Donepezil 10mg, Amlodipine 5mg","Dr. A. Khan","01234 567890",
         "DNACPR in place","Lacks capacity","Limited - uses walking frame","Continent with prompting",
         "Soft & bite-sized","None","High","High","Jean Brown","Daughter","07700 900001",
         "jean.brown@email.com","Emma Williams",1),
        ("RES002","Albert Singh","Albert","1944-07-22",80,"Male","Room 2","2021-06-01",
         "Nursing","Parkinson's Disease","Type 2 Diabetes, Depression",
         "Sulfonamides","Co-careldopa, Metformin, Sertraline","Dr. A. Khan","01234 567890",
         "Full resuscitation","Has capacity","Needs assistance - Zimmer frame","Continent",
         "Normal - soft","Diabetic","High","Medium","Priya Singh","Wife","07700 900002",
         "priya.singh@email.com","David Patel",1),
        ("RES003","Dorothy Clarke","Dot","1935-11-30",88,"Female","Room 3","2023-03-20",
         "Residential","Vascular Dementia","Heart Failure, Arthritis",
         "None known","Furosemide, Aspirin, Tramadol","Dr. B. Osei","01234 567891",
         "DNACPR in place","Lacks capacity","Wheelchair dependent","Incontinent - pads",
         "Pureed","Low sodium","Very High","High","Michael Clarke","Son","07700 900003",
         "m.clarke@email.com","James Thompson",1),
        ("RES004","George Patel","George","1948-05-10",76,"Male","Room 4","2023-09-15",
         "Nursing","COPD","Anxiety, Hypertension",
         "Aspirin","Salbutamol inhaler, Tiotropium, Ramipril","Dr. B. Osei","01234 567891",
         "Full resuscitation","Has capacity","Independent with stick","Continent",
         "Normal","None","Low","Low","Anita Patel","Wife","07700 900004",
         "anita.patel@email.com","Emma Williams",1),
        ("RES005","Ethel Davies","Ethel","1932-08-25",92,"Female","Room 5","2020-11-05",
         "Residential","Frailty syndrome","Osteoporosis, Depression, Hearing Loss",
         "Latex","Calcium/Vit D, Sertraline","Dr. A. Khan","01234 567890",
         "DNACPR in place","Has capacity","Bed/chair bound","Incontinent - catheter",
         "Pureed + thickened fluids","Low sugar","Very High","Very High","Susan Davies",
         "Daughter","07700 900005","susan.davies@email.com","David Patel",1),
    ]
    c.executemany("""INSERT INTO residents
        (resident_id,full_name,preferred_name,date_of_birth,age,gender,room_number,
         admission_date,care_type,primary_diagnosis,secondary_diagnoses,allergies,
         medications_summary,gp_name,gp_phone,dnacpr_status,mental_capacity,
         mobility_level,continence_needs,nutrition_texture,dietary_restrictions,
         falls_risk,pressure_sore_risk,nok_name,nok_relationship,nok_phone,
         nok_email,key_worker,active)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        residents
    )

    # ── Demo Care Notes ────────────────────────────────────────────────────────
    care_notes = [
        ("RES001","2024-01-15","Morning","Emma Williams","Care Worker","Daily Care",
         "Completed","Happy and engaged","Good","450",70.2,"Yes","None","Yes",
         "Yes","Participated in morning exercise","Good","No","","0",
         "Maggie had a positive morning. Personal care was completed with minimal assistance. She enjoyed the morning exercise group and was in good spirits throughout the shift.",
         "None","All care completed as planned","Continues to be settled. No concerns for handover.",
         1,"approved","David Patel","2024-01-15 14:00:00"),
        ("RES002","2024-01-15","Morning","James Thompson","Care Worker","Daily Care",
         "Completed","Low mood","Fair","320",78.5,"Yes","Redness on left heel","Yes",
         "Yes","Chair-based exercises","Poor","Yes","Left heel",0,
         "Albert appeared low in mood this morning. Personal care was completed with full assistance. Redness noted on left heel — pressure relieving cushion applied and repositioning schedule reviewed. Appetite was fair; encouraged fluids.",
         "Redness on left heel — monitor closely","Repositioning schedule implemented; family informed","Heel redness requires monitoring. Pain management reviewed.",
         1,"approved","David Patel","2024-01-15 14:00:00"),
    ]
    c.executemany("""INSERT INTO care_notes
        (resident_id,date,shift,staff_name,staff_role,note_type,personal_care,
         mood,appetite,fluid_intake_ml,weight_kg,skin_checked,skin_concern,
         repositioned,activity,activity_description,sleep_quality,pain_observed,
         pain_location,falls_this_shift,care_narrative,concerns,actions_taken,
         handover_notes,ai_generated,status,approved_by,approved_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        care_notes
    )

    # ── Demo Incident ──────────────────────────────────────────────────────────
    c.execute("""INSERT INTO incidents
        (incident_id,resident_id,date,time,shift,incident_type,severity,location,
         witnessed,witness_name,staff_first_on_scene,description,immediate_actions,
         injuries,medical_attention,outcome,gp_notified,family_notified,
         cqc_notification,risk_assessment_updated,care_plan_updated,
         investigation_required,lessons_learned,status)
        VALUES
        ('INC-2024-001','RES001','2024-01-14','10:35','Morning','Fall','Minor','Bedroom',
         'Yes','Emma Williams','Emma Williams',
         'Maggie was found on the floor beside her bed. She had attempted to get up independently without using her call bell.',
         'Resident assisted to chair, pain assessment completed, GP notified.',
         'Small bruise on right hip — no fractures noted on examination.',
         'GP contacted — no hospital admission required. Ice pack applied.',
         'Resident comfortable, no further deterioration.',
         'Yes','Yes','No','Yes','Yes','No',
         'Call bell positioned closer; bed sensor activated; family informed.',
         'closed')
    """)

    # ── Demo Medications ───────────────────────────────────────────────────────
    meds = [
        ("RES001","Donepezil","Donepezil","10mg","Oral","Once daily at night",
         "Alzheimer's Dementia","Dr. A. Khan","2022-01-10","2024-07-10",0,0,
         "","Administer with evening meal","Monitor for GI side effects","active"),
        ("RES001","Amlodipine","Amlodipine","5mg","Oral","Once daily in the morning",
         "Hypertension","Dr. A. Khan","2022-01-10","2024-07-10",0,0,
         "","","Monitor BP","active"),
        ("RES002","Co-careldopa","Co-careldopa 25/100","1 tablet","Oral","Three times daily",
         "Parkinson's Disease","Dr. A. Khan","2021-06-01","2024-06-01",0,0,
         "","DO NOT crush. Administer 30 mins before meals.",
         "Monitor for dyskinesia, nausea","active"),
    ]
    c.executemany("""INSERT INTO medications
        (resident_id,medication_name,generic_name,dose,route,frequency,indication,
         prescribing_gp,start_date,review_date,is_controlled,is_prn,prn_instructions,
         admin_notes,side_effects,status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", meds
    )

    conn.commit()
    print("✅  Demo data seeded.")


if __name__ == "__main__":
    init_db()
