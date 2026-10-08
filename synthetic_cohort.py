"""
synthetic_cohort.py — the generative model behind the CareHome AI dataset

What this file is
This is not a "fake data filler". It is the data generating process (DGP) for
the whole project, written explicitly so that every number reported downstream can
be traced back to the equations that produced it. In a data science dissertation
the DGP is a first-class artefact: it defines the ground truth against which the
learning pipeline is judged.

The model is a hierarchical, discrete-time latent state-space model with a
survival (hazard) outcome layer. Formally, for resident i on day t:

    LEVEL 1 — resident random effect (between-subject heterogeneity)
        frailty_i  ~  Beta(2.2, 2.6)                       (i = 1 … N)

    LEVEL 2 — latent health state (within-subject temporal process)
        d_i(t) = clip( level_i(t) + episode_i(t), 0, 1 )
        level_i(t) = level_i(t-1) + delta_i + eps_t,   eps_t ~ N(0, 0.010²)
        level_i(0) ~ N(0.18 + 0.35·frailty_i, 0.06²)
        delta_i    ~ N(0.00035, 0.0004²) · (1 + frailty_i)      slow drift
        episode_i(t): 0–3 trapezoidal excursions (ramp / plateau / recovery)

      The bounded random walk is what makes d_i(t) *autocorrelated*. Without
      autocorrelation, nothing observed at time t could possibly inform an
      outcome at t+7, and 7-day-ahead prediction would be provably impossible.
      The trapezoidal episodes stand in for an infection, a medication change,
      or a bereavement — clinically, deterioration arrives in bouts, not as
      smooth drift.

    LEVEL 3 — observation model (measurement given latent state)
        Every observable the feature extractor reads is drawn from a conditional
        distribution given d_i(t). See OBSERVATION MODEL below.

    LEVEL 4 — outcome model (discrete-time logistic survival / hazard)
        logit h_i(t) = β0 + β_d·d_i(t) + β_f·frailty_i + β_r·min(recent_falls, 3)
        Fall_i(t) ~ Bernoulli( h_i(t) )

      This is the standard discrete-time survival formulation (Singer & Willett,
      1993): a binary hazard per person-period, with the recent-falls term giving
      the well-documented recurrence effect.

    LEVEL 5 — missingness mechanism (deliberately MNAR)
        P(care note recorded on day t) = 0.90 − 0.12·d_i(t)

      Documentation *thins* as a resident deteriorates — a real and well known
      phenomenon in care homes (staff time is absorbed by direct care). Because
      the probability of observation depends on the unobserved state itself,
      this is Missing Not At Random. It is included on purpose: a pipeline that
      only works under MCAR is not a pipeline that works.

Why synthetic data at all
Care home records are special category personal data under UK GDPR Article 9.
No lawful basis existed for processing real records for an MSc project, and no
public care-home EHR corpus exists. The alternatives were (a) no data, (b) real
data under an unobtainable approval, or (c) a simulator whose generating
equations are known. Option (c) is the only one available, and it carries a
methodological advantage that real data does not: the ground truth is known,
so the pipeline can be tested for whether it recovers a signal that is provably
present, and — equally important — for whether it *fails* to hallucinate a signal
when one is provably absent (see the v1 null result, Chapter 6).

The corresponding cost is stated plainly and repeatedly: any accuracy figure
obtained here measures the pipeline's ability to recover equations written in
this file. It is internal validity, not clinical validity.

Calibration to published epidemiology
The hazard intercept was tuned so that the marginal fall rate lands inside the
range observed in the FinCH cluster-randomised trial of 84 UK care homes and
1,657 residents: 2.2 falls per resident-year (intervention arm) to 3.8 falls per
resident-year (usual care arm) — Logan et al., Health Technology Assessment 26(9),
NIHR Journals Library, 2022. This cohort produces ~2.8 falls per resident-year.

The class balance was NOT tuned. Roughly 4–5 % of 7-day windows contain a fall,
which is what a clinically plausible rate implies. Balancing the classes would
have produced a flattering headline accuracy and destroyed the very property that
makes the evaluation meaningful.

Version history
v1 (seed_data.py, retained)  5 residents. Fall events drawn at a fixed per-
                             resident monthly rate, INDEPENDENT of the time-
                             varying features. Consequence: nothing to learn
                             beyond a static mapping from fixed characteristics
                             to a fixed rate; LOSO AUC 0.632, CI spanning 0.5.
v2 (this file)               50 residents. Fall hazard coupled to the latent
                             state that also drives the observables, so features
                             at t genuinely carry information about t+7.

Usage
    python synthetic_cohort.py                 # regenerate carehome.db (50 residents)
    python synthetic_cohort.py --out other.db  # write elsewhere
    python synthetic_cohort.py --residents 20  # smaller cohort
    python synthetic_cohort.py --validate      # generate, then print fidelity report
    python synthetic_cohort.py --report out.json   # machine-readable validation

Reproducibility
Single seed (SEED below) drives one random.Random instance used for every draw,
in a fixed order. Re-running with the same seed and the same resident count
reproduces the database byte-for-byte in content.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import sys

# Parameters of the generating process
# Every coefficient below is reported in the dissertation. Changing one changes
# the ground truth, so they are collected here rather than scattered in the code.

SEED        = 4242
N_RESIDENTS = 50
DAYS        = 365
END_DATE    = datetime.date(2026, 8, 20)

# ── Level 4: fall hazard (log-odds scale) ────────────────────────────────────
FALL_INTERCEPT  = -9.75   # log-odds of a fall on a day with d = 0, frailty = 0
FALL_BETA_D     =  6.00   # effect of latent deterioration
FALL_BETA_FRAIL =  1.60   # effect of static frailty (between-subject)
FALL_BETA_RECUR =  0.35   # per prior fall in the last 30 days, capped at 3

OTHER_INC_BASE   = 0.0015  # daily hazard of a non-fall incident at d = 0
OTHER_INC_BETA_D = 0.0100

# ── Level 3: observation model ───────────────────────────────────────────────
#   fluid_ml      ~ N(1700 − 800·d − 120·frailty, 95²)
#   wellbeing     ~ N(9.6 − 6.8·d − 1.4·frailty, 0.42²)      fortnightly
#   P(dose missed) = 0.010 + 0.20·d      -> ~10 % marginal omission rate
#   risk_score    ~ N(3.5 + 19·d + 4·frailty, 1.1²)          every 60 days
FLUID_BASE, FLUID_BETA_D, FLUID_BETA_F, FLUID_SD = 1700.0, 800.0, 120.0, 95.0
WB_BASE,    WB_BETA_D,    WB_BETA_F,    WB_SD    =    9.6,   6.8,   1.4,  0.42
MISS_BASE,  MISS_BETA_D                          =  0.010,  0.20
RISK_BASE,  RISK_BETA_D,  RISK_BETA_F, RISK_SD   =    3.5,  19.0,   4.0,  1.1

# ── Level 5: MNAR documentation mechanism ────────────────────────────────────
NOTE_BASE, NOTE_BETA_D = 0.90, 0.12

# ── Level 2: latent process ──────────────────────────────────────────────────
LEVEL0_INTERCEPT, LEVEL0_FRAILTY, LEVEL0_SD = 0.18, 0.35, 0.06
DRIFT_MEAN, DRIFT_SD = 0.00035, 0.0004
WALK_SD = 0.010
LEVEL_MIN, LEVEL_MAX = 0.02, 0.92

DGP_VERSION = "2.0"


# CONTENT BANKS — the surface realisation of the latent state
# RAG retrieval is evaluated on this text, so the narratives must vary in wording
# while remaining faithful to d(t). Template-with-slots generation (rather than
# a single fixed sentence) is what stops the retrieval benchmark from degenerating
# into exact-duplicate matching.

DIAGNOSES = ["Alzheimer's Dementia", "Vascular Dementia", "Parkinson's Disease",
             "Frailty Syndrome", "COPD", "Stroke (CVA)", "Type 2 Diabetes",
             "Heart Failure", "Osteoarthritis", "Lewy Body Dementia"]

SECONDARY = ["Hypertension", "Osteoporosis", "Chronic kidney disease (stage 3)",
             "Depression", "Hypothyroidism", "Atrial fibrillation",
             "Macular degeneration", "Hearing impairment", "Constipation",
             "Recurrent urinary tract infection"]

MOBILITY = ["Independent", "Walking stick", "Walking frame, supervised",
            "Requires 1-person assist", "Requires 2-person assist",
            "Wheelchair dependent", "Hoist transfer only"]

ALLERGIES = ["None known", "Penicillin", "Sulfonamides", "Latex", "Codeine",
             "None known", "None known", "Aspirin"]

FIRST = ["Margaret", "Doris", "Ethel", "Arthur", "Albert", "Joan", "Betty", "Harold",
         "Edith", "Stanley", "Vera", "Norman", "Iris", "Reginald", "Gladys", "Cyril",
         "Mavis", "Wilfred", "Nora", "Leonard", "Phyllis", "Percy", "Enid", "Sidney",
         "Muriel", "Clifford", "Beryl", "Ronald", "Hilda", "Maurice", "Sylvia",
         "Herbert", "Rita", "Frank", "Audrey", "Cecil", "Pauline", "Alfred",
         "Marjorie", "Ernest", "Brenda", "Walter", "Sheila", "Victor", "Kathleen",
         "Horace", "Dorothy", "Bernard", "Gwen", "Rowland"]

# Names alternate between traditionally female and male; gender follows the
# name so that notes ("she"/"he") read consistently with the resident's name.
MALE_FIRST = {"Arthur", "Albert", "Harold", "Stanley", "Norman", "Reginald",
              "Cyril", "Wilfred", "Leonard", "Percy", "Sidney", "Clifford",
              "Ronald", "Maurice", "Herbert", "Frank", "Cecil", "Alfred",
              "Ernest", "Walter", "Victor", "Horace", "Bernard", "Rowland"}

LAST = ["Brown", "Wilson", "Taylor", "Davies", "Evans", "Thomas", "Roberts", "Walker",
        "Wright", "Hughes", "Green", "Hall", "Wood", "Harris", "Clarke", "Jackson",
        "Turner", "Hill", "Cooper", "Ward", "Morris", "Bell", "Baker", "Cook",
        "Bailey", "Murphy", "Rogers", "Gray", "James", "Watson", "Price", "Bennett",
        "Kelly", "Barnes", "Shaw", "Fisher", "Chapman", "Reid", "Marshall", "Owen",
        "Palmer", "Holmes", "Webb", "Ellis", "Grant", "Knight", "Lane", "Barker",
        "Dixon", "Hunt"]

MEDS = [("Amlodipine", "5mg", "Oral", "Once daily", "Hypertension"),
        ("Atorvastatin", "20mg", "Oral", "Once daily", "Hyperlipidaemia"),
        ("Donepezil", "10mg", "Oral", "Once daily", "Alzheimer's disease"),
        ("Levodopa/Carbidopa", "100/25mg", "Oral", "Three times daily", "Parkinson's disease"),
        ("Furosemide", "40mg", "Oral", "Once daily", "Oedema"),
        ("Paracetamol", "1g", "Oral", "Four times daily", "Pain"),
        ("Lansoprazole", "30mg", "Oral", "Once daily", "Gastro-protection"),
        ("Sertraline", "50mg", "Oral", "Once daily", "Low mood"),
        ("Metformin", "500mg", "Oral", "Twice daily", "Type 2 diabetes"),
        ("Bisoprolol", "2.5mg", "Oral", "Once daily", "Heart failure"),
        ("Alendronic acid", "70mg", "Oral", "Once weekly", "Osteoporosis"),
        ("Colecalciferol", "800iu", "Oral", "Once daily", "Vitamin D deficiency"),
        ("Rivaroxaban", "20mg", "Oral", "Once daily", "Atrial fibrillation"),
        ("Levothyroxine", "75mcg", "Oral", "Once daily", "Hypothyroidism"),
        ("Macrogol", "1 sachet", "Oral", "Twice daily", "Constipation")]

STAFF = ["S. Patel", "J. Okafor", "M. Nowak", "L. Ahmed", "R. Kowalski", "D. Osei",
         "C. Fernandes", "A. Whitfield", "T. Nguyen", "K. Adeyemi", "B. Mensah",
         "P. Sharma"]

SHIFTS = ["Morning", "Afternoon", "Night"]
GPS = [("Dr. A. Khan", "01234 567890"), ("Dr. B. Osei", "01234 567891"),
       ("Dr. C. Lindqvist", "01234 567892")]

PERSONAL_CARE = ["Full wash at the sink", "Assisted shower", "Bed bath",
                 "Supported wash with encouragement", "Full assistance with washing and dressing",
                 "Strip wash in bedroom"]

ACTIVITIES = ["Armchair exercises", "Reminiscence group", "Music session",
              "Garden walk with staff", "One-to-one chat", "Bingo in the lounge",
              "Hand massage", "Newspaper reading", "Visiting singer",
              "Craft table", "Rested in room"]

# Narrative fragments, indexed by deterioration band. band(d): 0 = settled,
# 1 = some concern, 2 = significant concern.
OPENERS = [
    ["{name} has had a settled {shift_l}.",
     "A good {shift_l} for {name}.",
     "{name} presented as {poss} usual self this {shift_l}.",
     "No concerns raised for {name} during the {shift_l}."],
    ["{name} has been less settled this {shift_l}.",
     "Staff noted a change in {name} over the {shift_l}.",
     "{name} needed more prompting than usual this {shift_l}.",
     "A more difficult {shift_l} for {name}."],
    ["{name} has had a poor {shift_l} and needed close support.",
     "Significant concerns for {name} throughout the {shift_l}.",
     "{name} was noticeably unwell during the {shift_l}.",
     "Increased supervision required for {name} this {shift_l}."],
]

CARE_LINES = [
    ["{pc} completed with the usual level of support. Skin checked, intact throughout.",
     "{pc} carried out; {name} was able to assist with parts of {poss} own care.",
     "{pc}; no skin concerns noted, pressure areas clear."],
    ["{pc} completed but {name} was reluctant and needed encouragement.",
     "{pc} with two staff as {name} was unsteady on transfer.",
     "{pc} completed; slight redness noted to the sacrum, repositioning chart started."],
    ["{pc} required full assistance from two staff; {name} was unable to weight-bear reliably.",
     "{pc} completed with difficulty. {name} was resistive at times and care was paused twice.",
     "{pc} with hoist assistance. Skin fragile, barrier cream applied."],
]

INTAKE_LINES = [
    "Fluids {fluid} ml over the {shift_l}, {appetite_l} appetite at meals.",
    "Took {fluid} ml of fluids; ate {appetite_l} at lunch and tea.",
    "Fluid intake recorded at {fluid} ml. Appetite described as {appetite_l}.",
    "{fluid} ml of fluid encouraged across the shift; appetite {appetite_l}.",
]

MOOD_LINES = [
    ["Mood {mood_l} and engaged well with {activity_l}.",
     "Presented as {mood_l}; joined in with {activity_l}.",
     "Bright and {mood_l} for most of the shift; enjoyed {activity_l}."],
    ["Mood {mood_l} at times; declined {activity_l} and preferred to stay in {poss} room.",
     "Appeared {mood_l}; needed reassurance on several occasions.",
     "Mood fluctuated and was largely {mood_l}. Did not settle to {activity_l}."],
    ["Mood {mood_l} throughout; calling out intermittently and difficult to reassure.",
     "Very {mood_l}. Attempted to mobilise unaided twice and required redirection.",
     "{mood_l} and restless. Remained in bed for most of the shift."],
]

CONCERN_LINES = [
    ["No concerns to hand over.",
     "Nothing further to report.",
     "Continue with the current plan of care."],
    ["Monitor fluid intake and report any further decline to the senior on duty.",
     "Please continue to observe closely; senior carer informed.",
     "Fluids to be encouraged on the next shift; keep under observation."],
    ["Escalated to the senior on duty. Consider GP review if no improvement.",
     "Falls risk discussed at handover; sensor mat in place and hourly checks agreed.",
     "GP review requested. Family to be updated by the manager."],
]

INCIDENT_DESCRIPTIONS = [
    "{name} was found on the floor beside the bed at {time}. No witnesses to the fall.",
    "{name} was seen to slip while transferring from the armchair at {time}. Staff were present but unable to prevent the fall.",
    "{name} was heard calling out and was found sitting on the floor of the {loc_l} at {time}.",
    "{name} lost balance while walking to the {loc_l} with a walking frame at {time} and lowered to the floor.",
    "{name} was found kneeling beside the commode at {time}, having attempted to mobilise unaided.",
]

CARE_PLAN_TEMPLATES = {
    "personal_identity_summary":
        "{name} is a {age}-year-old {gender_l} who has lived at the home since {admission}. "
        "{pronoun_c} prefers to be called {pref}. Primary diagnosis is {diagnosis}. "
        "{pronoun_c} worked as a {occupation} and values {value}. Staff should introduce "
        "themselves each time as {pronoun} does not always retain new faces.",
    "mobility_care_plan":
        "Mobility status: {mobility}. Falls risk assessed as {falls_risk}. "
        "{pronoun_c} requires {assist} for all transfers. Ensure the call bell and "
        "walking aid are within reach at all times. {sensor}",
    "personal_care_plan":
        "{pronoun_c} requires {assist} with washing and dressing and prefers to be "
        "supported in the morning. Skin integrity to be checked at every episode of "
        "personal care; pressure sore risk is {pressure}.",
    "continence_care_plan":
        "Continence needs: {continence}. Toileting to be offered on a two-hourly basis "
        "and after meals. Record output on the fluid balance chart.",
    "nutrition_hydration_plan":
        "Diet texture: {texture}. Target fluid intake 1500–2000 ml per 24 hours. "
        "Weekly weights. Fortified diet if weight loss exceeds 3 % in one month. "
        "Fluids to be offered at every interaction, not only at mealtimes.",
    "medication_management_plan":
        "{n_meds} regular medicines are prescribed, administered by trained staff and "
        "signed on the MAR chart. Any refusal must be recorded with a reason and "
        "reported to the senior on duty. Covert administration is not authorised.",
    "cognitive_support_plan":
        "Cognitive status consistent with {diagnosis}. Use short, single-step "
        "instructions. Avoid correcting disorientation; use validation and "
        "redirection. Familiar objects to remain in the same place in the bedroom.",
    "emotional_wellbeing_plan":
        "{pronoun_c} can become anxious in the late afternoon. Reassurance, a familiar "
        "voice and {value} are effective. Escalate any sustained low mood to the "
        "senior carer for wellbeing review.",
    "social_activity_plan":
        "Enjoys {activity1} and {activity2}. Invite to group activities daily but "
        "accept refusal without pressure. Maintain contact with {nok} by telephone weekly.",
    "end_of_life_preferences":
        "{dnacpr}. Advance care planning discussion held with {nok} ({nok_rel}). "
        "Preferred place of care is the home. Spiritual needs: {faith}.",
    "risk_summary":
        "Falls risk {falls_risk}; pressure sore risk {pressure}; nutritional risk "
        "monitored monthly. Risk assessments reviewed every 60 days or after any "
        "incident, whichever is sooner.",
    "goals_of_care":
        "1. Maintain current mobility with {assist} and avoid unwitnessed falls. "
        "2. Achieve the daily fluid target on at least five days per week. "
        "3. Participate in at least three social activities per week. "
        "4. Maintain weight within 2 kg of the current baseline.",
    "family_involvement_plan":
        "{nok} ({nok_rel}) is the main contact and wishes to be informed of any fall, "
        "GP visit or change in medication. Routine update by telephone every fortnight.",
}

OCCUPATIONS = ["seamstress", "railway engineer", "primary school teacher", "farmer",
               "nurse", "shopkeeper", "postal worker", "bookkeeper", "carpenter",
               "telephonist", "police officer", "baker"]
VALUES = ["her independence", "time in the garden", "music from the 1950s",
          "a strong cup of tea", "letters from family", "the daily newspaper"]
FAITHS = ["Church of England", "Roman Catholic", "No religious affiliation",
          "Methodist", "Not stated"]


# The latent process

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def latent_trajectory(rng: random.Random, frailty: float, days: int = DAYS) -> list[float]:
    """
    Simulate d(t) ∈ [0, 1], the latent deterioration state, for one resident.

    Three additive components, each with a clinical reading:

      * slow drift        — some residents decline across the year, some do not;
      * bounded random walk — makes state autocorrelated day to day. This is the
                            component that makes 7-day-ahead prediction possible
                            at all: without it, d(t) and d(t+7) would be
                            independent and no feature could carry information
                            about the label;
      * 0–3 episodes      — trapezoidal excursions (ramp over 10–24 days, plateau
                            4–16 days, partial recovery over 14–40 days) standing
                            in for infection, medication change or bereavement.

    Returns a list of length `days`.
    """
    d = [0.0] * days
    level = clamp(rng.gauss(LEVEL0_INTERCEPT + LEVEL0_FRAILTY * frailty, LEVEL0_SD),
                  LEVEL_MIN, 0.75)
    drift = rng.gauss(DRIFT_MEAN, DRIFT_SD) * (1 + frailty)

    episodes = []
    for _ in range(rng.choice([0, 1, 1, 2, 2, 3])):
        start = rng.randrange(0, max(1, days - 40))
        up    = rng.randint(10, 24)
        plat  = rng.randint(4, 16)
        down  = rng.randint(14, 40)
        amp   = clamp(rng.gauss(0.34, 0.11), 0.10, 0.62)
        episodes.append((start, up, plat, down, amp))

    for t in range(days):
        level = clamp(level + drift + rng.gauss(0, WALK_SD), LEVEL_MIN, LEVEL_MAX)
        bump = 0.0
        for (s, up, plat, down, amp) in episodes:
            if t < s:
                continue
            k = t - s
            if k < up:
                bump += amp * (k / up)
            elif k < up + plat:
                bump += amp
            elif k < up + plat + down:
                bump += amp * (1 - (k - up - plat) / down)
        d[t] = clamp(level + bump, 0.0, 1.0)
    return d


def band(d: float) -> int:
    """Map latent state to a narrative severity band (0 settled, 1 concern, 2 marked)."""
    return 0 if d < 0.33 else (1 if d < 0.62 else 2)


# Generation

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


DEMO_USERS = [
    ("manager1",  "manager123",  "Sarah Johnson",  "manager",      "manager@sunrisecare.example.com"),
    ("senior1",   "senior123",   "David Patel",    "senior_carer", "senior1@sunrisecare.example.com"),
    ("carer1",    "carer123",    "Emma Williams",  "care_worker",  "carer1@sunrisecare.example.com"),
    ("carer2",    "carer123",    "James Thompson", "care_worker",  "carer2@sunrisecare.example.com"),
    ("readonly1", "readonly123", "Dr. A. Khan",    "readonly",     "gp@practice.example.com"),
]


def _create_schema(db_path: str) -> None:
    """
    Create the project schema in a fresh file.

    database.init_db() is the single source of truth for the schema, so it is
    reused rather than duplicated here — a second copy of 16 CREATE TABLE
    statements would drift. Its demo-seed step is suppressed: this module is
    the seeder, and letting init_db() insert its own five residents first would
    silently mix two different generating processes in one database.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import database
    original_path, original_seed = database.DB_PATH, database._seed_demo_data
    database.DB_PATH = db_path
    database._seed_demo_data = lambda conn: None
    try:
        database.init_db()
    finally:
        database.DB_PATH = original_path
        database._seed_demo_data = original_seed


def generate(out_path: str,
             n_residents: int = N_RESIDENTS,
             days: int = DAYS,
             seed: int = SEED,
             end_date: datetime.date = END_DATE,
             verbose: bool = True) -> dict:
    """
    Generate the full cohort and write it to `out_path`, overwriting any existing
    file. Returns a stats dict, plus the per-resident latent trajectories and
    frailties under key "_truth" for the validation report.
    """
    rng_setup = random.Random(seed)
    start_date = end_date - datetime.timedelta(days=days - 1)

    for suffix in ("", "-wal", "-shm"):
        p = out_path + suffix
        if os.path.exists(p):
            os.remove(p)

    _create_schema(out_path)

    conn = sqlite3.connect(out_path)
    c = conn.cursor()

    # ── Users ────────────────────────────────────────────────────────────────
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.executemany(
            "INSERT INTO users (username,password,full_name,role,email) VALUES (?,?,?,?,?)",
            [(u, _hash_pw(p), f, r, e) for (u, p, f, r, e) in DEMO_USERS])

    stats = dict(residents=0, care_notes=0, incidents=0, falls=0, mar=0,
                 wellbeing=0, risk=0, meds=0, handovers=0, care_plans=0,
                 family_comms=0, resident_days=0)
    truth = []

    for i in range(1, n_residents + 1):
        # ── Independent random streams per resident ──────────────────────────
        # Each resident draws from streams seeded by (SEED, i) rather than from
        # one shared generator. Two consequences that matter for a dissertation:
        #   * resident k is IDENTICAL whether the cohort has 20 or 50 members,
        #     so cohort size can be varied as an experimental factor without
        #     confounding it with a change in every resident's history;
        #   * the latent trajectory has its own stream, so re-tuning an
        #     observation-model parameter (e.g. medication count) cannot shift
        #     the underlying health states and silently move the fall rate.
        rng      = random.Random(seed * 1_000_003 + i)   # demographics, observables
        rng_traj = random.Random(seed * 7_919 + i)       # latent state only
        rid   = f"RES{i:03d}"
        first = FIRST[(i - 1) % len(FIRST)]
        last  = LAST[(i - 1) % len(LAST)]
        name  = f"{first} {last}"

        # ── Level 1: resident random effect ──────────────────────────────────
        frailty = clamp(rng_traj.betavariate(2.2, 2.6), 0.02, 0.98)

        age    = int(clamp(rng.gauss(84 + 6 * frailty, 6), 65, 101))
        rng.choice(["Female", "Male"])   # draw kept so later values in this stream are unchanged
        gender = "Male" if first in MALE_FIRST else "Female"
        pronoun = "she" if gender == "Female" else "he"
        poss    = "her" if gender == "Female" else "his"
        mobility = MOBILITY[min(len(MOBILITY) - 1, int(frailty * len(MOBILITY)))]
        falls_risk = ("Very High" if frailty > 0.75 else "High" if frailty > 0.50
                      else "Medium" if frailty > 0.25 else "Low")
        pressure = "High" if frailty > 0.60 else "Medium"
        capacity = "Lacks capacity" if frailty > 0.60 else "Has capacity"
        continence = ("Incontinent - pads" if frailty > 0.70 else
                      "Continent with prompting" if frailty > 0.35 else "Continent")
        texture = ("Pureed" if frailty > 0.80 else "Soft & bite-sized"
                   if frailty > 0.50 else "Normal")
        assist = ("full assistance from two staff" if frailty > 0.70 else
                  "assistance from one member of staff" if frailty > 0.40 else
                  "supervision and prompting")
        admission = start_date - datetime.timedelta(days=rng.randint(60, 1800))
        gp_name, gp_phone = rng.choice(GPS)
        diagnosis = rng.choice(DIAGNOSES)
        nok_first = rng.choice(FIRST)
        nok_rel = rng.choice(["Son", "Daughter", "Niece", "Nephew", "Spouse", "Sister"])
        nok = f"{nok_first} {last}"
        key_worker = rng.choice(STAFF)
        dnacpr = rng.choice(["DNACPR in place", "Full resuscitation",
                             "DNACPR in place", "For active treatment"])

        c.execute("""INSERT INTO residents
            (resident_id, full_name, preferred_name, date_of_birth, age, gender,
             room_number, admission_date, care_type, primary_diagnosis,
             secondary_diagnoses, allergies, gp_name, gp_phone, dnacpr_status,
             mental_capacity, mobility_level, continence_needs, nutrition_texture,
             dietary_restrictions, falls_risk, pressure_sore_risk,
             nok_name, nok_relationship, nok_phone, nok_email, key_worker, active)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (rid, name, first,
             (datetime.date(2026, 1, 1) - datetime.timedelta(days=age * 365)).isoformat(),
             age, gender, f"Room {i}", admission.isoformat(),
             rng.choice(["Residential", "Nursing", "Dementia"]), diagnosis,
             ", ".join(rng.sample(SECONDARY, rng.randint(1, 3))),
             rng.choice(ALLERGIES), gp_name, gp_phone, dnacpr, capacity,
             mobility, continence, texture,
             rng.choice(["None", "None", "Diabetic", "Low sodium", "Gluten free"]),
             falls_risk, pressure, nok, nok_rel,
             f"07700 9{i:05d}", f"{nok_first.lower()}.{last.lower()}@example.com",
             key_worker))
        stats["residents"] += 1

        # ── Medications ──────────────────────────────────────────────────────
        # Polypharmacy scales with frailty. Capped at 9 regular medicines: the MAR
        # table is one row per medicine per day, so this parameter alone drives
        # roughly half the database size.
        n_meds = int(clamp(round(rng.gauss(3.0 + 4.5 * frailty, 1.3)), 2, 9))
        med_ids = []
        for (mname, dose, route, freq, ind) in rng.sample(MEDS, n_meds):
            c.execute("""INSERT INTO medications
                (resident_id, medication_name, dose, route, frequency, indication,
                 prescribing_gp, start_date, review_date, status, is_prn, is_controlled,
                 admin_notes)
                VALUES (?,?,?,?,?,?,?,?,?,'active',0,0,?)""",
                (rid, mname, dose, route, freq, ind, gp_name,
                 (start_date - datetime.timedelta(days=rng.randint(30, 900))).isoformat(),
                 (end_date + datetime.timedelta(days=rng.randint(10, 180))).isoformat(),
                 "Administer with food." if mname in ("Metformin", "Lansoprazole")
                 else "Crush only if the resident is on a modified texture diet."))
            med_ids.append(c.lastrowid)
        stats["meds"] += n_meds

        # ── Care plan ───────────────────────────────────────────────────────
        # status 'Active' matches the state machine app.py uses when a new plan
        # supersedes an old one. The v1 generator wrote 'approved', which no
        # query in the codebase looked for — the root cause of the missing
        # care-plan chunks documented in Chapter 6.
        act1, act2 = rng.sample(ACTIVITIES[:9], 2)
        plan_vals = dict(
            name=name, age=age, gender_l=gender.lower(), admission=admission.strftime("%B %Y"),
            pref=first, diagnosis=diagnosis, occupation=rng.choice(OCCUPATIONS),
            value=rng.choice(VALUES), pronoun=pronoun, pronoun_c=pronoun.capitalize(),
            mobility=mobility, falls_risk=falls_risk, assist=assist,
            sensor=("A falls sensor mat is in use overnight." if frailty > 0.6 else
                    "No sensor equipment is currently required."),
            pressure=pressure, continence=continence, texture=texture, n_meds=n_meds,
            activity1=act1.lower(), activity2=act2.lower(), nok=nok, nok_rel=nok_rel.lower(),
            dnacpr=dnacpr, faith=rng.choice(FAITHS),
        )
        plan = {k: v.format(**plan_vals) for k, v in CARE_PLAN_TEMPLATES.items()}
        c.execute("""INSERT INTO care_plans
            (resident_id, version, effective_from, review_date, reviewed_by, status,
             personal_identity_summary, mobility_care_plan, personal_care_plan,
             continence_care_plan, nutrition_hydration_plan, medication_management_plan,
             cognitive_support_plan, emotional_wellbeing_plan, social_activity_plan,
             end_of_life_preferences, risk_summary, goals_of_care,
             family_involvement_plan, ai_generated, generation_method)
            VALUES (?,1,?,?,?,'Active',?,?,?,?,?,?,?,?,?,?,?,?,?,0,'seeded')""",
            (rid, start_date.isoformat(),
             (end_date + datetime.timedelta(days=60)).isoformat(), "Sarah Johnson",
             plan["personal_identity_summary"], plan["mobility_care_plan"],
             plan["personal_care_plan"], plan["continence_care_plan"],
             plan["nutrition_hydration_plan"], plan["medication_management_plan"],
             plan["cognitive_support_plan"], plan["emotional_wellbeing_plan"],
             plan["social_activity_plan"], plan["end_of_life_preferences"],
             plan["risk_summary"], plan["goals_of_care"],
             plan["family_involvement_plan"]))
        stats["care_plans"] += 1

        # ── Level 2: latent trajectory ───────────────────────────────────────
        d = latent_trajectory(rng_traj, frailty, days)
        # The latent trajectory is retained, not discarded. Because d(t) and the
        # hazard coefficients are known, the BAYES-OPTIMAL predictor for this
        # problem can be computed exactly (see oracle_benchmark.py) — which
        # turns "is 0.86 good?" from a matter of opinion into a measurement
        # against the irreducible ceiling. Real data can never offer this.
        truth.append({"resident_id": rid, "frailty": round(frailty, 4),
                      "d_mean": round(sum(d) / len(d), 4),
                      "d_max": round(max(d), 4),
                      "d": [round(v, 4) for v in d]})

        recent_falls = 0
        n_falls_this_resident = 0

        for t in range(days):
            day = start_date + datetime.timedelta(days=t)
            ds  = day.isoformat()
            dt_ = d[t]
            b   = band(dt_)
            stats["resident_days"] += 1

            # ── Level 5: MNAR documentation ──────────────────────────────────
            if rng.random() < NOTE_BASE - NOTE_BETA_D * dt_:
                fluid = int(clamp(rng.gauss(FLUID_BASE - FLUID_BETA_D * dt_
                                            - FLUID_BETA_F * frailty, FLUID_SD), 250, 2600))
                mood = ["Settled", "Low", rng.choice(["Anxious", "Confused", "Agitated"])][b]
                appetite = ["Good", "Fair", "Poor"][b]
                pain = "Yes" if rng.random() < 0.10 + 0.45 * dt_ else "No"
                shift = rng.choice(SHIFTS)
                pc = rng.choice(PERSONAL_CARE)
                activity = rng.choice(ACTIVITIES)
                slots = dict(name=first, shift_l=shift.lower(), pc=pc, fluid=fluid, poss=poss,
                             appetite_l=appetite.lower(), mood_l=mood.lower(),
                             activity_l=activity.lower())
                narrative = " ".join([
                    rng.choice(OPENERS[b]).format(**slots),
                    rng.choice(CARE_LINES[b]).format(**slots),
                    rng.choice(INTAKE_LINES).format(**slots),
                    rng.choice(MOOD_LINES[b]).format(**slots),
                ])
                c.execute("""INSERT INTO care_notes
                    (resident_id, date, shift, staff_name, staff_role, note_type,
                     personal_care, mood, appetite, fluid_intake_ml, skin_checked,
                     repositioned, activity, activity_description, sleep_quality,
                     pain_observed, falls_this_shift, care_narrative, concerns,
                     actions_taken, handover_notes, status, ai_generated)
                    VALUES (?,?,?,?,'Care Worker','Daily',?,?,?,?,'Yes',?,?,?,?,?,0,?,?,?,?,'approved',0)""",
                    (rid, ds, shift, rng.choice(STAFF), pc, mood, appetite, fluid,
                     "Yes" if frailty > 0.6 else "No", activity,
                     f"{activity} — {'participated well' if b == 0 else 'brief participation' if b == 1 else 'declined'}.",
                     ["Good", "Disturbed", "Poor"][b], pain, narrative,
                     rng.choice(CONCERN_LINES[b]),
                     ["Routine care given." ,
                      "Fluids encouraged; senior carer made aware.",
                      "Escalated to senior carer; observations taken."][b],
                     ["Nil to hand over.", "Monitor intake and mood.",
                      "Falls precautions in place; see incident log."][b]))
                stats["care_notes"] += 1

            # ── Handover, every third day ────────────────────────────────────
            if t % 3 == 0:
                c.execute("""INSERT INTO handovers
                    (date, shift_ending, shift_starting, compiled_by, resident_id,
                     overall_summary, care_completed, concerns_next_shift,
                     outstanding_tasks, fluid_target_met, escalation_required,
                     status, ai_generated)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?, 'approved', 0)""",
                    (ds, "Morning", "Afternoon", rng.choice(STAFF), rid,
                     ["Settled shift, no concerns.",
                      "Increased support required; monitor closely.",
                      "Poor shift; escalation in place."][b],
                     "Personal care, meals and medication completed as planned.",
                     ["None.", "Watch fluid intake and mood.",
                      "Falls risk — hourly checks and sensor mat in place."][b],
                     ["Nil.", "Fluid chart to be totalled.",
                      "GP review outstanding."][b],
                     "Yes" if dt_ < 0.5 else "No",
                     "Yes" if dt_ > 0.7 and rng.random() < 0.25 else "No"))
                stats["handovers"] += 1

            # ── Medication administration (adherence observable) ─────────────
            p_miss = MISS_BASE + MISS_BETA_D * dt_
            for mid in med_ids:
                given = rng.random() >= p_miss
                c.execute("""INSERT INTO mar_records
                    (medication_id, resident_id, date, time_given, shift, given_by,
                     administered, refusal_reason)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (mid, rid, ds, "08:00", "Morning", rng.choice(STAFF),
                     "Yes" if given else "No",
                     None if given else rng.choice(
                         ["Resident declined", "Asleep", "Nausea", "Spat out", "Off ward"])))
                stats["mar"] += 1

            # ── Wellbeing assessment, fortnightly ────────────────────────────
            if t % 14 == 0:
                score = int(clamp(round(rng.gauss(WB_BASE - WB_BETA_D * dt_
                                                  - WB_BETA_F * frailty, WB_SD)), 1, 10))
                c.execute("""INSERT INTO wellbeing
                    (resident_id, assessment_date, assessed_by, period_covered,
                     physical_health_score, mental_health_score, social_engagement_score,
                     personal_care_score, nutrition_score, pain_management_score,
                     overall_score, physical_notes, mental_notes, summary,
                     status, ai_generated)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'approved', 0)""",
                    (rid, ds, rng.choice(STAFF), "Fortnight",
                     score, score, max(1, score - 1), score, score, score, score,
                     ["Physically stable over the period.",
                      "Some physical decline noted.",
                      "Marked physical deterioration this fortnight."][b],
                     ["Engaged and orientated.", "Intermittently low in mood.",
                      "Withdrawn and frequently distressed."][b],
                     f"Overall wellbeing scored {score}/10 for the fortnight ending {ds}."))
                stats["wellbeing"] += 1

            # ── Formal falls risk assessment, every 60 days ──────────────────
            if t % 60 == 0:
                sc = int(clamp(round(rng.gauss(RISK_BASE + RISK_BETA_D * dt_
                                               + RISK_BETA_F * frailty, RISK_SD)), 0, 25))
                lvl = ("very_high" if sc >= 18 else "high" if sc >= 12
                       else "medium" if sc >= 6 else "low")
                c.execute("""INSERT INTO risk_assessments
                    (resident_id, assessment_type, date_assessed, assessed_by,
                     review_date, score, risk_level, risk_factors, interventions,
                     narrative, status, ai_generated)
                    VALUES (?, 'falls', ?, ?, ?, ?, ?, ?, ?, ?, 'approved', 0)""",
                    (rid, ds, rng.choice(STAFF),
                     (day + datetime.timedelta(days=60)).isoformat(), sc, lvl,
                     f"{mobility}; polypharmacy ({n_meds} medicines); {diagnosis}.",
                     ("Hourly checks, sensor mat, low bed." if lvl in ("high", "very_high")
                      else "Standard falls precautions and call bell in reach."),
                     f"Falls risk assessed as {lvl.replace('_',' ')} with a score of {sc}/25 "
                     f"on {ds}. Review due in 60 days or sooner after any incident."))
                stats["risk"] += 1

            # ── Level 4: fall hazard ─────────────────────────────────────────
            logit = (FALL_INTERCEPT + FALL_BETA_D * dt_ + FALL_BETA_FRAIL * frailty
                     + FALL_BETA_RECUR * min(recent_falls, 3))
            if rng.random() < sigmoid(logit):
                sev = "Moderate" if dt_ > 0.6 else "Minor"
                loc = rng.choice(["Bedroom", "Lounge", "Corridor", "Bathroom", "Dining room"])
                tm = f"{rng.randint(6,23):02d}:{rng.choice(['05','15','20','35','40','55'])}"
                c.execute("""INSERT INTO incidents
                    (incident_id, resident_id, date, time, shift, incident_type, severity,
                     location, witnessed, witness_name, staff_first_on_scene, description,
                     immediate_actions, injuries, medical_attention, outcome,
                     gp_notified, family_notified, risk_assessment_updated,
                     investigation_required, lessons_learned, preventative_actions,
                     status, ai_generated)
                    VALUES (?,?,?,?,?, 'Fall', ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'closed', 0)""",
                    (f"INC-{rid}-{t:04d}", rid, ds, tm, rng.choice(SHIFTS), sev, loc,
                     rng.choice(["Yes", "No"]), rng.choice(STAFF), rng.choice(STAFF),
                     rng.choice(INCIDENT_DESCRIPTIONS).format(
                         name=first, time=tm, loc_l=loc.lower()),
                     "Assessed in situ for injury before moving. Neurological observations "
                     "commenced. Senior carer and nurse in charge informed.",
                     rng.choice(["None apparent", "Skin tear to left forearm",
                                 "Bruising to right hip", "Graze to knee",
                                 "Bruising to elbow"]),
                     "GP informed" if sev == "Moderate" else "No medical attention required",
                     "Monitoring continued; no further deterioration in the following 24 hours.",
                     "Yes" if sev == "Moderate" else "No", "Yes", "Yes",
                     "Yes" if sev == "Moderate" else "No",
                     "Fall occurred during an unsupervised transfer; call bell was in reach "
                     "but not used.",
                     "Falls risk assessment reviewed. Hourly checks continued and mobility "
                     "aid position reinforced with the resident and staff."))
                stats["incidents"] += 1
                stats["falls"] += 1
                recent_falls += 1
                n_falls_this_resident += 1
            elif rng.random() < OTHER_INC_BASE + OTHER_INC_BETA_D * dt_:
                itype = rng.choice(["Skin concern", "Behavioural", "Medication error",
                                    "Near miss", "Choking risk"])
                c.execute("""INSERT INTO incidents
                    (incident_id, resident_id, date, time, shift, incident_type, severity,
                     location, witnessed, staff_first_on_scene, description,
                     immediate_actions, status, ai_generated)
                    VALUES (?,?,?,?,?,?,'Minor',?,?,?,?,?, 'closed', 0)""",
                    (f"INC-{rid}-{t:04d}-O", rid, ds,
                     f"{rng.randint(7,21):02d}:05", rng.choice(SHIFTS), itype,
                     rng.choice(["Bedroom", "Lounge", "Dining room"]), "Yes",
                     rng.choice(STAFF),
                     f"{itype} recorded for {first} and actioned by the shift lead.",
                     "Recorded, resident reassured and senior carer informed."))
                stats["incidents"] += 1

            # ── Family communication, roughly monthly plus after every fall ──
            if t % 30 == 15:
                update = [
                    f"Reported that {pronoun} has been settled, is eating and drinking "
                    f"well and has joined in with activities.",
                    f"Explained that we have noticed some decline in {first}'s appetite "
                    f"and mood, and that fluids are being monitored closely.",
                    f"Explained the recent deterioration, the increased observations now "
                    f"in place and that a GP review has been requested.",
                ][b]
                c.execute("""INSERT INTO family_comms
                    (resident_id, date, comm_type, direction, staff_member,
                     family_contact, subject, trigger_event, body, follow_up,
                     ai_drafted)
                    VALUES (?,?,?,?,?,?,?,?,?,?,0)""",
                    (rid, ds, rng.choice(["Telephone", "Email", "In person"]),
                     "Outgoing", rng.choice(STAFF), f"{nok} ({nok_rel})",
                     f"Monthly update — {first}",
                     ["Routine update", "Change in condition",
                      "Deterioration / GP review"][b],
                     f"Spoke with {nok} to give an update on {first}. " + update,
                     ["No", "Yes", "Yes"][b]))
                stats["family_comms"] += 1

            if t % 30 == 0:
                recent_falls = 0

        if verbose and i % 10 == 0:
            print(f"    ... {i}/{n_residents} residents generated", flush=True)

    conn.commit()
    c.execute("VACUUM")
    conn.commit()
    conn.close()

    # ── Persist the latent ground truth beside the database ─────────────────
    gt_path = os.path.splitext(out_path)[0] + "_ground_truth.json.gz"
    import gzip
    with gzip.open(gt_path, "wt", encoding="utf-8") as f:
        json.dump({
            "dgp_version": DGP_VERSION, "seed": seed,
            "start_date": start_date.isoformat(), "days": days,
            "hazard": {"intercept": FALL_INTERCEPT, "beta_d": FALL_BETA_D,
                       "beta_frailty": FALL_BETA_FRAIL, "beta_recur": FALL_BETA_RECUR},
            "residents": truth,
        }, f)

    stats["_truth"] = truth
    stats["_ground_truth_path"] = gt_path
    stats["_meta"] = {
        "dgp_version": DGP_VERSION, "seed": seed, "n_residents": n_residents,
        "days": days, "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    return stats


# VALIDATION — does the synthetic cohort behave the way the model says it should?
#
# A generative model is only useful if its output is checked. Following the
# fidelity / utility / privacy taxonomy for medical synthetic data (Kaabachi et
# al., npj Digital Medicine 8:60, 2025), this function reports the FIDELITY and
# part of the UTILITY leg. The privacy leg is trivially satisfied and reported as
# such: no real record was used as input, so re-identification, membership
# inference and attribute disclosure risk are all structurally zero rather than
# empirically small.

def validate(db_path: str, truth: list[dict] | None = None) -> dict:
    """Compute fidelity statistics for a generated database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    def one(q, *a):
        r = c.execute(q, a).fetchone()
        return r[0] if r else None

    n_res  = one("SELECT COUNT(*) FROM residents WHERE active=1")
    d0, d1 = c.execute("SELECT MIN(date), MAX(date) FROM care_notes").fetchone()
    days   = (datetime.date.fromisoformat(d1) - datetime.date.fromisoformat(d0)).days + 1
    res_years = n_res * days / 365.25

    n_falls = one("SELECT COUNT(*) FROM incidents WHERE incident_type LIKE '%fall%'")
    n_notes = one("SELECT COUNT(*) FROM care_notes")
    n_mar   = one("SELECT COUNT(*) FROM mar_records")
    n_miss  = one("SELECT COUNT(*) FROM mar_records WHERE administered<>'Yes'")

    fluids = [r[0] for r in c.execute(
        "SELECT fluid_intake_ml FROM care_notes WHERE fluid_intake_ml IS NOT NULL")]
    wb = [r[0] for r in c.execute(
        "SELECT overall_score FROM wellbeing WHERE overall_score IS NOT NULL")]
    risk = [r[0] for r in c.execute(
        "SELECT score FROM risk_assessments WHERE score IS NOT NULL")]
    ages = [r[0] for r in c.execute("SELECT age FROM residents WHERE active=1")]

    # Falls per resident — dispersion tells us whether the between-subject random
    # effect actually produced heterogeneity, or whether every resident is alike.
    per_res = [r[0] for r in c.execute(
        "SELECT COUNT(*) FROM incidents WHERE incident_type LIKE '%fall%' "
        "GROUP BY resident_id")]
    per_res += [0] * (n_res - len(per_res))

    # Documentation-rate check for the MNAR mechanism: notes per resident-day
    # should be lower for residents with more falls (i.e. worse latent state).
    rows = c.execute("""
        SELECT r.resident_id,
               (SELECT COUNT(*) FROM care_notes n WHERE n.resident_id=r.resident_id) AS notes,
               (SELECT COUNT(*) FROM incidents i WHERE i.resident_id=r.resident_id
                 AND i.incident_type LIKE '%fall%') AS falls
        FROM residents r WHERE r.active=1""").fetchall()
    xs = [r["falls"] for r in rows]
    ys = [r["notes"] for r in rows]
    mnar_r = _pearson(xs, ys)

    out = {
        "cohort": {
            "n_residents": n_res, "days": days, "date_from": d0, "date_to": d1,
            "resident_years": round(res_years, 1),
        },
        "volume": {
            "care_notes": n_notes, "mar_records": n_mar,
            "wellbeing": one("SELECT COUNT(*) FROM wellbeing"),
            "risk_assessments": one("SELECT COUNT(*) FROM risk_assessments"),
            "incidents": one("SELECT COUNT(*) FROM incidents"),
            "falls": n_falls,
            "handovers": one("SELECT COUNT(*) FROM handovers"),
            "care_plans": one("SELECT COUNT(*) FROM care_plans"),
            "family_comms": one("SELECT COUNT(*) FROM family_comms"),
            "medications": one("SELECT COUNT(*) FROM medications"),
        },
        "fidelity": {
            "falls_per_resident_year": round(n_falls / res_years, 2),
            "external_benchmark": {
                "source": "Logan et al., FinCH cluster RCT, HTA 26(9), NIHR 2022 "
                          "(84 UK care homes, 1,657 residents)",
                "range_falls_per_resident_year": [2.2, 3.8],
            },
            "falls_per_resident": {
                "mean": round(statistics.mean(per_res), 2),
                "sd": round(statistics.pstdev(per_res), 2),
                "min": min(per_res), "max": max(per_res),
                "variance_to_mean_ratio": round(
                    statistics.pvariance(per_res) / statistics.mean(per_res), 2)
                if statistics.mean(per_res) else None,
            },
            "documentation_rate_per_resident_day": round(n_notes / (n_res * days), 3),
            "mnar_check_corr_falls_vs_notes": round(mnar_r, 3),
            "mar_missed_dose_rate": round(n_miss / n_mar, 4) if n_mar else None,
            "fluid_intake_ml": _dist(fluids),
            "wellbeing_score": _dist(wb),
            "falls_risk_score": _dist(risk),
            "age": _dist(ages),
        },
        "privacy": {
            "input_records_from_real_people": 0,
            "reidentification_risk": "structurally zero — no real record was an input "
                                     "to generation, so membership inference and "
                                     "attribute disclosure are undefined rather than small",
            "note": "Privacy leg of Kaabachi et al. (2025) taxonomy. Reported for "
                    "completeness; it is not an achievement of the method, it is a "
                    "consequence of simulating rather than transforming real data.",
        },
        "dgp": {
            "version": DGP_VERSION,
            "fall_hazard": f"logit h(t) = {FALL_INTERCEPT} + {FALL_BETA_D}·d(t) "
                           f"+ {FALL_BETA_FRAIL}·frailty + {FALL_BETA_RECUR}·min(recent_falls,3)",
            "coupled_observables": ["fluid_intake_ml", "wellbeing.overall_score",
                                    "mar_records.administered", "risk_assessments.score",
                                    "care-note frequency (MNAR)"],
        },
    }
    conn.close()

    if truth:
        fr = [t["frailty"] for t in truth]
        dm = [t["d_mean"] for t in truth]
        out["fidelity"]["latent"] = {
            "frailty": _dist(fr), "d_mean": _dist(dm),
            "corr_frailty_d_mean": round(_pearson(fr, dm), 3),
        }
    return out


def _dist(vals) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "n": len(s), "mean": round(statistics.mean(s), 2),
        "sd": round(statistics.pstdev(s), 2),
        "p05": round(s[int(0.05 * (len(s) - 1))], 2),
        "median": round(statistics.median(s), 2),
        "p95": round(s[int(0.95 * (len(s) - 1))], 2),
    }


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else 0.0


def print_validation(v: dict) -> None:
    BAR = "=" * 76
    print(BAR)
    print(" SYNTHETIC COHORT — VALIDATION REPORT")
    print(BAR)
    ch, vol, fid = v["cohort"], v["volume"], v["fidelity"]
    print(f" Cohort        : {ch['n_residents']} residents x {ch['days']} days "
          f"({ch['date_from']} to {ch['date_to']}) = {ch['resident_years']} resident-years")
    print()
    print(" Volume")
    for k, val in vol.items():
        print(f"   {k:<20} {val:>8,}")
    print()
    print(" Fidelity — external calibration")
    bm = fid["external_benchmark"]
    fpry = fid["falls_per_resident_year"]
    lo, hi = bm["range_falls_per_resident_year"]
    verdict = "WITHIN benchmark range" if lo <= fpry <= hi else "OUTSIDE benchmark range"
    print(f"   Falls per resident-year   {fpry}   benchmark {lo}-{hi}   -> {verdict}")
    print(f"   Source: {bm['source']}")
    print()
    print(" Fidelity — internal structure")
    fr = fid["falls_per_resident"]
    print(f"   Falls per resident        mean {fr['mean']}  sd {fr['sd']}  "
          f"range {fr['min']}-{fr['max']}  VMR {fr['variance_to_mean_ratio']}")
    print(f"     (VMR > 1 confirms over-dispersion, i.e. the between-subject random")
    print(f"      effect produced real heterogeneity rather than a Poisson cohort)")
    print(f"   Missed-dose rate          {fid['mar_missed_dose_rate']}")
    print(f"   Notes per resident-day    {fid['documentation_rate_per_resident_day']}")
    print(f"   MNAR check corr(falls, notes) = {fid['mnar_check_corr_falls_vs_notes']} "
          f"({'negative as designed' if fid['mnar_check_corr_falls_vs_notes'] < 0 else 'UNEXPECTED — should be negative'})")
    print()
    print(" Fidelity — marginal distributions")
    for key in ("fluid_intake_ml", "wellbeing_score", "falls_risk_score", "age"):
        dd = fid.get(key) or {}
        if dd:
            print(f"   {key:<20} mean {dd['mean']:>8}  sd {dd['sd']:>6}  "
                  f"p05 {dd['p05']:>7}  median {dd['median']:>7}  p95 {dd['p95']:>7}")
    if "latent" in fid:
        lt = fid["latent"]
        print(f"   corr(frailty, mean d(t)) = {lt['corr_frailty_d_mean']} "
              f"(positive by construction)")
    print()
    print(" Privacy")
    print(f"   Real records used as input: {v['privacy']['input_records_from_real_people']}")
    print(f"   {v['privacy']['reidentification_risk']}")
    print(BAR)



def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate the CareHome synthetic cohort (hierarchical latent "
                    "state-space model with a discrete-time survival outcome).")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "carehome.db"),
                    help="output database path (default: carehome.db beside this file)")
    ap.add_argument("--residents", type=int, default=N_RESIDENTS)
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--validate", action="store_true",
                    help="print the fidelity validation report after generating")
    ap.add_argument("--report", metavar="PATH",
                    help="write the validation report to PATH as JSON")
    ap.add_argument("--yes", action="store_true", help="do not prompt before overwriting")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.yes:
        resp = input(f"{args.out} exists and will be OVERWRITTEN. Continue? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    t0 = datetime.datetime.now()
    print(f"[*] Generating {args.residents} residents x {args.days} days "
          f"(seed {args.seed}) -> {args.out}")
    stats = generate(args.out, n_residents=args.residents, days=args.days,
                     seed=args.seed)
    truth = stats.pop("_truth")
    meta = stats.pop("_meta")
    gt_path = stats.pop("_ground_truth_path", None)
    print(f"[*] Done in {(datetime.datetime.now()-t0).total_seconds():.1f}s: "
          + ", ".join(f"{k}={v:,}" for k, v in stats.items()))
    if gt_path:
        print(f"[*] Latent ground truth (for the Bayes-ceiling benchmark): {gt_path}")

    if args.validate or args.report:
        v = validate(args.out, truth)
        v["generation"] = meta
        if args.validate:
            print()
            print_validation(v)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(v, f, indent=2)
            print(f"[*] Validation report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
