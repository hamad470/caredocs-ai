"""
CareDocs AI — Flask application.

The web layer for an MSc data science project on care home documentation. The
application exists to make three data science components inspectable rather than
to be a care records product, so the interface is deliberately narrow:

  Records      the care documentation the components are built on — care notes,
               incidents, handovers, wellbeing reviews, risk assessments, care
               plans, and the medication administration record.

  Component 1  a supervised fall-risk classifier. Fall Risk Ranking shows what
               it predicts for today; Model Evaluation shows how it was built,
               how it was tested and where it fails.

  Component 2  retrieval-augmented generation over the same records. Ask the
               Records is the interface, Knowledge Base is the index, and
               Retrieval Evaluation is the evidence that the answers are
               grounded in the file rather than invented.

  Narratives   a language model drafts the free-text summary on the care note,
               incident, handover, wellbeing and care plan forms. Every draft is
               editable and every one is attributed in the record.

Three places use a model, and only three: fall risk, retrieval, and narrative
drafting. Nothing else on any screen is generated. The medication administration
record in particular is a verbatim audit trail with no model in its path.

Roles:
  manager      — full access, sign-off, user management, model retraining
  senior_carer — approve notes and handovers, access all residents
  care_worker  — complete forms for their shift, submit for approval
  readonly     — view only

Run: python app.py
"""
import os
import secrets
import json
import time
from datetime import datetime, date, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_file, abort, g
)
import io

from database import get_db, init_db, hash_pw, DB_PATH
import ai_config                                      # load saved keys into env first
from ai_service import generate_narrative, get_ai_status, get_last_provider
import report_generator as rg
import xlsx_generator as xg
import rag_engine
import rag_evaluator
import rag_advanced          # hybrid BM25 + dense retrieval (chat assistant)
import analytics_tools       # parameterised SQL analytics tools
import chat_engine           # conversational RAG orchestration
import llm_client            # multi-provider chat completions

# ─── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_HTTPONLY"] = True

# Initialise DB on startup
with app.app_context():
    init_db()

ROLES = {
    "manager":      {"label": "Manager",      "color": "#7c3aed"},
    "senior_carer": {"label": "Senior Carer", "color": "#1d4ed8"},
    "care_worker":  {"label": "Care Worker",  "color": "#065f46"},
    "readonly":     {"label": "Read Only",    "color": "#78350f"},
}

# ─── Auth helpers ───────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            if session.get("role") not in roles:
                flash("You don't have permission to access that page.", "error")
                return redirect(url_for("dashboard"))
            return f(*args, **kwargs)
        return decorated
    return decorator


def can_edit():
    return session.get("role") in ("manager", "senior_carer", "care_worker")


def can_approve():
    return session.get("role") in ("manager", "senior_carer")


def can_manage():
    return session.get("role") == "manager"


@app.context_processor
def inject_globals():
    return {"current_date": date.today().strftime("%d %b %Y")}


@app.before_request
def load_logged_in_user():
    g.user = None
    if "user_id" in session:
        db = get_db()
        g.user = db.execute(
            "SELECT * FROM users WHERE id = ?", (session["user_id"],)
        ).fetchone()
        db.close()


def log_action(action, table_name=None, record_id=None, details=None):
    db = get_db()
    db.execute(
        "INSERT INTO audit_log (user_id,username,action,table_name,record_id,details,ip_address) "
        "VALUES (?,?,?,?,?,?,?)",
        (session.get("user_id"), session.get("username"), action,
         table_name, record_id, details, request.remote_addr)
    )
    db.commit()
    db.close()


# ─── Auth routes ───────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username=? AND active=1", (username,)
        ).fetchone()
        db.close()
        if user and user["password"] == hash_pw(password):
            session.clear()
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["full_name"]= user["full_name"]
            session["role"]     = user["role"]
            log_action("LOGIN")
            next_page = request.args.get("next", url_for("dashboard"))
            return redirect(next_page)
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    log_action("LOGOUT")
    session.clear()
    return redirect(url_for("login"))


# ─── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def dashboard():
    db = get_db()
    today = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    residents = db.execute(
        "SELECT * FROM residents WHERE active=1 ORDER BY room_number"
    ).fetchall()

    total_res    = len(residents)
    open_inc     = db.execute("SELECT COUNT(*) FROM incidents WHERE status='open'").fetchone()[0]
    pending_app  = db.execute(
        "SELECT COUNT(*) FROM care_notes WHERE status='pending_approval'"
    ).fetchone()[0]
    notes_week   = db.execute(
        "SELECT COUNT(*) FROM care_notes WHERE date>=?", (week_ago,)
    ).fetchone()[0]

    # Build resident cards with quick stats
    res_cards = []
    for r in residents:
        note_count = db.execute(
            "SELECT COUNT(*) FROM care_notes WHERE resident_id=? AND date>=?",
            (r["resident_id"], week_ago)
        ).fetchone()[0]
        open_inc_r = db.execute(
            "SELECT COUNT(*) FROM incidents WHERE resident_id=? AND status='open'",
            (r["resident_id"],)
        ).fetchone()[0]
        last_note = db.execute(
            "SELECT date FROM care_notes WHERE resident_id=? ORDER BY date DESC LIMIT 1",
            (r["resident_id"],)
        ).fetchone()
        # RAG status: red if note gap > 2 days or open incident
        if open_inc_r > 0:
            rag = "red"
        elif not last_note or last_note["date"] < (date.today()-timedelta(days=2)).isoformat():
            rag = "amber"
        else:
            rag = "green"
        res_cards.append({**dict(r), "note_count": note_count,
                          "open_incidents": open_inc_r, "rag": rag,
                          "last_note": last_note["date"] if last_note else "Never"})

    recent_notes = db.execute(
        "SELECT cn.*, r.preferred_name, r.room_number FROM care_notes cn "
        "JOIN residents r ON cn.resident_id=r.resident_id "
        "ORDER BY cn.created DESC LIMIT 8"
    ).fetchall()

    recent_incidents = db.execute(
        "SELECT i.*, r.preferred_name FROM incidents i "
        "JOIN residents r ON i.resident_id=r.resident_id "
        "WHERE i.status='open' ORDER BY i.date DESC LIMIT 5"
    ).fetchall()

    ai_status = get_ai_status()
    db.close()

    return render_template("dashboard.html",
        residents=res_cards, total_res=total_res,
        open_inc=open_inc, pending_app=pending_app,
        notes_week=notes_week, recent_notes=recent_notes,
        recent_incidents=recent_incidents, ai_status=ai_status,
        today=today
    )


# ─── Residents ─────────────────────────────────────────────────────────────────
@app.route("/residents")
@login_required
def residents_list():
    db = get_db()
    residents = db.execute("SELECT * FROM residents WHERE active=1 ORDER BY room_number").fetchall()
    db.close()
    return render_template("residents.html", residents=residents)


@app.route("/residents/<resident_id>")
@login_required
def resident_detail(resident_id):
    db = get_db()
    r = db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone()
    if not r:
        abort(404)
    notes = db.execute(
        "SELECT * FROM care_notes WHERE resident_id=? ORDER BY date DESC, created DESC LIMIT 20",
        (resident_id,)
    ).fetchall()
    incidents = db.execute(
        "SELECT * FROM incidents WHERE resident_id=? ORDER BY date DESC",
        (resident_id,)
    ).fetchall()
    meds = db.execute(
        "SELECT * FROM medications WHERE resident_id=? AND status='active' ORDER BY medication_name",
        (resident_id,)
    ).fetchall()
    wellbeing = db.execute(
        "SELECT * FROM wellbeing WHERE resident_id=? ORDER BY assessment_date DESC LIMIT 5",
        (resident_id,)
    ).fetchall()
    risks = db.execute(
        "SELECT * FROM risk_assessments WHERE resident_id=? ORDER BY date_assessed DESC",
        (resident_id,)
    ).fetchall()
    handovers = db.execute(
        "SELECT * FROM handovers WHERE resident_id=? ORDER BY date DESC LIMIT 10",
        (resident_id,)
    ).fetchall()
    care_plans = db.execute(
        "SELECT * FROM care_plans WHERE resident_id=? ORDER BY version DESC",
        (resident_id,)
    ).fetchall()
    family_comms_list = db.execute(
        "SELECT * FROM family_comms WHERE resident_id=? ORDER BY date DESC",
        (resident_id,)
    ).fetchall()
    db.close()
    return render_template("resident_detail.html", r=r, notes=notes,
                           incidents=incidents, meds=meds,
                           wellbeing=wellbeing, risks=risks, handovers=handovers,
                           care_plans=care_plans, family_comms_list=family_comms_list)


# ─── Care Notes ────────────────────────────────────────────────────────────────
@app.route("/care-notes/new", methods=["GET", "POST"])
@login_required
def care_note_new():
    if not can_edit():
        abort(403)
    db = get_db()
    residents = db.execute("SELECT * FROM residents WHERE active=1 ORDER BY full_name").fetchall()

    if request.method == "POST":
        f = request.form
        resident_id = f.get("resident_id")
        r = db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone()

        # AI narrative generation
        data = {
            "preferred_name":   r["preferred_name"] if r else "Resident",
            "shift":            f.get("shift",""),
            "personal_care":    f.get("personal_care",""),
            "mood":             f.get("mood",""),
            "appetite":         f.get("appetite",""),
            "fluid_intake_ml":  f.get("fluid_intake_ml",""),
            "activity_description": f.get("activity_description",""),
            "pain_observed":    f.get("pain_observed",""),
            "pain_location":    f.get("pain_location",""),
            "concerns":         f.get("concerns",""),
            "actions_taken":    f.get("actions_taken",""),
        }
        user_narrative = f.get("care_narrative","").strip()
        ai_used = False
        if not user_narrative and f.get("use_ai") == "yes":
            user_narrative, ai_used = generate_narrative("care_note", data)

        status = "pending_approval" if session.get("role") == "care_worker" else "approved"
        approved_by = session.get("full_name") if status == "approved" else None
        approved_at = datetime.now().isoformat() if status == "approved" else None

        db.execute("""INSERT INTO care_notes
            (resident_id,date,shift,staff_name,staff_role,note_type,
             personal_care,mood,appetite,fluid_intake_ml,weight_kg,
             skin_checked,skin_concern,repositioned,activity,activity_description,
             sleep_quality,pain_observed,pain_location,falls_this_shift,
             care_narrative,concerns,actions_taken,handover_notes,
             ai_generated,status,approved_by,approved_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (resident_id, f.get("date",date.today().isoformat()),
             f.get("shift"), session.get("full_name"), session.get("role","").replace("_"," ").title(),
             f.get("note_type","Daily Care"),
             f.get("personal_care"), f.get("mood"), f.get("appetite"),
             f.get("fluid_intake_ml") or None, f.get("weight_kg") or None,
             f.get("skin_checked","No"), f.get("skin_concern",""),
             f.get("repositioned","No"), f.get("activity","No"),
             f.get("activity_description",""),
             f.get("sleep_quality",""), f.get("pain_observed","No"),
             f.get("pain_location",""), int(f.get("falls_this_shift",0) or 0),
             user_narrative, f.get("concerns",""), f.get("actions_taken",""),
             f.get("handover_notes",""),
             1 if ai_used else 0, status, approved_by, approved_at)
        )
        db.commit()
        log_action("CREATE_CARE_NOTE", "care_notes", details=f"Resident: {resident_id}")
        db.close()
        flash("Care note saved successfully.", "success")
        return redirect(url_for("resident_detail", resident_id=resident_id))

    db.close()
    return render_template("care_note_form.html", residents=residents,
                           today=date.today().isoformat(), can_approve=can_approve())


@app.route("/care-notes/<int:note_id>")
@login_required
def care_note_view(note_id):
    db = get_db()
    note = db.execute("SELECT * FROM care_notes WHERE id=?", (note_id,)).fetchone()
    if not note:
        abort(404)
    r = db.execute("SELECT * FROM residents WHERE resident_id=?", (note["resident_id"],)).fetchone()
    db.close()
    return render_template("care_note_view.html", note=note, r=r)


@app.route("/care-notes/<int:note_id>/approve", methods=["POST"])
@login_required
def care_note_approve(note_id):
    if not can_approve():
        abort(403)
    db = get_db()
    db.execute(
        "UPDATE care_notes SET status='approved', approved_by=?, approved_at=? WHERE id=?",
        (session.get("full_name"), datetime.now().isoformat(), note_id)
    )
    db.commit()
    log_action("APPROVE_CARE_NOTE", "care_notes", note_id)
    note = db.execute("SELECT resident_id FROM care_notes WHERE id=?", (note_id,)).fetchone()
    db.close()
    flash("Care note approved.", "success")
    return redirect(url_for("resident_detail", resident_id=note["resident_id"]))


@app.route("/care-notes/<int:note_id>/pdf")
@login_required
def care_note_pdf(note_id):
    if not rg.report_available():
        flash("PDF generation requires the 'reportlab' package.", "error")
        return redirect(request.referrer or url_for("dashboard"))
    db = get_db()
    note = db.execute("SELECT * FROM care_notes WHERE id=?", (note_id,)).fetchone()
    r    = db.execute("SELECT * FROM residents WHERE resident_id=?", (note["resident_id"],)).fetchone()
    db.close()
    pdf_bytes = rg.generate_care_note_pdf(dict(note), dict(r))
    log_action("EXPORT_PDF", "care_notes", note_id)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"care_note_{r['preferred_name']}_{note['date']}.pdf"
    )


# ─── Incidents ─────────────────────────────────────────────────────────────────
@app.route("/incidents")
@login_required
def incidents_list():
    db = get_db()
    incidents = db.execute(
        "SELECT i.*, r.preferred_name, r.room_number FROM incidents i "
        "JOIN residents r ON i.resident_id=r.resident_id ORDER BY i.date DESC"
    ).fetchall()
    db.close()
    return render_template("incidents.html", incidents=incidents)


@app.route("/incidents/new", methods=["GET","POST"])
@login_required
def incident_new():
    if not can_edit():
        abort(403)
    db = get_db()
    residents = db.execute("SELECT * FROM residents WHERE active=1 ORDER BY full_name").fetchall()

    if request.method == "POST":
        f = request.form
        resident_id = f.get("resident_id")
        r = db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone()

        # Generate incident ID
        count = db.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        inc_id = f"INC-{date.today().year}-{str(count+1).zfill(3)}"

        data = {
            "preferred_name":   r["preferred_name"] if r else "Resident",
            "incident_type":    f.get("incident_type",""),
            "severity":         f.get("severity",""),
            "time":             f.get("time",""),
            "location":         f.get("location",""),
            "witness_name":     f.get("witness_name",""),
            "description":      f.get("description",""),
            "injuries":         f.get("injuries",""),
            "immediate_actions":f.get("immediate_actions",""),
        }
        ai_used = False
        desc = f.get("description","").strip()
        if not desc and f.get("use_ai") == "yes":
            desc, ai_used = generate_narrative("incident", data)

        db.execute("""INSERT INTO incidents
            (incident_id,resident_id,date,time,shift,incident_type,severity,location,
             witnessed,witness_name,staff_first_on_scene,description,immediate_actions,
             injuries,medical_attention,outcome,gp_notified,family_notified,
             cqc_notification,risk_assessment_updated,care_plan_updated,
             investigation_required,lessons_learned,preventative_actions,
             ai_generated,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (inc_id, resident_id,
             f.get("date", date.today().isoformat()),
             f.get("time"), f.get("shift"),
             f.get("incident_type"), f.get("severity"),
             f.get("location"), f.get("witnessed","No"),
             f.get("witness_name",""), session.get("full_name"),
             desc, f.get("immediate_actions",""),
             f.get("injuries",""), f.get("medical_attention","No"),
             f.get("outcome",""), f.get("gp_notified","No"),
             f.get("family_notified","No"), f.get("cqc_notification","No"),
             f.get("risk_assessment_updated","No"), f.get("care_plan_updated","No"),
             f.get("investigation_required","No"),
             f.get("lessons_learned",""), f.get("preventative_actions",""),
             1 if ai_used else 0, "open")
        )
        db.commit()
        log_action("CREATE_INCIDENT", "incidents", details=f"Resident: {resident_id} Type: {f.get('incident_type')}")

        severity = f.get("severity", "")
        db.close()
        if severity in ("Major", "Critical"):
            flash(f"Incident {inc_id} recorded — severity {severity}. "
                  f"Escalate through the home's normal channel.", "warning")
        else:
            flash(f"Incident {inc_id} recorded.", "success")
        return redirect(url_for("incidents_list"))

    db.close()
    return render_template("incident_form.html", residents=residents,
                           today=date.today().isoformat(),
                           now=datetime.now().strftime("%H:%M"))


@app.route("/incidents/<int:inc_id>")
@login_required
def incident_view(inc_id):
    db = get_db()
    incident = db.execute("SELECT * FROM incidents WHERE id=?", (inc_id,)).fetchone()
    if not incident:
        abort(404)
    r = db.execute("SELECT * FROM residents WHERE resident_id=?", (incident["resident_id"],)).fetchone()
    db.close()
    return render_template("incident_view.html", incident=incident, r=r)


@app.route("/incidents/<int:inc_id>/close", methods=["POST"])
@login_required
def incident_close(inc_id):
    if not can_approve():
        abort(403)
    db = get_db()
    db.execute(
        "UPDATE incidents SET status='closed', manager_sign_off=?, manager_sign_off_date=? WHERE id=?",
        (session.get("full_name"), date.today().isoformat(), inc_id)
    )
    db.commit()
    log_action("CLOSE_INCIDENT", "incidents", inc_id)
    db.close()
    flash("Incident closed and signed off.", "success")
    return redirect(url_for("incidents_list"))


@app.route("/incidents/<int:inc_id>/pdf")
@login_required
def incident_pdf(inc_id):
    if not rg.report_available():
        flash("PDF generation requires the 'reportlab' package.", "error")
        return redirect(request.referrer or url_for("dashboard"))
    db = get_db()
    incident = db.execute("SELECT * FROM incidents WHERE id=?", (inc_id,)).fetchone()
    r = db.execute("SELECT * FROM residents WHERE resident_id=?", (incident["resident_id"],)).fetchone()
    db.close()
    pdf_bytes = rg.generate_incident_pdf(dict(incident), dict(r))
    log_action("EXPORT_PDF", "incidents", inc_id)
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True,
        download_name=f"incident_{incident['incident_id']}.pdf"
    )


# ─── Handovers ─────────────────────────────────────────────────────────────────
@app.route("/handovers")
@login_required
def handovers_list():
    db = get_db()
    handovers = db.execute(
        "SELECT h.*, r.preferred_name, r.room_number FROM handovers h "
        "JOIN residents r ON h.resident_id=r.resident_id ORDER BY h.date DESC, h.created DESC LIMIT 50"
    ).fetchall()
    db.close()
    return render_template("handovers.html", handovers=handovers)


@app.route("/handovers/new", methods=["GET","POST"])
@login_required
def handover_new():
    if not can_edit():
        abort(403)
    db = get_db()
    residents = db.execute("SELECT * FROM residents WHERE active=1 ORDER BY room_number").fetchall()

    if request.method == "POST":
        f = request.form
        resident_id = f.get("resident_id")
        r = db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone()

        data = {
            "preferred_name":     r["preferred_name"] if r else "Resident",
            "shift_ending":       f.get("shift_ending",""),
            "care_completed":     f.get("care_completed",""),
            "concerns_next_shift":f.get("concerns_next_shift",""),
            "outstanding_tasks":  f.get("outstanding_tasks",""),
            "fluid_target_met":   f.get("fluid_target_met",""),
            "incidents_this_shift":f.get("incidents_this_shift",0),
        }
        summary = f.get("overall_summary","").strip()
        ai_used = False
        if not summary and f.get("use_ai") == "yes":
            summary, ai_used = generate_narrative("handover", data)

        db.execute("""INSERT INTO handovers
            (date,shift_ending,shift_starting,compiled_by,resident_id,
             overall_summary,care_completed,concerns_next_shift,outstanding_tasks,
             medication_notes,fluid_target_met,incidents_this_shift,incident_reference,
             family_contact,family_contact_notes,escalation_required,escalation_details,
             ai_generated,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f.get("date", date.today().isoformat()),
             f.get("shift_ending"), f.get("shift_starting"),
             session.get("full_name"), resident_id,
             summary, f.get("care_completed",""),
             f.get("concerns_next_shift",""), f.get("outstanding_tasks",""),
             f.get("medication_notes",""), f.get("fluid_target_met","No"),
             int(f.get("incidents_this_shift",0) or 0), f.get("incident_reference",""),
             f.get("family_contact","No"), f.get("family_contact_notes",""),
             f.get("escalation_required","No"), f.get("escalation_details",""),
             1 if ai_used else 0, "draft")
        )
        db.commit()
        log_action("CREATE_HANDOVER", "handovers", details=f"Resident: {resident_id}")
        db.close()
        flash("Handover recorded.", "success")
        return redirect(url_for("resident_detail", resident_id=resident_id))

    db.close()
    return render_template("handover_form.html", residents=residents,
                           today=date.today().isoformat())


@app.route("/handovers/<int:hid>/pdf")
@login_required
def handover_pdf(hid):
    if not rg.report_available():
        flash("PDF generation requires the 'reportlab' package.", "error")
        return redirect(request.referrer or url_for("dashboard"))
    db = get_db()
    h = db.execute("SELECT * FROM handovers WHERE id=?", (hid,)).fetchone()
    r = db.execute("SELECT * FROM residents WHERE resident_id=?", (h["resident_id"],)).fetchone()
    db.close()
    pdf_bytes = rg.generate_handover_pdf(dict(h), dict(r))
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True,
        download_name=f"handover_{r['preferred_name']}_{h['date']}.pdf"
    )


# ─── Medications ────────────────────────────────────────────────────────────────
@app.route("/medications")
@login_required
def medications_list():
    db = get_db()
    meds = db.execute(
        "SELECT m.*, r.preferred_name, r.room_number FROM medications m "
        "JOIN residents r ON m.resident_id=r.resident_id "
        "WHERE m.status='active' ORDER BY r.room_number, m.medication_name"
    ).fetchall()
    db.close()
    return render_template("medications.html", meds=meds)


@app.route("/medications/new", methods=["GET","POST"])
@login_required
def medication_new():
    if not can_approve():
        abort(403)
    db = get_db()
    residents = db.execute("SELECT * FROM residents WHERE active=1 ORDER BY full_name").fetchall()

    if request.method == "POST":
        f = request.form
        db.execute("""INSERT INTO medications
            (resident_id,medication_name,generic_name,dose,route,frequency,
             indication,prescribing_gp,start_date,review_date,is_controlled,
             is_prn,prn_instructions,admin_notes,side_effects,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'active')""",
            (f.get("resident_id"), f.get("medication_name"), f.get("generic_name",""),
             f.get("dose"), f.get("route"), f.get("frequency"),
             f.get("indication",""), f.get("prescribing_gp",""),
             f.get("start_date",""), f.get("review_date",""),
             1 if f.get("is_controlled") else 0,
             1 if f.get("is_prn") else 0,
             f.get("prn_instructions",""), f.get("admin_notes",""), f.get("side_effects",""))
        )
        db.commit()
        log_action("CREATE_MEDICATION", "medications")
        db.close()
        flash("Medication added.", "success")
        return redirect(url_for("medications_list"))

    db.close()
    return render_template("medication_form.html", residents=residents,
                           today=date.today().isoformat())


@app.route("/medications/<int:mid>/mar", methods=["GET","POST"])
@login_required
def mar_record(mid):
    """Medication Administration Record entry."""
    if not can_edit():
        abort(403)
    db = get_db()
    med = db.execute(
        "SELECT m.*, r.preferred_name FROM medications m "
        "JOIN residents r ON m.resident_id=r.resident_id WHERE m.id=?", (mid,)
    ).fetchone()
    if not med:
        abort(404)

    if request.method == "POST":
        f = request.form
        db.execute("""INSERT INTO mar_records
            (medication_id,resident_id,date,time_given,shift,given_by,
             administered,refusal_reason,notes)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (mid, med["resident_id"],
             f.get("date", date.today().isoformat()),
             f.get("time_given"), f.get("shift"),
             session.get("full_name"),
             f.get("administered"), f.get("refusal_reason",""), f.get("notes",""))
        )
        db.commit()
        log_action("MAR_ENTRY", "mar_records", details=f"Med: {med['medication_name']}")
        db.close()
        flash("MAR entry recorded.", "success")
        return redirect(url_for("medications_list"))

    recent = db.execute(
        "SELECT * FROM mar_records WHERE medication_id=? ORDER BY date DESC, created DESC LIMIT 7",
        (mid,)
    ).fetchall()
    db.close()
    return render_template("mar_form.html", med=med, recent=recent,
                           today=date.today().isoformat(),
                           now=datetime.now().strftime("%H:%M"))


# ─── Wellbeing ──────────────────────────────────────────────────────────────────
@app.route("/wellbeing/new", methods=["GET","POST"])
@login_required
def wellbeing_new():
    if not can_edit():
        abort(403)
    db = get_db()
    residents = db.execute("SELECT * FROM residents WHERE active=1 ORDER BY full_name").fetchall()

    if request.method == "POST":
        f = request.form
        resident_id = f.get("resident_id")
        r = db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone()

        overall = _wellbeing_overall(f)

        # Same payload the Generate button sends, so a summary written on save
        # is identical to one previewed in the browser. Two code paths building
        # two different prompts is how the two quietly drift apart.
        data = {
            "preferred_name":    r["preferred_name"] if r else "Resident",
            "resident_id":       resident_id,
            "period_covered":    f.get("period_covered", ""),
            "overall_score":     overall,
            "physical_notes":    f.get("physical_notes", ""),
            "mental_notes":      f.get("mental_notes", ""),
            "social_notes":      f.get("social_notes", ""),
            "goals_progress":    f.get("goals_progress", ""),
            "concerns":          f.get("concerns", ""),
            "positive_outcomes": f.get("positive_outcomes", ""),
            "actions_next_period": f.get("actions_next_period", ""),
            "resident_voice":    f.get("resident_voice", ""),
            "family_feedback":   f.get("family_feedback", ""),
        }
        for field in WELLBEING_SCORE_FIELDS:
            data[field] = f.get(field)
        prior = _wellbeing_previous(db, resident_id)
        if prior:
            data.update(prior)

        summary = f.get("summary", "").strip()
        ai_used = False
        if not summary and f.get("use_ai") == "yes":
            summary, ai_used = generate_narrative("wellbeing", data)

        db.execute("""INSERT INTO wellbeing
            (resident_id,assessment_date,assessed_by,period_covered,
             physical_health_score,mental_health_score,social_engagement_score,
             personal_care_score,nutrition_score,pain_management_score,overall_score,
             physical_notes,mental_notes,social_notes,goals_progress,concerns,
             positive_outcomes,actions_next_period,family_feedback,resident_voice,
             summary,ai_generated,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (resident_id, f.get("assessment_date", date.today().isoformat()),
             session.get("full_name"), f.get("period_covered",""),
             f.get("physical_health_score") or None, f.get("mental_health_score") or None,
             f.get("social_engagement_score") or None, f.get("personal_care_score") or None,
             f.get("nutrition_score") or None, f.get("pain_management_score") or None,
             overall,
             f.get("physical_notes",""), f.get("mental_notes",""),
             f.get("social_notes",""), f.get("goals_progress",""),
             f.get("concerns",""), f.get("positive_outcomes",""),
             f.get("actions_next_period",""), f.get("family_feedback",""),
             f.get("resident_voice",""), summary,
             1 if ai_used else 0, "draft")
        )
        db.commit()
        log_action("CREATE_WELLBEING", "wellbeing", details=f"Resident: {resident_id}")
        db.close()
        flash("Wellbeing assessment saved.", "success")
        return redirect(url_for("resident_detail", resident_id=resident_id))

    db.close()
    return render_template("wellbeing_form.html", residents=residents,
                           today=date.today().isoformat())


@app.route("/wellbeing/<int:wid>/pdf")
@login_required
def wellbeing_pdf(wid):
    if not rg.report_available():
        flash("PDF generation requires the 'reportlab' package.", "error")
        return redirect(request.referrer or url_for("dashboard"))
    db = get_db()
    w = db.execute("SELECT * FROM wellbeing WHERE id=?", (wid,)).fetchone()
    r = db.execute("SELECT * FROM residents WHERE resident_id=?", (w["resident_id"],)).fetchone()
    db.close()
    pdf_bytes = rg.generate_wellbeing_pdf(dict(w), dict(r))
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True,
        download_name=f"wellbeing_{r['preferred_name']}_{w['assessment_date']}.pdf"
    )


# ─── Risk Assessments ──────────────────────────────────────────────────────────
@app.route("/risks/new", methods=["GET","POST"])
@login_required
def risk_new():
    if not can_edit():
        abort(403)
    db = get_db()
    residents = db.execute("SELECT * FROM residents WHERE active=1 ORDER BY full_name").fetchall()

    if request.method == "POST":
        f = request.form
        resident_id = f.get("resident_id")
        r = db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone()

        data = {
            "preferred_name":   r["preferred_name"] if r else "Resident",
            "assessment_type":  f.get("assessment_type",""),
            "risk_level":       f.get("risk_level",""),
            "score":            f.get("score",""),
            "risk_factors":     f.get("risk_factors",""),
            "interventions":    f.get("interventions",""),
        }
        narrative = f.get("narrative","").strip()
        ai_used = False
        if not narrative and f.get("use_ai") == "yes":
            narrative, ai_used = generate_narrative("risk", data)

        db.execute("""INSERT INTO risk_assessments
            (resident_id,assessment_type,date_assessed,assessed_by,review_date,
             score,risk_level,risk_factors,interventions,additional_actions,
             outcome_measures,narrative,ai_generated,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (resident_id, f.get("assessment_type"),
             f.get("date_assessed", date.today().isoformat()),
             session.get("full_name"), f.get("review_date",""),
             f.get("score") or None, f.get("risk_level",""),
             f.get("risk_factors",""), f.get("interventions",""),
             f.get("additional_actions",""), f.get("outcome_measures",""),
             narrative, 1 if ai_used else 0, "draft")
        )
        db.commit()
        log_action("CREATE_RISK", "risk_assessments", details=f"Resident: {resident_id}")

        risk_level = f.get("risk_level", "")
        db.close()
        if risk_level in ("Very High", "High"):
            flash(f"Risk assessment saved — {risk_level} risk. "
                  f"Escalate through the home's normal channel.", "warning")
        else:
            flash("Risk assessment saved.", "success")
        return redirect(url_for("resident_detail", resident_id=resident_id))

    db.close()
    return render_template("risk_form.html", residents=residents,
                           today=date.today().isoformat())


# ─── Compliance Dashboard ───────────────────────────────────────────────────────
# ─── User Management ────────────────────────────────────────────────────────────
@app.route("/users")
@role_required("manager")
def users_list():
    db = get_db()
    users = db.execute("SELECT * FROM users ORDER BY role, full_name").fetchall()
    db.close()
    return render_template("users.html", users=users, roles=ROLES)


@app.route("/users/new", methods=["GET","POST"])
@role_required("manager")
def user_new():
    if request.method == "POST":
        f = request.form
        try:
            db = get_db()
            db.execute(
                "INSERT INTO users (username,password,full_name,role) VALUES (?,?,?,?)",
                (f.get("username"), hash_pw(f.get("password","")),
                 f.get("full_name"), f.get("role"))
            )
            db.commit()
            log_action("CREATE_USER", "users", details=f.get("username"))
            db.close()
            flash(f"User {f.get('username')} created.", "success")
            return redirect(url_for("users_list"))
        except Exception as e:
            flash(f"Error: {e}", "error")
    return render_template("user_form.html", roles=ROLES)


@app.route("/users/<int:uid>/toggle", methods=["POST"])
@role_required("manager")
def user_toggle(uid):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if user:
        db.execute("UPDATE users SET active=? WHERE id=?", (0 if user["active"] else 1, uid))
        db.commit()
        log_action("TOGGLE_USER", "users", uid)
    db.close()
    flash("User status updated.", "success")
    return redirect(url_for("users_list"))


# ─── ML Models ──────────────────────────────────────────────────────────────────
@app.route("/risk-ranking")
@login_required
def risk_ranking():
    """
    Two scores side by side, from two different methods:
      * composite_score — rule-based, hand-weighted clinical policy
      * ml_probability  — supervised classifier, loaded from the saved artefact

    compute_daily_risk_ranking() calls predict_fall_risk() internally, which
    reads ml_fall_risk_model.pkl. If no artefact is present the ML column is
    simply absent and the rule-based ranking still works — the page degrades,
    it does not fail.
    """
    import ml_models
    ranking = ml_models.compute_daily_risk_ranking()
    cusum   = ml_models.cusum_adherence_anomalies()
    cusum_count = sum(len(v) for v in cusum.values())
    return render_template("risk_ranking.html",
        ranking=ranking,
        cusum_alerts=cusum,
        cusum_count=cusum_count,
        model_status=ml_models.get_model_status(),
        today=date.today().strftime("%d %B %Y")
    )


@app.route("/ml-evaluation")
@role_required("manager", "senior_carer")
def ml_evaluation():
    """
    SERVES the persisted model artefact — it does not train.

    The metrics rendered here are the ones recorded when the deployed model was
    fitted, read back out of ml_fall_risk_model.pkl. That guarantees the page
    describes the model that is actually making predictions, and keeps the
    request fast (a disk read, not a cross-validation run). Training is an
    explicit action: the Retrain button below, or `python train_model.py`.
    """
    import ml_models
    status  = ml_models.get_model_status()
    results = ml_models.get_cached_results()

    if results is None:
        # First run on a fresh checkout: bootstrap an artefact once so the page
        # is never empty, then tell the user what happened.
        results = ml_models.train_fall_risk_model(save=True)
        status  = ml_models.get_model_status()
        if not results.get("error"):
            flash("No saved model was found, so one was trained and saved now. "
                  "Future visits will load it from disk.", "info")

    # Deliberately no call to compute_daily_risk_ranking() here. This page is
    # about how the model was built and tested, not about today's scores, and
    # scoring the whole cohort costs a full feature-extraction pass — two
    # minutes of work for a figure the page never renders.

    # Cohort-size experiment, produced offline by experiments/learning_curve.py.
    # It is read, never run here: a page load must not fit hundreds of models.
    here = os.path.dirname(os.path.abspath(__file__))
    def _read(name):
        try:
            with open(os.path.join(here, "results", name), "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    learning_curve = _read("learning_curve.json")
    validation     = _read("synthetic_data_validation.json")

    return render_template("ml_evaluation.html",
        results=results,
        model_status=status,
        learning_curve=learning_curve,
        validation=validation,
        j_results=json.dumps(results, default=str),
        j_curve=json.dumps(learning_curve) if learning_curve else "null",
        today=date.today().strftime("%d %B %Y")
    )


@app.route("/ml-evaluation/retrain", methods=["POST"])
@role_required("manager")
def ml_retrain():
    """
    Explicit retrain — the only path in the web app that fits a model.

    Restricted to managers and POST-only (so it cannot be triggered by a link,
    a crawler or a page refresh), because retraining overwrites the artefact
    that is producing live clinical risk scores.
    """
    import ml_models
    t0 = time.time()
    results = ml_models.train_fall_risk_model(save=True)
    elapsed = round(time.time() - t0, 1)

    if results.get("error"):
        flash(f"Retraining failed: {results['error']}", "danger")
    else:
        status = ml_models.get_model_status()
        sel = (results.get("model_selection") or {}).get("selected", "model")
        flash(
            f"Model retrained and saved in {elapsed}s — {sel.replace('_', ' ')} "
            f"selected on {status.get('n_samples')} windows "
            f"({status.get('n_positive')} positive). "
            f"Artefact: {status.get('filename')}.",
            "success"
        )
        log_action("RETRAIN_MODEL", "ml_fall_risk_model", None)

    return redirect(url_for("ml_evaluation"))


# ─── AI Settings ────────────────────────────────────────────────────────────────
@app.route("/ai-settings", methods=["GET", "POST"])
@role_required("manager")
def ai_settings():
    """
    Manage the Gemini keys.

    Only one backend exists, so this page is a key manager rather than a
    provider chooser. The single remaining choice is whether narrative fields
    call the API at all: "template" pins the deterministic offline path, which
    the retrieval evaluation uses as its control arm and which every screen
    falls back to when no key answers.
    """
    stored = ai_config.get_stored_keys()

    if request.method == "POST":
        action = request.form.get("action", "save")

        if action == "clear":
            ai_config.save_keys(gemini_keys=[], provider="auto")
            flash("Gemini keys cleared. Narrative fields will use the offline template.",
                  "success")
            return redirect(url_for("ai_settings"))

        stored_now = ai_config.get_stored_keys()
        stored_gemini = stored_now.get("gemini_keys") or []

        # A field still showing its masked placeholder keeps whatever was saved
        # in that slot, so re-saving the form to change one setting never wipes
        # a key the user cannot see.
        #
        # Resolving a mask back to its key needs care. Matching by mask value
        # alone breaks when two keys mask identically — Gemini keys share a long
        # prefix, so this is not hypothetical — and the second key is silently
        # lost as a duplicate. Matching by slot position alone breaks when slots
        # have compacted after a deletion. So each stored key is consumed at
        # most once: the slot's own key is preferred, and only if that has
        # already been taken does the search widen to any unconsumed key with
        # the same mask.
        remaining = list(stored_gemini)
        gemini_list = []
        for i in range(ai_config.MAX_GEMINI_KEYS):
            raw = request.form.get(f"gemini_key_{i + 1}", "").strip()
            if not raw:
                continue
            if "..." in raw or "\u2026" in raw:
                resolved = None
                if i < len(stored_gemini) and stored_gemini[i] in remaining \
                        and ai_config.mask(stored_gemini[i]) == raw:
                    resolved = stored_gemini[i]
                if resolved is None:
                    resolved = next((k for k in remaining
                                     if ai_config.mask(k) == raw), None)
                if resolved is None and i < len(stored_gemini):
                    resolved = stored_gemini[i] if stored_gemini[i] in remaining else None
                if resolved:
                    remaining.remove(resolved)
                    gemini_list.append(resolved)
            else:
                gemini_list.append(raw)

        # A form posted by an older single-field template still works.
        legacy = request.form.get("gemini_key", "").strip()
        if not gemini_list and legacy and "..." not in legacy:
            gemini_list = [legacy]

        gemini_list = ai_config.normalise_gemini_keys(gemini_list)
        prov = request.form.get("provider", "auto").strip().lower()
        if prov not in ("auto", "template"):
            prov = "auto"

        ai_config.save_keys(gemini_keys=gemini_list, provider=prov)

        status_now = get_ai_status()
        if gemini_list:
            flash(f"Saved {len(gemini_list)} Gemini key"
                  f"{'' if len(gemini_list) == 1 else 's'}. "
                  f"Active: {status_now['labels'][status_now['active']]}. No restart needed.",
                  "success")
        else:
            flash("No keys saved — narrative fields will use the offline template. "
                  "That is a supported mode, not a failure.", "info")

        return redirect(url_for("ai_settings"))

    # Per-key health is best-effort: the settings page must render even if the
    # client module cannot be imported or has never been called.
    try:
        import llm_client
        key_report = llm_client.gemini_key_report()
    except Exception:
        key_report = []

    return render_template("ai_settings.html",
        status=get_ai_status(),
        provider=stored.get("provider", "auto"),
        gemini_keys=stored.get("gemini_keys") or [],
        max_gemini_keys=ai_config.MAX_GEMINI_KEYS,
        key_report=key_report,
        mask_key=ai_config.mask,
    )


# ─── XLSX Exports ───────────────────────────────────────────────────────────────

def _xlsx_response(data: bytes, filename: str):
    return send_file(
        io.BytesIO(data),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename
    )


def _get_resident(db, resident_id):
    return dict(db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone() or {})


@app.route("/care-notes/<int:note_id>/xlsx")
@login_required
def care_note_xlsx(note_id):
    db = get_db()
    note = db.execute("SELECT * FROM care_notes WHERE id=?", (note_id,)).fetchone()
    if not note:
        abort(404)
    r = _get_resident(db, note["resident_id"])
    notes = [dict(note)]
    db.close()
    log_action("EXPORT_XLSX", "care_notes", note_id)
    return _xlsx_response(
        xg.generate_care_note_xlsx(notes, r),
        f"care_note_{r.get('preferred_name','res')}_{note['date']}.xlsx"
    )


@app.route("/residents/<resident_id>/care-notes/xlsx")
@login_required
def resident_care_notes_xlsx(resident_id):
    """Export ALL care notes for a resident as one XLSX."""
    db = get_db()
    r = _get_resident(db, resident_id)
    notes = [dict(n) for n in db.execute(
        "SELECT * FROM care_notes WHERE resident_id=? ORDER BY date DESC, created DESC",
        (resident_id,)
    ).fetchall()]
    db.close()
    log_action("EXPORT_XLSX", "care_notes", details=f"All notes for {resident_id}")
    return _xlsx_response(
        xg.generate_care_note_xlsx(notes, r),
        f"care_notes_{r.get('preferred_name','res')}_all.xlsx"
    )


@app.route("/residents/<resident_id>/incidents/xlsx")
@login_required
def resident_incidents_xlsx(resident_id):
    db = get_db()
    r = _get_resident(db, resident_id)
    incidents = [dict(i) for i in db.execute(
        "SELECT * FROM incidents WHERE resident_id=? ORDER BY date DESC", (resident_id,)
    ).fetchall()]
    db.close()
    return _xlsx_response(
        xg.generate_incident_xlsx(incidents, {}),
        f"incidents_{r.get('preferred_name','res')}.xlsx"
    )


@app.route("/incidents/xlsx")
@login_required
def all_incidents_xlsx():
    """All incidents across all residents."""
    db = get_db()
    incidents = [dict(i) for i in db.execute(
        "SELECT * FROM incidents ORDER BY date DESC"
    ).fetchall()]
    db.close()
    log_action("EXPORT_XLSX", "incidents", details="All incidents")
    return _xlsx_response(
        xg.generate_incident_xlsx(incidents, {}),
        f"all_incidents_{date.today().isoformat()}.xlsx"
    )


@app.route("/residents/<resident_id>/handovers/xlsx")
@login_required
def resident_handovers_xlsx(resident_id):
    db = get_db()
    r = _get_resident(db, resident_id)
    handovers = [dict(h) for h in db.execute(
        "SELECT * FROM handovers WHERE resident_id=? ORDER BY date DESC", (resident_id,)
    ).fetchall()]
    db.close()
    return _xlsx_response(
        xg.generate_handover_xlsx(handovers, r),
        f"handovers_{r.get('preferred_name','res')}.xlsx"
    )


@app.route("/residents/<resident_id>/wellbeing/xlsx")
@login_required
def resident_wellbeing_xlsx(resident_id):
    db = get_db()
    r = _get_resident(db, resident_id)
    assessments = [dict(a) for a in db.execute(
        "SELECT * FROM wellbeing WHERE resident_id=? ORDER BY assessment_date DESC", (resident_id,)
    ).fetchall()]
    db.close()
    return _xlsx_response(
        xg.generate_wellbeing_xlsx(assessments, r),
        f"wellbeing_{r.get('preferred_name','res')}.xlsx"
    )


@app.route("/residents/<resident_id>/risks/xlsx")
@login_required
def resident_risks_xlsx(resident_id):
    db = get_db()
    r = _get_resident(db, resident_id)
    risks = [dict(x) for x in db.execute(
        "SELECT * FROM risk_assessments WHERE resident_id=? ORDER BY date_assessed DESC", (resident_id,)
    ).fetchall()]
    db.close()
    return _xlsx_response(
        xg.generate_risk_xlsx(risks, r),
        f"risk_assessments_{r.get('preferred_name','res')}.xlsx"
    )


@app.route("/residents/<resident_id>/medications/xlsx")
@login_required
def resident_medications_xlsx(resident_id):
    db = get_db()
    r = _get_resident(db, resident_id)
    meds = [dict(m) for m in db.execute(
        "SELECT * FROM medications WHERE resident_id=? ORDER BY medication_name", (resident_id,)
    ).fetchall()]
    mar = db.execute(
        "SELECT mr.*, m.medication_name FROM mar_records mr "
        "JOIN medications m ON mr.medication_id=m.id "
        "WHERE mr.resident_id=? ORDER BY mr.date DESC, mr.created DESC",
        (resident_id,)
    ).fetchall()
    mar_list = [dict(rec) for rec in mar]
    db.close()
    return _xlsx_response(
        xg.generate_medication_xlsx(meds, mar_list, r),
        f"medications_{r.get('preferred_name','res')}.xlsx"
    )


@app.route("/residents/<resident_id>/care-plans/xlsx")
@login_required
def resident_care_plans_xlsx(resident_id):
    db = get_db()
    r = _get_resident(db, resident_id)
    plans = [dict(p) for p in db.execute(
        "SELECT * FROM care_plans WHERE resident_id=? ORDER BY created DESC", (resident_id,)
    ).fetchall()]
    db.close()
    return _xlsx_response(
        xg.generate_care_plan_xlsx(plans, r),
        f"care_plans_{r.get('preferred_name','res')}.xlsx"
    )


# ─── Care Plans ─────────────────────────────────────────────────────────────────
@app.route("/care-plans/new", methods=["GET", "POST"])
@login_required
def care_plan_new():
    if not can_edit():
        abort(403)
    db = get_db()
    residents = db.execute("SELECT * FROM residents WHERE active=1 ORDER BY full_name").fetchall()

    if request.method == "POST":
        f = request.form
        resident_id = f.get("resident_id")
        r = db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone()

        # Get next version number
        last = db.execute(
            "SELECT MAX(version) FROM care_plans WHERE resident_id=?", (resident_id,)
        ).fetchone()[0]
        version = (last or 0) + 1

        # Archive old active plans
        db.execute(
            "UPDATE care_plans SET status='Superseded' WHERE resident_id=? AND status='Active'",
            (resident_id,)
        )

        # AI-generate each section if left blank
        sections = [
            "personal_identity_summary", "mobility_care_plan", "personal_care_plan",
            "continence_care_plan", "nutrition_hydration_plan", "medication_management_plan",
            "cognitive_support_plan", "emotional_wellbeing_plan", "social_activity_plan",
            "end_of_life_preferences", "risk_summary", "goals_of_care", "family_involvement_plan"
        ]
        ai_used = False
        section_values = {}
        for sec in sections:
            val = f.get(sec, "").strip()
            if not val and f.get("use_ai") == "yes" and r:
                prompt_data = {
                    "preferred_name": r["preferred_name"],
                    "section": sec.replace("_", " "),
                    "primary_diagnosis": r["primary_diagnosis"],
                    "mobility_level": r["mobility_level"],
                    "continence_needs": r["continence_needs"],
                    "nutrition_texture": r["nutrition_texture"],
                    "falls_risk": r["falls_risk"],
                    "dnacpr_status": r["dnacpr_status"],
                    "mental_capacity": r["mental_capacity"],
                }
                val, used = generate_narrative("care_plan_section", prompt_data)
                if used:
                    ai_used = True
            section_values[sec] = val

        db.execute("""INSERT INTO care_plans
            (resident_id, version, effective_from, review_date, reviewed_by, status,
             personal_identity_summary, mobility_care_plan, personal_care_plan,
             continence_care_plan, nutrition_hydration_plan, medication_management_plan,
             cognitive_support_plan, emotional_wellbeing_plan, social_activity_plan,
             end_of_life_preferences, risk_summary, goals_of_care,
             family_involvement_plan, ai_generated)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (resident_id, version,
             f.get("effective_from", date.today().isoformat()),
             f.get("review_date", ""),
             session.get("full_name"),
             "Active",
             section_values["personal_identity_summary"],
             section_values["mobility_care_plan"],
             section_values["personal_care_plan"],
             section_values["continence_care_plan"],
             section_values["nutrition_hydration_plan"],
             section_values["medication_management_plan"],
             section_values["cognitive_support_plan"],
             section_values["emotional_wellbeing_plan"],
             section_values["social_activity_plan"],
             section_values["end_of_life_preferences"],
             section_values["risk_summary"],
             section_values["goals_of_care"],
             section_values["family_involvement_plan"],
             1 if ai_used else 0)
        )
        db.commit()
        log_action("CREATE_CARE_PLAN", "care_plans", details=f"Resident: {resident_id} v{version}")
        db.close()
        flash(f"Care Plan v{version} saved. Download XLSX from the resident record.", "success")
        return redirect(url_for("resident_detail", resident_id=resident_id))

    db.close()
    return render_template("care_plan_form.html", residents=residents,
                           today=date.today().isoformat())


# ─── Family Communications ───────────────────────────────────────────────────────
# ─── AI Generate Endpoint (AJAX) ────────────────────────────────────────────────
# Domains scored on every wellbeing review, in the order they appear on the form.
WELLBEING_SCORE_FIELDS = (
    "physical_health_score", "mental_health_score", "social_engagement_score",
    "personal_care_score", "nutrition_score", "pain_management_score",
)


def _wellbeing_previous(db, resident_id):
    """
    The resident's most recent completed wellbeing review, or None.

    A wellbeing score in isolation is close to meaningless — 6/10 is good news
    for a resident who scored 3 last month and bad news for one who scored 9.
    Pulling the previous review server-side means the narrative can be written
    about the change, and it keeps the browser from having to hold clinical
    history it has no reason to see.
    """
    if not resident_id:
        return None
    row = db.execute(
        "SELECT * FROM wellbeing WHERE resident_id=? "
        "ORDER BY assessment_date DESC, id DESC LIMIT 1", (resident_id,)
    ).fetchone()
    if not row:
        return None
    row = dict(row)
    return {
        "previous_scores": {f: row.get(f) for f in WELLBEING_SCORE_FIELDS
                            if row.get(f) is not None},
        "previous_overall": row.get("overall_score"),
        "previous_summary": row.get("summary") or "",
        "previous_date": row.get("assessment_date"),
    }


def _wellbeing_overall(source):
    """Mean of the domain scores actually recorded, rounded. 0 if none are."""
    vals = []
    for f in WELLBEING_SCORE_FIELDS:
        raw = source.get(f)
        try:
            if raw not in (None, "", "None"):
                vals.append(int(raw))
        except (TypeError, ValueError):
            continue
    return round(sum(vals) / len(vals)) if vals else 0


@app.route("/api/generate", methods=["POST"])
@login_required
def api_generate():
    if not can_edit():
        return jsonify({"error": "Permission denied"}), 403
    data = request.get_json(force=True)
    report_type = data.get("type", "care_note")

    # Wellbeing is the one form whose narrative needs history the browser does
    # not hold, so the payload is completed here before it reaches the model.
    if report_type == "wellbeing":
        data.setdefault("overall_score", _wellbeing_overall(data))
        db = get_db()
        try:
            prior = _wellbeing_previous(db, data.get("resident_id"))
        finally:
            db.close()
        if prior:
            data.update(prior)

    text, ai_used = generate_narrative(report_type, data)
    return jsonify({
        "text": text,
        "ai_used": ai_used,
        "provider": get_last_provider(),
        "compared_with": (data.get("previous_date") if report_type == "wellbeing" else None),
    })


# ─── Email Queue Routes ─────────────────────────────────────────────────────────
# ─── Duty Dashboard ─────────────────────────────────────────────────────────────
# ─── Enhanced Care Plan (with full patient context) ──────────────────────────────
@app.route("/residents/<resident_id>/care-plan/new", methods=["GET", "POST"])
@login_required
def care_plan_new_enhanced(resident_id):
    if not can_edit():
        abort(403)
    db = get_db()
    r = db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone()
    if not r:
        abort(404)

    # Gather full patient context
    meds = db.execute(
        "SELECT * FROM medications WHERE resident_id=? AND status='active' ORDER BY medication_name",
        (resident_id,)
    ).fetchall()
    risks = db.execute(
        "SELECT * FROM risk_assessments WHERE resident_id=? ORDER BY date_assessed DESC LIMIT 10",
        (resident_id,)
    ).fetchall()
    incidents = db.execute(
        "SELECT * FROM incidents WHERE resident_id=? ORDER BY date DESC LIMIT 5",
        (resident_id,)
    ).fetchall()
    wellbeing_list = db.execute(
        "SELECT * FROM wellbeing WHERE resident_id=? ORDER BY assessment_date DESC LIMIT 3",
        (resident_id,)
    ).fetchall()
    recent_notes = db.execute(
        "SELECT * FROM care_notes WHERE resident_id=? ORDER BY date DESC LIMIT 5",
        (resident_id,)
    ).fetchall()
    prev_plan = db.execute(
        "SELECT * FROM care_plans WHERE resident_id=? ORDER BY version DESC LIMIT 1",
        (resident_id,)
    ).fetchone()

    if request.method == "POST":
        f = request.form
        use_ai = f.get("use_ai") == "yes"

        # Build context strings for AI
        meds_str = "\n".join(
            f"- {m['medication_name']} {m['dose']} {m['route']} {m['frequency']}"
            + (f" [CONTROLLED]" if m["is_controlled"] else "")
            + (f" [PRN: {m['prn_instructions']}]" if m["is_prn"] else "")
            + (f" NOTE: {m['admin_notes']}" if m["admin_notes"] else "")
            for m in meds
        )
        risk_str = "\n".join(
            f"- {rk['assessment_type']}: {rk['risk_level']} (scored {rk['score'] or '?'}/20, {rk['date_assessed']})"
            for rk in risks
        )
        inc_str = "\n".join(
            f"- {inc['date']}: {inc['incident_type']} ({inc['severity']}) — {inc['description'][:100]}"
            for inc in incidents
        )
        wb_str = "\n".join(
            f"- {w['assessment_date']}: Overall {w['overall_score']}/10, "
            f"Physical {w['physical_health_score']}/10, Mental {w['mental_health_score']}/10"
            for w in wellbeing_list
        )
        notes_str = "\n".join(
            f"- {n['date']} ({n['shift']}): Mood={n['mood']}, Appetite={n['appetite']}, "
            f"Fluid={n['fluid_intake_ml']}ml. {(n['care_narrative'] or '')[:100]}"
            for n in recent_notes
        )

        base_ai_data = {
            **dict(r),
            "medications_list":      meds_str,
            "risk_summary_context":  risk_str,
            "recent_incidents":      inc_str,
            "wellbeing_summary":     wb_str,
            "recent_care_notes":     notes_str,
        }

        sections = [
            "personal_identity_summary", "mobility_care_plan", "personal_care_plan",
            "continence_care_plan", "nutrition_hydration_plan", "medication_management_plan",
            "cognitive_support_plan", "emotional_wellbeing_plan", "social_activity_plan",
            "end_of_life_preferences", "risk_summary", "goals_of_care", "family_involvement_plan"
        ]

        SECTION_LABELS = {
            "personal_identity_summary":  "Personal Identity Summary",
            "mobility_care_plan":         "Mobility Care Plan",
            "personal_care_plan":         "Personal Care Plan",
            "continence_care_plan":       "Continence Care Plan",
            "nutrition_hydration_plan":   "Nutrition & Hydration Plan",
            "medication_management_plan": "Medication Management Plan",
            "cognitive_support_plan":     "Cognitive Support Plan",
            "emotional_wellbeing_plan":   "Emotional Wellbeing Plan",
            "social_activity_plan":       "Social Activity Plan",
            "end_of_life_preferences":    "End of Life Preferences",
            "risk_summary":               "Risk Summary",
            "goals_of_care":              "Goals of Care",
            "family_involvement_plan":    "Family Involvement Plan",
        }

        filled = {}
        ai_used_any = False
        rag_chunks_used_all = []
        rag_mode = session.get("rag_mode", "tfidf")
        index_built = rag_engine.get_index_status().get("built", False)

        for sec in sections:
            val = f.get(sec, "").strip()
            if not val and use_ai:
                prev_sec = (prev_plan[sec] if prev_plan and prev_plan[sec] else "") if prev_plan else ""
                sec_data = {**base_ai_data,
                            "section": SECTION_LABELS.get(sec, sec),
                            "previous_plan_section": prev_sec}

                # ── RAG: retrieve similar records before generation ────────────
                rag_chunks = []
                if index_built:
                    query_text = (
                        f"{SECTION_LABELS.get(sec, sec)} care plan for resident with "
                        f"{base_ai_data.get('primary_diagnosis','unknown diagnosis')}"
                    )
                    rag_chunks = rag_engine.retrieve(
                        query=query_text,
                        resident_id=resident_id,
                        top_k=5,
                        mode=rag_mode,
                    )
                    if rag_chunks:
                        sec_data["rag_context"] = rag_engine.build_rag_context_block(rag_chunks)
                        rag_chunks_used_all.extend([c["chunk_id"] for c in rag_chunks])
                        rag_engine.log_retrieval(
                            db_path=DB_PATH,
                            resident_id=resident_id,
                            query_summary=query_text[:200],
                            section=sec,
                            num_chunks=len(rag_chunks),
                            mode=rag_mode,
                            chunk_ids=[c["chunk_id"] for c in rag_chunks],
                            generated_by=session.get("username", "unknown"),
                        )

                val, ai_ok = generate_narrative("care_plan_section", sec_data)
                if ai_ok:
                    ai_used_any = True
            filled[sec] = val

        # Archive previous active plans
        db.execute(
            "UPDATE care_plans SET status='Superseded' WHERE resident_id=? AND status='Active'",
            (resident_id,)
        )
        version = (prev_plan["version"] + 1) if prev_plan else 1

        generation_method = (
            "rag" if rag_chunks_used_all else ("direct" if ai_used_any else "manual")
        )
        db.execute("""INSERT INTO care_plans
            (resident_id, version, effective_from, review_date, reviewed_by, status,
             personal_identity_summary, mobility_care_plan, personal_care_plan,
             continence_care_plan, nutrition_hydration_plan, medication_management_plan,
             cognitive_support_plan, emotional_wellbeing_plan, social_activity_plan,
             end_of_life_preferences, risk_summary, goals_of_care,
             family_involvement_plan, ai_generated,
             generation_method, rag_chunks_used, rag_mode)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (resident_id, version,
             f.get("effective_from", date.today().isoformat()),
             f.get("review_date", ""),
             session.get("full_name"), "Active",
             filled["personal_identity_summary"], filled["mobility_care_plan"],
             filled["personal_care_plan"], filled["continence_care_plan"],
             filled["nutrition_hydration_plan"], filled["medication_management_plan"],
             filled["cognitive_support_plan"], filled["emotional_wellbeing_plan"],
             filled["social_activity_plan"], filled["end_of_life_preferences"],
             filled["risk_summary"], filled["goals_of_care"],
             filled["family_involvement_plan"], 1 if ai_used_any else 0,
             generation_method,
             json.dumps(list(set(rag_chunks_used_all))) if rag_chunks_used_all else None,
             rag_mode if rag_chunks_used_all else None)
        )
        db.commit()
        log_action("CREATE_CARE_PLAN", "care_plans",
                   details=f"Resident: {resident_id} Version: {version}")
        db.close()
        flash(f"Care plan v{version} created{'  (AI-assisted)' if ai_used_any else ''}.", "success")
        return redirect(url_for("resident_detail", resident_id=resident_id))

    db.close()
    return render_template("care_plan_enhanced.html",
                           r=dict(r),
                           meds=[dict(m) for m in meds],
                           risks=[dict(rk) for rk in risks],
                           incidents=[dict(i) for i in incidents],
                           wellbeing_list=[dict(w) for w in wellbeing_list],
                           recent_notes=[dict(n) for n in recent_notes],
                           prev_plan=dict(prev_plan) if prev_plan else None,
                           today=date.today().isoformat(),
                           next_review=(date.today() + timedelta(days=90)).isoformat())


# ─── RAG Knowledge Base Routes ──────────────────────────────────────────────────

@app.route("/knowledge-base")
@login_required
def knowledge_base():
    """Admin page: index stats, rebuild controls."""
    status = rag_engine.get_index_status()
    mode = request.args.get("mode", session.get("rag_mode", "tfidf"))
    db = get_db()
    audit_rows = db.execute(
        "SELECT * FROM rag_audit_log ORDER BY retrieved_at DESC LIMIT 20"
    ).fetchall()
    db.close()
    return render_template("knowledge_base.html",
                           index_status=status,
                           rag_mode=mode,
                           rag_audit_rows=audit_rows,
                           current_user_role=session.get("role", ""))


@app.route("/knowledge-base/build", methods=["POST"])
@login_required
def knowledge_base_build():
    if not can_approve():
        flash("Only managers and senior carers can rebuild the knowledge base.", "error")
        return redirect(url_for("knowledge_base"))
    mode = request.form.get("mode", "tfidf")
    session["rag_mode"] = mode
    rag_engine.invalidate_cache()
    try:
        result = rag_engine.build_index(DB_PATH, mode=mode)
        if "error" in result:
            flash(f"Index build failed: {result['error']}", "error")
        else:
            flash(
                f"Knowledge base built — {result['num_chunks']:,} chunks from "
                f"{result['num_docs']} documents (mode: {result['mode']})",
                "success"
            )
    except Exception as e:
        flash(f"Index build error: {e}", "error")
    return redirect(url_for("knowledge_base"))


@app.route("/knowledge-base/clear", methods=["POST"])
@role_required("manager")
def knowledge_base_clear():
    import shutil
    try:
        index_dir = rag_engine.INDEX_DIR
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            rag_engine.invalidate_cache()
        flash("Knowledge base index cleared.", "info")
    except Exception as e:
        flash(f"Could not clear index: {e}", "error")
    return redirect(url_for("knowledge_base"))


@app.route("/api/rag-retrieve", methods=["POST"])
@login_required
def api_rag_retrieve():
    data = request.get_json(force=True) or {}
    query       = data.get("query", "")
    resident_id = data.get("resident_id", "")
    section     = data.get("section", "")
    top_k       = int(data.get("top_k", 5))
    mode        = session.get("rag_mode", "tfidf")

    if not resident_id or not query:
        return jsonify({"error": "resident_id and query required"}), 400

    chunks = rag_engine.retrieve(
        query=query, resident_id=resident_id, top_k=top_k, mode=mode,
    )
    context_block = rag_engine.build_rag_context_block(chunks)

    rag_engine.log_retrieval(
        db_path=DB_PATH, resident_id=resident_id, query_summary=query[:200],
        section=section, num_chunks=len(chunks), mode=mode,
        chunk_ids=[c["chunk_id"] for c in chunks],
        generated_by=session.get("username", "unknown"),
    )

    return jsonify({
        "chunks": chunks,
        "context_block": context_block,
        "mode": mode,
        "num_retrieved": len(chunks),
    })


# ─── RAG Evaluation Routes ───────────────────────────────────────────────────────

@app.route("/api/rag-trace", methods=["POST"])
@login_required
def api_rag_trace():
    """
    Generate one care plan section and show the working.

    This exists for the reader who does not want a table of metrics, they want
    to see the thing happen once: here is the question, here are the records it
    pulled from this resident's file, here is what it wrote, and here is each
    sentence of that answer traced back to the record it came from. The same
    attribution function that produces the aggregate grounding figure produces
    this trace, so the demonstration and the metric cannot disagree.
    """
    data = request.get_json(force=True) or {}
    resident_id = (data.get("resident_id") or "").strip()
    section = (data.get("section") or "mobility").strip()
    try:
        top_k = max(1, min(int(data.get("top_k", 6)), 12))
    except (TypeError, ValueError):
        top_k = 6

    if not resident_id:
        return jsonify({"error": "Choose a resident."}), 400
    if not rag_engine.get_index_status().get("built", False):
        return jsonify({"error": "The knowledge base index has not been built yet. "
                                 "Build it on the Knowledge Base page first."}), 400

    db = get_db()
    r = db.execute("SELECT * FROM residents WHERE resident_id=?", (resident_id,)).fetchone()
    db.close()
    if not r:
        return jsonify({"error": "Unknown resident."}), 404
    r = dict(r)

    section_label = section.replace("_", " ")
    query = (f"{r.get('preferred_name', resident_id)} {section_label} "
             f"needs, current support and recent changes")

    t0 = time.time()
    chunks = rag_engine.retrieve(query, resident_id=resident_id,
                                 top_k=top_k, mode=session.get("rag_mode", "tfidf"))
    retrieval_ms = round((time.time() - t0) * 1000, 1)

    gen_data = {
        "preferred_name": r.get("preferred_name", resident_id),
        "full_name": r.get("full_name", ""),
        "section_name": section,
        "section_label": section_label.title(),
        "allergies": r.get("allergies", ""),
        "rag_context": rag_engine.build_rag_context_block(chunks),
    }

    t0 = time.time()
    text, ai_used = generate_narrative("care_plan_section", gen_data)
    generation_ms = round((time.time() - t0) * 1000, 1)

    trace = rag_evaluator.attribute_sentences(text, chunks)

    # The control arm: the same request with the resident's records withheld.
    # Without it a reader has no way to judge whether the grounding figure is a
    # property of the system or a property of the metric.
    control = None
    if data.get("include_control"):
        blind = dict(gen_data)
        blind["rag_context"] = ""
        blind_text, _ = generate_narrative("care_plan_section", blind)
        control = {
            "text": blind_text,
            "attribution": rag_evaluator.attribute_sentences(blind_text, chunks),
        }

    return jsonify({
        "resident": {"id": resident_id, "name": r.get("preferred_name", resident_id),
                     "room": r.get("room_number", "")},
        "section": section,
        "query": query,
        "ai_used": ai_used,
        "provider": get_last_provider(),
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "chunks": [
            {"rank": i + 1,
             "source_type": c.get("source_type", ""),
             "date": c.get("date", ""),
             "score": round(float(c.get("score", 0)), 4),
             "text": c.get("text", "")}
            for i, c in enumerate(chunks)
        ],
        "text": text,
        "trace": trace,
        "control": control,
    })


@app.route("/rag-evaluation")
@login_required
def rag_evaluation():
    cached = rag_evaluator.load_cached_results()
    index_status = rag_engine.get_index_status()
    interpretation = rag_evaluator.interpret_results(cached) if cached else None

    db = get_db()
    residents = db.execute(
        "SELECT resident_id, preferred_name, room_number FROM residents "
        "WHERE active=1 ORDER BY room_number"
    ).fetchall()
    db.close()

    # The retrieval experiments are run offline by rag_experiments.py; the page
    # reads their output rather than re-running anything on a page load.
    experiments = None
    exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "rag_experiments.json")
    try:
        with open(exp_path, "r", encoding="utf-8") as fh:
            experiments = json.load(fh)
    except (OSError, ValueError):
        pass

    return render_template("rag_evaluation.html",
                           results=cached,
                           interpretation=interpretation,
                           index_status=index_status,
                           residents=residents,
                           experiments=experiments,
                           j_experiments=json.dumps(experiments) if experiments else "null",
                           j_results=json.dumps(cached, default=str) if cached else "null")


@app.route("/rag-evaluation/run", methods=["POST"])
@role_required("manager", "senior_carer")
def rag_evaluation_run():
    """
    Start the evaluation on a worker thread and return immediately.

    This used to run inside the request. Each evaluated section costs two LLM
    calls, so a full run is dozens of API round-trips: the browser sat on a
    spinner with no progress and eventually gave up, which is indistinguishable
    from a crash. The page now polls /rag-evaluation/status instead.
    """
    mode  = request.form.get("rag_mode", session.get("rag_mode", "tfidf"))
    try:
        top_k = int(request.form.get("top_k", 5))
    except (TypeError, ValueError):
        top_k = 5
    try:
        max_sections = int(request.form.get("max_sections", 12))
    except (TypeError, ValueError):
        max_sections = 12
    max_sections = max(1, min(max_sections, 40))

    wants_json = request.headers.get("Accept", "").startswith("application/json") \
        or request.form.get("ajax") == "1"

    if not rag_engine.get_index_status().get("built", False):
        msg = "Build the knowledge base index first (Knowledge Base → Build)."
        if wants_json:
            return jsonify({"started": False, "reason": msg}), 400
        flash(msg, "warning")
        return redirect(url_for("rag_evaluation"))

    result = rag_evaluator.start_job(db_path=DB_PATH, rag_mode=mode,
                                     top_k=top_k, max_sections=max_sections)
    if wants_json:
        return jsonify(result)
    if result.get("started"):
        flash(f"Evaluation started in the background ({max_sections} section(s)). "
              "Progress is shown on this page.", "info")
    else:
        flash(result.get("reason", "Could not start evaluation."), "warning")
    return redirect(url_for("rag_evaluation"))


@app.route("/rag-evaluation/status")
@login_required
def rag_evaluation_status():
    """Progress of the background evaluation, polled by the dashboard."""
    return jsonify(rag_evaluator.job_status())


@app.route("/api/ai-diagnostics", methods=["POST", "GET"])
@login_required
def api_ai_diagnostics():
    """
    Ping every configured AI provider and report exactly what happened.
    Turns "the AI is offline" into an actionable HTTP status and message.
    """
    try:
        return jsonify(llm_client.diagnose(timeout=20))
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500




# ─── Conversational RAG Assistant ("Ask the Records") ───────────────────────────
#
# A chat interface over the whole record: hybrid retrieval for narrative
# evidence, parameterised SQL tools for exact figures, and a per-answer trace
# so any number can be traced back to the query that produced it.

def _chat_session_key() -> str:
    """
    The conversation currently open for this user. Held in the Flask session so
    a browser refresh stays in the same thread; the conversation itself lives in
    the database, so it survives logout and can be reopened from the sidebar.
    """
    if "chat_key" not in session:
        session["chat_key"] = chat_engine.create_conversation(
            DB_PATH, session.get("username", "anon"))
    return session["chat_key"]


def _owns_conversation(key: str) -> bool:
    """A user may only open their own conversations."""
    owner = chat_engine.conversation_owner(DB_PATH, key)
    return owner is None or owner == session.get("username")


@app.route("/chat")
@login_required
def chat_page():
    chat_engine.ensure_tables(DB_PATH)

    # ?c=<key> opens an existing conversation from the sidebar.
    requested = (request.args.get("c") or "").strip()
    if requested:
        if _owns_conversation(requested):
            session["chat_key"] = requested
        else:
            flash("That conversation belongs to another user.", "error")
    key = _chat_session_key()
    ai = llm_client.status()
    ai["order"] = llm_client.auto_order()
    return render_template(
        "chat.html",
        residents=chat_engine.get_residents(DB_PATH),
        history=chat_engine.load_history(DB_PATH, key, limit=60),
        conversations=chat_engine.list_conversations(
            DB_PATH, session.get("username", "anon")),
        active_chat=key,
        index_status=rag_advanced.get_status(),
        ai_providers=ai["providers"],
        ai_active=ai["active"],
        ai_order=ai.get("order") or [],
    )


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question."}), 400
    if len(question) > 1000:
        question = question[:1000]

    resident_id = (data.get("resident_id") or "").strip()
    scope = [resident_id] if resident_id else None

    try:
        months = int(data.get("months", 12))
    except (TypeError, ValueError):
        months = 12
    months = max(1, min(months, 240))

    try:
        top_k = int(data.get("top_k", 8))
    except (TypeError, ValueError):
        top_k = 8
    top_k = max(3, min(top_k, 16))

    # "template" pins the offline path so a reader can see what the assistant
    # produces with no model at all; anything else means "use Gemini".
    provider = (data.get("provider") or "").strip().lower() or None
    if provider not in (None, "gemini", "template"):
        provider = None

    try:
        result = chat_engine.answer(
            db_path=DB_PATH,
            question=question,
            session_key=_chat_session_key(),
            username=session.get("username", "unknown"),
            scope_resident_ids=scope,
            months=months,
            top_k=top_k,
            provider=provider,
            use_llm_planner=bool(data.get("use_llm_planner", True)),
            retrieval_opts={
                "use_hybrid":  bool(data.get("use_hybrid", True)),
                "use_mmr":     bool(data.get("use_mmr", True)),
                "use_recency": bool(data.get("use_recency", True)),
            },
        )
    except Exception as e:
        app.logger.exception("chat failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    log_action("chat_query", "chat_messages", None, question[:200])
    return jsonify(result)


@app.route("/chat/reset", methods=["POST"])
@login_required
def chat_reset():
    """Start a new conversation. The previous one stays in the sidebar."""
    session["chat_key"] = chat_engine.create_conversation(
        DB_PATH, session.get("username", "anon"))
    return jsonify({"ok": True, "session_key": session["chat_key"]})


@app.route("/api/chats")
@login_required
def api_chats():
    """Conversation list for the sidebar."""
    return jsonify({
        "conversations": chat_engine.list_conversations(
            DB_PATH, session.get("username", "anon")),
        "active": session.get("chat_key"),
    })


@app.route("/api/chats/<key>/rename", methods=["POST"])
@login_required
def api_chat_rename(key):
    if not _owns_conversation(key):
        return jsonify({"error": "Not your conversation."}), 403
    title = ((request.get_json(silent=True) or {}).get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title required."}), 400
    chat_engine.rename_conversation(DB_PATH, key, title)
    return jsonify({"ok": True, "title": title[:120]})


@app.route("/api/chats/<key>/delete", methods=["POST"])
@login_required
def api_chat_delete(key):
    if not _owns_conversation(key):
        return jsonify({"error": "Not your conversation."}), 403
    chat_engine.delete_conversation(DB_PATH, key)
    log_action("chat_delete", "chat_conversations", None, key)
    if session.get("chat_key") == key:
        session["chat_key"] = chat_engine.create_conversation(
            DB_PATH, session.get("username", "anon"))
    return jsonify({"ok": True, "active": session.get("chat_key")})


@app.route("/chat/build-index", methods=["POST"])
@login_required
def chat_build_index():
    if not can_approve():
        flash("Only managers and senior carers can rebuild the knowledge index.", "error")
        return redirect(url_for("chat_page"))
    try:
        result = rag_advanced.build_index(DB_PATH)
        if result.get("error"):
            flash(f"Index build failed: {result['error']}", "error")
        else:
            flash(f"Hybrid index built — {result['num_chunks']:,} chunks from "
                  f"{result['num_docs']:,} records covering "
                  f"{result['date_range'][0]} to {result['date_range'][1]}.", "success")
        log_action("rag_index_build_v2", details=json.dumps(result)[:400])
    except Exception as e:
        flash(f"Index build error: {e}", "error")
    return redirect(url_for("chat_page"))


# ─── Analytics Dashboard ─────────────────────────────────────────────────────────
@app.route("/analytics")
@login_required
def analytics():
    import statistics as _stats
    import math as _math

    db = get_db()
    c  = db.cursor()

    def box_stats(vals):
        if not vals:
            return {"min":0,"q1":0,"median":0,"q3":0,"max":0,"mean":0,"std":0}
        s = sorted(vals); n = len(s)
        def pct(p):
            idx = p/100*(n-1); lo = int(idx)
            return s[lo]+(idx-lo)*(s[lo+1]-s[lo]) if lo < n-1 else s[lo]
        return {"min":s[0],"q1":round(pct(25),2),"median":round(pct(50),2),
                "q3":round(pct(75),2),"max":s[-1],
                "mean":round(_stats.mean(vals),2),
                "std":round(_stats.stdev(vals) if len(vals)>1 else 0,2)}

    def short_month(ym):
        from datetime import date as _d
        y,m = ym.split('-')
        return _d(int(y),int(m),1).strftime("%b '%y")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_notes       = c.execute("SELECT COUNT(*) FROM care_notes").fetchone()[0]
    total_incidents   = c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    serious_incidents = c.execute("SELECT COUNT(*) FROM incidents WHERE severity IN ('Major','Critical')").fetchone()[0]
    avg_fluid_row     = c.execute("SELECT ROUND(AVG(fluid_intake_ml),0) FROM care_notes").fetchone()[0]
    avg_fluid         = int(avg_fluid_row or 0)
    adherence_row     = c.execute("SELECT ROUND(100.0*SUM(CASE WHEN administered='Yes' THEN 1 ELSE 0 END)/COUNT(*),1) FROM mar_records").fetchone()[0]
    overall_adherence = adherence_row or 0
    avg_wb_row        = c.execute("SELECT ROUND(AVG(overall_score),2) FROM wellbeing").fetchone()[0]
    avg_wellbeing     = avg_wb_row or 0
    total_mar         = c.execute("SELECT COUNT(*) FROM mar_records").fetchone()[0]

    # ── Normalised incident rate ───────────────────────────────────────────────
    rows = c.execute("SELECT strftime('%Y-%m',date) m, COUNT(*) n FROM incidents GROUP BY m ORDER BY m").fetchall()
    inc_months = [short_month(r[0]) for r in rows]
    inc_rates  = [round(r[1]/(5*30)*100,2) for r in rows]

    # ── Incident severity totals ───────────────────────────────────────────────
    sev_totals = []
    for sv in ["Minor","Moderate","Major","Critical"]:
        sev_totals.append(c.execute("SELECT COUNT(*) FROM incidents WHERE severity=?",(sv,)).fetchone()[0])

    # ── Wellbeing box plots per resident ──────────────────────────────────────
    rows = c.execute("""SELECT r.preferred_name, w.overall_score, w.physical_health_score,
                        w.mental_health_score, w.social_engagement_score
                        FROM wellbeing w JOIN residents r ON w.resident_id=r.resident_id
                        ORDER BY r.preferred_name""").fetchall()
    wb_by_res = {}
    for row in rows:
        n = row[0]
        if n not in wb_by_res:
            wb_by_res[n] = {"overall":[],"physical":[],"mental":[],"social":[]}
        wb_by_res[n]["overall"].append(row[1])
        wb_by_res[n]["physical"].append(row[2])
        wb_by_res[n]["mental"].append(row[3])
        wb_by_res[n]["social"].append(row[4])
    wb_res_names = list(wb_by_res.keys())
    wb_box       = {n: box_stats(d["overall"]) for n,d in wb_by_res.items()}

    # ── Wellbeing trend + OLS regression ─────────────────────────────────────
    rows = c.execute("""SELECT strftime('%Y-%m',assessment_date) m,
                        ROUND(AVG(overall_score),2), ROUND(AVG(physical_health_score),2),
                        ROUND(AVG(mental_health_score),2)
                        FROM wellbeing GROUP BY m ORDER BY m""").fetchall()
    wb_months = [short_month(r[0]) for r in rows]
    wb_avg    = [r[1] for r in rows]
    wb_phys   = [r[2] for r in rows]
    wb_ment   = [r[3] for r in rows]
    x = list(range(len(wb_avg))); y = wb_avg
    if len(x) > 1:
        xm = _stats.mean(x); ym = _stats.mean(y)
        denom = sum((xi-xm)**2 for xi in x)
        b1 = sum((xi-xm)*(yi-ym) for xi,yi in zip(x,y))/denom if denom else 0
        b0 = ym - b1*xm
        wb_trend = [round(b0+b1*xi,2) for xi in x]
        wb_slope = round(b1,4)
    else:
        wb_trend = wb_avg; wb_slope = 0.0

    # ── Radar per resident (6 domains) ────────────────────────────────────────
    rows = c.execute("""SELECT r.preferred_name,
                        ROUND(AVG(w.physical_health_score),1),ROUND(AVG(w.mental_health_score),1),
                        ROUND(AVG(w.social_engagement_score),1),ROUND(AVG(w.personal_care_score),1),
                        ROUND(AVG(w.nutrition_score),1),ROUND(AVG(w.pain_management_score),1)
                        FROM wellbeing w JOIN residents r ON w.resident_id=r.resident_id
                        GROUP BY r.preferred_name""").fetchall()
    radar_names = [r[0] for r in rows]
    radar_data  = [[r[1],r[2],r[3],r[4],r[5],r[6]] for r in rows]
    radar_labels = ["Physical","Mental","Social","Personal Care","Nutrition","Pain Mgmt"]

    # ── Falls risk score vs actual falls ──────────────────────────────────────
    rows = c.execute("""SELECT r.preferred_name, ROUND(AVG(ra.score),1),
                        (SELECT COUNT(*) FROM care_notes cn
                         WHERE cn.resident_id=r.resident_id AND cn.falls_this_shift=1)
                        FROM risk_assessments ra JOIN residents r ON ra.resident_id=r.resident_id
                        WHERE ra.assessment_type='Falls' GROUP BY r.resident_id""").fetchall()
    falls_scatter = [{"name":r[0],"risk_score":r[1],"actual_falls":r[2]} for r in rows]

    # ── Fluid vs Wellbeing correlation ────────────────────────────────────────
    fluid_map = {(r[0],r[1]):r[2] for r in c.execute(
        "SELECT resident_id,strftime('%Y-%m',date),AVG(fluid_intake_ml) FROM care_notes GROUP BY resident_id,strftime('%Y-%m',date)"
    ).fetchall()}
    wb_map_rows = c.execute(
        "SELECT resident_id,strftime('%Y-%m',assessment_date),overall_score FROM wellbeing"
    ).fetchall()
    pairs = []
    for row in wb_map_rows:
        key = (row[0], row[1])
        if key in fluid_map:
            pairs.append({"fluid": round(fluid_map[key],0), "wb": row[2]})
    fluid_wb_r   = 0.0
    fluid_wb_reg = {}
    if len(pairs) > 2:
        fx = [p["fluid"] for p in pairs]; wy = [p["wb"] for p in pairs]
        fm = _stats.mean(fx); wm = _stats.mean(wy)
        cov = sum((a-fm)*(b-wm) for a,b in zip(fx,wy))/len(fx)
        sd_f = _stats.stdev(fx); sd_w = _stats.stdev(wy)
        fluid_wb_r = round(cov/(sd_f*sd_w),3) if sd_f and sd_w else 0
        b1r = cov/(sd_f**2) if sd_f else 0; b0r = wm - b1r*fm
        fluid_wb_reg = {"x0":round(min(fx),0),"y0":round(b0r+b1r*min(fx),2),
                        "x1":round(max(fx),0),"y1":round(b0r+b1r*max(fx),2)}

    # ── Medication adherence with 95% CI ──────────────────────────────────────
    rows = c.execute("""SELECT r.preferred_name,
                        SUM(CASE WHEN m.administered='Yes' THEN 1 ELSE 0 END), COUNT(*)
                        FROM mar_records m JOIN residents r ON m.resident_id=r.resident_id
                        GROUP BY r.preferred_name""").fetchall()
    adherence_detail = []
    for row in rows:
        p  = row[1]/row[2] if row[2] else 0
        se = _math.sqrt(p*(1-p)/row[2]) if row[2] else 0
        adherence_detail.append({"name":row[0],"pct":round(p*100,1),
                                  "ci":round(se*1.96*100,2),"n":row[2]})

    # ── Heatmap: incident type × severity ────────────────────────────────────
    itypes = [r[0] for r in c.execute(
        "SELECT DISTINCT incident_type FROM incidents ORDER BY incident_type").fetchall()]
    sevs   = ["Minor","Moderate","Major","Critical"]
    heatmap_data = []
    for it in itypes:
        row = []
        for sv in sevs:
            row.append(c.execute(
                "SELECT COUNT(*) FROM incidents WHERE incident_type=? AND severity=?",(it,sv)
            ).fetchone()[0])
        heatmap_data.append(row)

    # ── Fluid box plots per resident ──────────────────────────────────────────
    rows = c.execute("""SELECT r.preferred_name, cn.fluid_intake_ml
                        FROM care_notes cn JOIN residents r ON cn.resident_id=r.resident_id""").fetchall()
    fd = {}
    for row in rows:
        fd.setdefault(row[0],[]).append(row[1])
    fluid_res_names = list(fd.keys())
    fluid_box       = {n: box_stats(v) for n,v in fd.items()}

    # ── Mood distribution ────────────────────────────────────────────────────
    rows = c.execute("SELECT mood, COUNT(*) FROM care_notes GROUP BY mood ORDER BY COUNT(*) DESC").fetchall()
    mood_labels = [r[0] for r in rows]
    mood_counts = [r[1] for r in rows]

    # ── Statistical summary table ─────────────────────────────────────────────
    stat_domains = {}
    for col, label in [("overall_score","Overall"),("physical_health_score","Physical"),
                        ("mental_health_score","Mental"),("nutrition_score","Nutrition"),
                        ("pain_management_score","Pain Mgmt")]:
        vals = [r[0] for r in c.execute(f"SELECT {col} FROM wellbeing").fetchall() if r[0] is not None]
        stat_domains[label] = {"mean":round(_stats.mean(vals),2),"std":round(_stats.stdev(vals),2),
                                "median":round(_stats.median(vals),2),"min":min(vals),"max":max(vals)} if vals else {}

    # ── Data completeness ────────────────────────────────────────────────────
    missing_narr = c.execute("SELECT COUNT(*) FROM care_notes WHERE care_narrative IS NULL OR care_narrative=''").fetchone()[0]
    completeness = round(100 - missing_narr/total_notes*100, 1) if total_notes else 100.0

    db.close()

    return render_template("analytics.html",
        total_notes=total_notes, total_incidents=total_incidents,
        serious_incidents=serious_incidents, avg_fluid=avg_fluid,
        overall_adherence=overall_adherence, avg_wellbeing=avg_wellbeing,
        total_mar=total_mar, completeness=completeness, wb_slope=wb_slope,
        fluid_wb_r=fluid_wb_r,
        # JSON payloads
        j_inc_months=json.dumps(inc_months),
        j_inc_rates=json.dumps(inc_rates),
        j_sev_totals=json.dumps(sev_totals),
        j_wb_res_names=json.dumps(wb_res_names),
        j_wb_box=json.dumps(wb_box),
        j_wb_months=json.dumps(wb_months),
        j_wb_avg=json.dumps(wb_avg),
        j_wb_phys=json.dumps(wb_phys),
        j_wb_ment=json.dumps(wb_ment),
        j_wb_trend=json.dumps(wb_trend),
        j_radar_names=json.dumps(radar_names),
        j_radar_data=json.dumps(radar_data),
        j_radar_labels=json.dumps(radar_labels),
        j_falls_scatter=json.dumps(falls_scatter),
        j_pairs=json.dumps(pairs),
        j_fluid_wb_reg=json.dumps(fluid_wb_reg),
        j_adherence=json.dumps(adherence_detail),
        j_fluid_res_names=json.dumps(fluid_res_names),
        j_fluid_box=json.dumps(fluid_box),
        j_mood_labels=json.dumps(mood_labels),
        j_mood_counts=json.dumps(mood_counts),
        j_heatmap_rows=json.dumps(itypes),
        j_heatmap_cols=json.dumps(sevs),
        j_heatmap_data=json.dumps(heatmap_data),
        stat_domains=stat_domains,
    )


# ─── Error handlers ──────────────────────────────────────────────────────────────
@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403, msg="Access denied."), 403

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, msg="Page not found."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, msg="Something went wrong."), 500


# ─── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n[*] CareDocs AI running at http://localhost:{port}")
    print("   Demo logins:")
    print("     manager1 / manager123  (full access)")
    print("     senior1  / senior123")
    print("     carer1   / carer123\n")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=debug)
