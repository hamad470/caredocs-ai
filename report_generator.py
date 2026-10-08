"""
PDF report generator using ReportLab.
Produces professional care home reports for:
  - Care Notes
  - Incident Reports
  - Handover Reports
  - Wellbeing Assessments
  - Risk Assessments
  - Medication Charts (MAR)
  - Compliance Summary
"""
import io
import os
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


BRAND_BLUE   = colors.HexColor("#1e3a5f")
BRAND_LIGHT  = colors.HexColor("#e8f0fe")
BRAND_ACCENT = colors.HexColor("#2563eb")
GRAY_LIGHT   = colors.HexColor("#f1f5f9")
GRAY_MID     = colors.HexColor("#94a3b8")
RED          = colors.HexColor("#dc2626")
AMBER        = colors.HexColor("#d97706")
GREEN        = colors.HexColor("#16a34a")
WHITE        = colors.white


def _rag_color(status):
    s = (status or "").lower()
    if s in ("red", "high", "very_high", "critical", "open"):
        return RED
    if s in ("amber", "medium", "pending_approval", "draft"):
        return AMBER
    return GREEN


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title":   ParagraphStyle("title",   fontName="Helvetica-Bold", fontSize=16,
                                  textColor=WHITE,   spaceAfter=2),
        "subtitle":ParagraphStyle("subtitle",fontName="Helvetica",      fontSize=10,
                                  textColor=WHITE,   spaceAfter=2),
        "h2":      ParagraphStyle("h2",      fontName="Helvetica-Bold", fontSize=12,
                                  textColor=BRAND_BLUE, spaceBefore=10, spaceAfter=4),
        "body":    ParagraphStyle("body",    fontName="Helvetica",      fontSize=9,
                                  textColor=colors.black, spaceAfter=3, leading=14),
        "label":   ParagraphStyle("label",   fontName="Helvetica-Bold", fontSize=8,
                                  textColor=GRAY_MID),
        "value":   ParagraphStyle("value",   fontName="Helvetica",      fontSize=9,
                                  textColor=colors.black),
        "narrative":ParagraphStyle("narr",  fontName="Helvetica-Oblique", fontSize=9,
                                   textColor=colors.HexColor("#1e293b"),
                                   backColor=GRAY_LIGHT, leading=14,
                                   borderPadding=(6,8,6,8)),
        "footer":  ParagraphStyle("footer",  fontName="Helvetica",      fontSize=7,
                                   textColor=GRAY_MID, alignment=TA_CENTER),
        "small":   ParagraphStyle("small",   fontName="Helvetica",      fontSize=8,
                                   textColor=colors.black),
    }


def _synthetic_banner() -> Table:
    """Red banner inserted at top of every report to label synthetic data."""
    st = _styles()
    disclaimer_style = ParagraphStyle(
        "disclaimer", fontName="Helvetica-Bold", fontSize=7,
        textColor=colors.white, alignment=TA_CENTER, leading=10,
    )
    t = Table([[Paragraph(
        "⚠  SYNTHETIC DATA — ALL NAMES &amp; CLINICAL DETAILS ARE ENTIRELY FICTIONAL  "
        "— GENERATED FOR ACADEMIC RESEARCH (LJMU MSc Data Science 2026) — NOT REAL RECORDS  ⚠",
        disclaimer_style,
    )]], colWidths=[18*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#DC2626")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    return t


def _header_table(title: str, subtitle: str, home_name: str = "Sunrise Care Home (Fictional)") -> Table:
    st = _styles()
    data = [[
        Paragraph(f"<b>{home_name}</b>", st["title"]),
        Paragraph(title, st["title"]),
    ],[
        Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}", st["subtitle"]),
        Paragraph(subtitle, st["subtitle"]),
    ]]
    t = Table(data, colWidths=[8*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BRAND_BLUE),
        ("TEXTCOLOR",  (0,0), (-1,-1), WHITE),
        ("ALIGN",      (1,0), (1,-1),  "RIGHT"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0), (-1,-1), 12),
        ("RIGHTPADDING",(0,0),(-1,-1), 12),
        ("ROUNDEDCORNERS", [4,4,0,0]),
    ]))
    return t


def _kv_table(rows: list[tuple], cols: int = 2) -> Table:
    """Render a key-value grid — rows is list of (label, value) tuples."""
    st = _styles()
    col_w = 9*cm if cols == 2 else [4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm]

    # Group into cols pairs per row
    table_data = []
    for i in range(0, len(rows), cols):
        row = []
        for j in range(cols):
            if i+j < len(rows):
                k, v = rows[i+j]
                row += [Paragraph(str(k), st["label"]),
                        Paragraph(str(v) if v else "—", st["value"])]
            else:
                row += [Paragraph("", st["label"]), Paragraph("", st["value"])]
        table_data.append(row)

    widths = [3*cm, 5.5*cm] * cols if cols <= 2 else [2.5*cm, 4*cm] * cols
    t = Table(table_data, colWidths=widths)
    t.setStyle(TableStyle([
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, GRAY_LIGHT]),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("GRID", (0,0), (-1,-1), 0.25, GRAY_MID),
    ]))
    return t


def _narrative_block(text: str, label: str = "Narrative") -> list:
    st = _styles()
    return [
        Paragraph(label, st["h2"]),
        Paragraph(text or "No narrative recorded.", st["narrative"]),
        Spacer(1, 6),
    ]


def _signature_block(signed_by: str = "", signed_at: str = "") -> Table:
    st = _styles()
    data = [
        [Paragraph("Staff Signature", st["label"]),
         Paragraph("Print Name", st["label"]),
         Paragraph("Date/Time", st["label"]),
         Paragraph("Role", st["label"])],
        [Paragraph(signed_by or "___________________", st["value"]),
         Paragraph(signed_by or "___________________", st["value"]),
         Paragraph(signed_at or "___________________", st["value"]),
         Paragraph("___________________", st["value"])],
    ]
    t = Table(data, colWidths=[4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm])
    t.setStyle(TableStyle([
        ("BOX",        (0,0), (-1,-1), 0.5, BRAND_BLUE),
        ("INNERGRID",  (0,0), (-1,-1), 0.25, GRAY_MID),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0), (-1,-1), 6),
    ]))
    return t


SYNTHETIC_DISCLAIMER = (
    "⚠  SYNTHETIC DATA — ALL NAMES, DIAGNOSES AND CLINICAL DETAILS ARE ENTIRELY FICTIONAL  "
    "— GENERATED FOR ACADEMIC RESEARCH (LJMU MSc Data Science 2026) — NOT REAL RESIDENT RECORDS  ⚠"
)

def _footer(canvas, doc):
    canvas.saveState()
    # ── Synthetic data watermark (top of every page) ───────────────────────
    canvas.setFont("Helvetica-Bold", 6.5)
    canvas.setFillColor(colors.HexColor("#DC2626"))
    canvas.drawCentredString(A4[0]/2, A4[1] - 0.6*cm, SYNTHETIC_DISCLAIMER)
    # ── Normal footer ─────────────────────────────────────────────────────
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GRAY_MID)
    canvas.drawCentredString(
        A4[0]/2, 1*cm,
        f"SYNTHETIC DATA — FOR ACADEMIC USE ONLY — Sunrise Care Home (Fictional) — "
        f"Page {doc.page} — Generated {datetime.now().strftime('%d %b %Y %H:%M')}"
    )
    canvas.restoreState()


# ─── Public report functions ────────────────────────────────────────────────────

def generate_care_note_pdf(note: dict, resident: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    st = _styles()
    els = []
    els.append(_synthetic_banner())
    els.append(Spacer(1, 0.2*cm))

    els.append(_header_table(
        "Daily Care Note",
        f"{resident.get('preferred_name','')} — Room {resident.get('room_number','')} — "
        f"{note.get('date','')} ({note.get('shift','')} Shift)"
    ))
    els.append(Spacer(1, 8))

    # Resident details
    els.append(Paragraph("Resident Details", st["h2"]))
    els.append(_kv_table([
        ("Full Name",      resident.get("full_name","")),
        ("Preferred Name", resident.get("preferred_name","")),
        ("Room",           resident.get("room_number","")),
        ("DOB",            resident.get("date_of_birth","")),
        ("Primary Dx",     resident.get("primary_diagnosis","")),
        ("DNACPR Status",  resident.get("dnacpr_status","")),
    ]))
    els.append(Spacer(1, 6))

    # Care observation
    els.append(Paragraph("Shift Observations", st["h2"]))
    status_color = _rag_color(note.get("status",""))
    els.append(_kv_table([
        ("Date",           note.get("date","")),
        ("Shift",          note.get("shift","")),
        ("Staff Member",   note.get("staff_name","")),
        ("Staff Role",     note.get("staff_role","")),
        ("Note Type",      note.get("note_type","")),
        ("Status",         note.get("status","").replace("_"," ").title()),
        ("Personal Care",  note.get("personal_care","")),
        ("Mood",           note.get("mood","")),
        ("Appetite",       note.get("appetite","")),
        ("Fluid Intake",   f"{note.get('fluid_intake_ml','')} ml"),
        ("Weight",         f"{note.get('weight_kg','')} kg"),
        ("Skin Checked",   note.get("skin_checked","")),
        ("Skin Concern",   note.get("skin_concern","None")),
        ("Repositioned",   note.get("repositioned","")),
        ("Activity",       note.get("activity","")),
        ("Activity Desc.", note.get("activity_description","")),
        ("Sleep Quality",  note.get("sleep_quality","")),
        ("Pain Observed",  note.get("pain_observed","")),
        ("Pain Location",  note.get("pain_location","N/A")),
        ("Falls This Shift", str(note.get("falls_this_shift","0"))),
    ], cols=2))
    els.append(Spacer(1, 6))

    els += _narrative_block(note.get("care_narrative",""), "Care Narrative")
    if note.get("concerns"):
        els += _narrative_block(note["concerns"], "Concerns Flagged")
    if note.get("actions_taken"):
        els += _narrative_block(note["actions_taken"], "Actions Taken")
    if note.get("handover_notes"):
        els += _narrative_block(note["handover_notes"], "Handover Notes")

    # AI indicator
    if note.get("ai_generated"):
        els.append(Paragraph(
            "⚡ Narrative generated with AI assistance — reviewed and approved by staff.",
            st["small"]
        ))
        els.append(Spacer(1, 6))

    els.append(Paragraph("Staff Authorisation", st["h2"]))
    els.append(_signature_block(note.get("approved_by",""), note.get("approved_at","")))

    doc.build(els, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def generate_incident_pdf(incident: dict, resident: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    st = _styles()
    els = []
    els.append(_synthetic_banner())
    els.append(Spacer(1, 0.2*cm))

    els.append(_header_table(
        "Incident Report",
        f"{incident.get('incident_id','NEW')} — {incident.get('date','')} — "
        f"Severity: {incident.get('severity','').upper()}"
    ))
    els.append(Spacer(1, 8))

    els.append(Paragraph("Resident Details", st["h2"]))
    els.append(_kv_table([
        ("Full Name",  resident.get("full_name","")),
        ("Room",       resident.get("room_number","")),
        ("DOB",        resident.get("date_of_birth","")),
        ("NHS Number", resident.get("nhs_number","")),
        ("Primary Dx", resident.get("primary_diagnosis","")),
        ("GP",         resident.get("gp_name","")),
    ]))
    els.append(Spacer(1, 6))

    els.append(Paragraph("Incident Details", st["h2"]))
    els.append(_kv_table([
        ("Incident ID",    incident.get("incident_id","")),
        ("Date",           incident.get("date","")),
        ("Time",           incident.get("time","")),
        ("Shift",          incident.get("shift","")),
        ("Incident Type",  incident.get("incident_type","")),
        ("Severity",       incident.get("severity","")),
        ("Location",       incident.get("location","")),
        ("Witnessed",      incident.get("witnessed","")),
        ("Witness Name",   incident.get("witness_name","N/A")),
        ("1st On Scene",   incident.get("staff_first_on_scene","")),
        ("Injuries",       incident.get("injuries","None noted")),
        ("Medical Attn",   incident.get("medical_attention","No")),
        ("GP Notified",    incident.get("gp_notified","No")),
        ("Family Notified",incident.get("family_notified","No")),
        ("CQC Notification",incident.get("cqc_notification","No")),
        ("Status",         incident.get("status","").title()),
    ], cols=2))
    els.append(Spacer(1, 6))

    els += _narrative_block(incident.get("description",""), "Incident Description")
    els += _narrative_block(incident.get("immediate_actions",""), "Immediate Actions Taken")
    if incident.get("outcome"):
        els += _narrative_block(incident["outcome"], "Outcome")
    if incident.get("investigation_summary"):
        els += _narrative_block(incident["investigation_summary"], "Investigation Summary")
    if incident.get("lessons_learned"):
        els += _narrative_block(incident["lessons_learned"], "Lessons Learned")
    if incident.get("preventative_actions"):
        els += _narrative_block(incident["preventative_actions"], "Preventative Actions")

    if incident.get("ai_generated"):
        els.append(Paragraph(
            "⚡ Narrative generated with AI assistance — reviewed and approved by staff.",
            st["small"]
        ))
        els.append(Spacer(1, 6))

    els.append(Paragraph("Manager Sign-Off", st["h2"]))
    els.append(_signature_block(
        incident.get("manager_sign_off",""),
        incident.get("manager_sign_off_date","")
    ))

    doc.build(els, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def generate_handover_pdf(handover: dict, resident: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    st = _styles()
    els = []
    els.append(_synthetic_banner())
    els.append(Spacer(1, 0.2*cm))

    els.append(_header_table(
        "Shift Handover Report",
        f"{resident.get('preferred_name','')} — {handover.get('date','')} — "
        f"{handover.get('shift_ending','')} → {handover.get('shift_starting','')} Shift"
    ))
    els.append(Spacer(1, 8))

    els.append(_kv_table([
        ("Resident",       resident.get("full_name","")),
        ("Room",           resident.get("room_number","")),
        ("Date",           handover.get("date","")),
        ("Shift Ending",   handover.get("shift_ending","")),
        ("Shift Starting", handover.get("shift_starting","")),
        ("Compiled By",    handover.get("compiled_by","")),
        ("Fluid Target Met", handover.get("fluid_target_met","No")),
        ("Incidents",      str(handover.get("incidents_this_shift","0"))),
        ("Escalation Required", handover.get("escalation_required","No")),
        ("Family Contact", handover.get("family_contact","No")),
    ]))
    els.append(Spacer(1, 6))

    els += _narrative_block(handover.get("overall_summary",""), "Overall Shift Summary")
    els += _narrative_block(handover.get("care_completed",""), "Care Completed")
    if handover.get("concerns_next_shift"):
        els += _narrative_block(handover["concerns_next_shift"], "Concerns for Next Shift")
    if handover.get("outstanding_tasks"):
        els += _narrative_block(handover["outstanding_tasks"], "Outstanding Tasks")
    if handover.get("medication_notes"):
        els += _narrative_block(handover["medication_notes"], "Medication Notes")
    if handover.get("escalation_details"):
        els += _narrative_block(handover["escalation_details"], "Escalation Details")
    if handover.get("family_contact_notes"):
        els += _narrative_block(handover["family_contact_notes"], "Family Contact Notes")

    if handover.get("ai_generated"):
        els.append(Paragraph(
            "⚡ Narrative generated with AI assistance — reviewed and approved by staff.",
            st["small"]
        ))
        els.append(Spacer(1, 6))

    els.append(Paragraph("Staff Authorisation", st["h2"]))
    els.append(_signature_block(handover.get("compiled_by",""), handover.get("created","")))

    doc.build(els, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def generate_wellbeing_pdf(assessment: dict, resident: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    st = _styles()
    els = []
    els.append(_synthetic_banner())
    els.append(Spacer(1, 0.2*cm))

    els.append(_header_table(
        "Wellbeing Assessment",
        f"{resident.get('preferred_name','')} — {assessment.get('assessment_date','')} — "
        f"Period: {assessment.get('period_covered','')}"
    ))
    els.append(Spacer(1, 8))

    els.append(_kv_table([
        ("Resident",        resident.get("full_name","")),
        ("Room",            resident.get("room_number","")),
        ("Assessment Date", assessment.get("assessment_date","")),
        ("Assessed By",     assessment.get("assessed_by","")),
        ("Period Covered",  assessment.get("period_covered","")),
        ("Overall Score",   f"{assessment.get('overall_score','')}/10"),
    ]))
    els.append(Spacer(1, 6))

    # Score table
    els.append(Paragraph("Wellbeing Scores (out of 10)", st["h2"]))
    score_rows = [
        [Paragraph("Domain", st["label"]),
         Paragraph("Score", st["label"]),
         Paragraph("Interpretation", st["label"])],
    ]
    domains = [
        ("Physical Health",    assessment.get("physical_health_score")),
        ("Mental Health",      assessment.get("mental_health_score")),
        ("Social Engagement",  assessment.get("social_engagement_score")),
        ("Personal Care",      assessment.get("personal_care_score")),
        ("Nutrition",          assessment.get("nutrition_score")),
        ("Pain Management",    assessment.get("pain_management_score")),
    ]
    for domain, score in domains:
        s = int(score) if score else 0
        interp = "Good" if s >= 7 else ("Moderate" if s >= 4 else "Needs Attention")
        c = GREEN if s >= 7 else (AMBER if s >= 4 else RED)
        score_rows.append([
            Paragraph(domain, st["value"]),
            Paragraph(str(score) if score else "—", st["value"]),
            Paragraph(interp, ParagraphStyle("ri", fontName="Helvetica-Bold",
                                              fontSize=9, textColor=c)),
        ])
    st2 = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BRAND_BLUE),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, GRAY_LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.25, GRAY_MID),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ])
    t = Table(score_rows, colWidths=[7*cm, 3*cm, 8*cm])
    t.setStyle(st2)
    els.append(t)
    els.append(Spacer(1, 8))

    if assessment.get("summary"):
        els += _narrative_block(assessment["summary"], "Overall Summary")
    if assessment.get("concerns"):
        els += _narrative_block(assessment["concerns"], "Concerns This Period")
    if assessment.get("positive_outcomes"):
        els += _narrative_block(assessment["positive_outcomes"], "Positive Outcomes")
    if assessment.get("actions_next_period"):
        els += _narrative_block(assessment["actions_next_period"], "Actions for Next Period")
    if assessment.get("resident_voice"):
        els += _narrative_block(assessment["resident_voice"], "Resident's Voice")
    if assessment.get("family_feedback"):
        els += _narrative_block(assessment["family_feedback"], "Family Feedback")

    if assessment.get("ai_generated"):
        els.append(Paragraph(
            "⚡ Summary generated with AI assistance — reviewed and approved by staff.",
            st["small"]
        ))
        els.append(Spacer(1, 6))

    els.append(Paragraph("Staff Authorisation", st["h2"]))
    els.append(_signature_block(assessment.get("assessed_by",""), assessment.get("created","")))

    doc.build(els, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def generate_compliance_pdf(residents: list, stats: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    st = _styles()
    els = []
    els.append(_synthetic_banner())
    els.append(Spacer(1, 0.2*cm))

    els.append(_header_table(
        "Compliance & Quality Dashboard",
        f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}"
    ))
    els.append(Spacer(1, 8))

    # Summary stats
    els.append(Paragraph("Home Summary", st["h2"]))
    els.append(_kv_table([
        ("Total Residents",    str(stats.get("total_residents", 0))),
        ("Active Residents",   str(stats.get("active_residents", 0))),
        ("Open Incidents",     str(stats.get("open_incidents", 0))),
        ("Notes This Week",    str(stats.get("notes_this_week", 0))),
        ("Pending Approvals",  str(stats.get("pending_approvals", 0))),
        ("Overdue Reviews",    str(stats.get("overdue_reviews", 0))),
    ]))
    els.append(Spacer(1, 8))

    # Resident compliance table
    els.append(Paragraph("Resident Compliance Status", st["h2"]))
    headers = ["Resident", "Room", "Care Type", "Notes (7d)", "Open Inc.", "Status"]
    rows = [[Paragraph(h, ParagraphStyle("th", fontName="Helvetica-Bold",
                                          fontSize=8, textColor=WHITE))
             for h in headers]]
    for r in residents:
        status = r.get("rag_status", "green")
        sc = _rag_color(status)
        rows.append([
            Paragraph(r.get("full_name",""), st["small"]),
            Paragraph(str(r.get("room_number","")), st["small"]),
            Paragraph(str(r.get("care_type","")), st["small"]),
            Paragraph(str(r.get("notes_count","0")), st["small"]),
            Paragraph(str(r.get("open_incidents","0")), st["small"]),
            Paragraph(status.title(),
                      ParagraphStyle("sc", fontName="Helvetica-Bold",
                                     fontSize=8, textColor=sc)),
        ])
    t = Table(rows, colWidths=[5*cm, 2.5*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BRAND_BLUE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, GRAY_LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.25, GRAY_MID),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ]))
    els.append(t)

    doc.build(els, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def report_available() -> bool:
    return REPORTLAB_OK
