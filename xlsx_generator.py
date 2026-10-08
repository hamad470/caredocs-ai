"""
XLSX Report Generator — matches dataset column structure exactly.
Produces styled Excel workbooks for all 9 report types.

Columns match the synthetic dataset sheets:
  02_Care_Notes, 03_Incidents, 04_Care_Plans, 05_Risk_Assessments,
  06_Medications, 07_Handovers, 08_Family_Comms, 09_Wellbeing, 10_Compliance
"""
import io
from datetime import datetime, date

from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter
try:
    from openpyxl.styles.numbers import FORMAT_DATE_DDMMYY
except ImportError:
    FORMAT_DATE_DDMMYY = "DD/MM/YY"
OPENPYXL_OK = True


# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY    = "1E3A5F"
BLUE    = "2563EB"
LBLUE   = "DBEAFE"
WHITE   = "FFFFFF"
LGRAY   = "F1F5F9"
MGRAY   = "CBD5E1"
DGRAY   = "475569"
GREEN   = "16A34A"
LGREEN  = "DCFCE7"
AMBER   = "D97706"
LAMBER  = "FEF9C3"
RED_C   = "DC2626"
LRED    = "FEE2E2"
PURPLE  = "7C3AED"
LPURPLE = "EDE9FE"


def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def _font(bold=False, color=None, size=10, italic=False):
    return Font(name="Calibri", bold=bold, color=color or "000000",
                size=size, italic=italic)

def _border(style="thin"):
    s = Side(style=style, color=MGRAY)
    return Border(left=s, right=s, top=s, bottom=s)

def _align(h="left", v="top", wrap=True):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)


def _write_header_block(ws, title, subtitle, home="Sunrise Care Home"):
    """Rows 1-3: branded title block."""
    ws.row_dimensions[1].height = 28
    ws.row_dimensions[2].height = 18
    ws.row_dimensions[3].height = 14

    ws["A1"] = home
    ws["A1"].font = _font(bold=True, color=WHITE, size=14)
    ws["A1"].fill = _fill(NAVY)
    ws["A1"].alignment = _align("left", "center", False)

    ws["A2"] = title
    ws["A2"].font = _font(bold=True, color=WHITE, size=11)
    ws["A2"].fill = _fill(BLUE)
    ws["A2"].alignment = _align("left", "center", False)

    ws["A3"] = f"{subtitle}     Generated: {datetime.now().strftime('%d %b %Y %H:%M')}"
    ws["A3"].font = _font(color=DGRAY, size=9, italic=True)
    ws["A3"].fill = _fill(LGRAY)
    ws["A3"].alignment = _align("left", "center", False)


def _write_col_headers(ws, row, columns, bg=NAVY):
    """Write a styled column header row."""
    ws.row_dimensions[row].height = 20
    for col_idx, col_name in enumerate(columns, 1):
        cell = ws.cell(row=row, column=col_idx, value=col_name)
        cell.font = _font(bold=True, color=WHITE, size=9)
        cell.fill = _fill(bg)
        cell.alignment = _align("center", "center", True)
        cell.border = _border()


def _write_data_row(ws, row_idx, values, row_num=0):
    """Write a data row with alternating background."""
    bg = LGRAY if row_num % 2 == 0 else WHITE
    ws.row_dimensions[row_idx].height = 60
    for col_idx, val in enumerate(values, 1):
        cell = ws.cell(row=row_idx, column=col_idx, value=str(val) if val is not None else "")
        cell.font = _font(size=9)
        cell.fill = _fill(bg)
        cell.alignment = _align("left", "top", True)
        cell.border = _border()


def _rag_cell(cell, status):
    s = str(status).lower()
    if s in ("red", "overdue", "high", "very high", "open", "critical", "major"):
        cell.fill = _fill(LRED)
        cell.font = _font(bold=True, color=RED_C, size=9)
    elif s in ("amber", "due soon", "medium", "pending", "moderate"):
        cell.fill = _fill(LAMBER)
        cell.font = _font(bold=True, color=AMBER, size=9)
    elif s in ("green", "current", "low", "closed", "approved", "active"):
        cell.fill = _fill(LGREEN)
        cell.font = _font(bold=True, color=GREEN, size=9)


def _set_col_widths(ws, widths):
    for col_idx, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = w


def _merge_header(ws, max_col):
    """Merge title rows across all columns."""
    for row in [1, 2, 3]:
        ws.merge_cells(start_row=row, start_column=1,
                       end_row=row, end_column=max_col)


def _ai_note(ws, row, max_col, ai_used):
    if ai_used:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=max_col)
        c = ws.cell(row=row, column=1,
                    value="★ AI-assisted narrative — reviewed and approved by staff")
        c.font = _font(bold=True, color=PURPLE, size=9)
        c.fill = _fill(LPURPLE)
        c.alignment = _align("center", "center", False)


# 02_Care_Notes
CARE_NOTE_COLS = [
    "note_id", "resident_id", "date", "shift", "staff_name", "staff_role",
    "note_type", "personal_care_completed", "mood_observed", "appetite",
    "fluid_intake_ml", "weight_kg", "skin_checked", "skin_concern_noted",
    "repositioned", "activity_participated", "activity_description",
    "sleep_quality", "pain_observed", "pain_location", "falls_this_shift",
    "care_note_narrative [RAG CHUNK]", "concerns_flagged [RAG CHUNK]",
    "actions_taken [RAG CHUNK]", "handover_notes [RAG CHUNK]",
    "ai_generated", "approved_by", "approved_at"
]

def generate_care_note_xlsx(notes: list[dict], resident: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "02_Care_Notes"
    ws.sheet_view.showGridLines = False

    _write_header_block(ws,
        "Care Notes — 02_Care_Notes",
        f"Resident: {resident.get('full_name','')} | Room {resident.get('room_number','')} | {resident.get('primary_diagnosis','')}"
    )
    _merge_header(ws, len(CARE_NOTE_COLS))

    _write_col_headers(ws, 4, CARE_NOTE_COLS)

    for i, note in enumerate(notes):
        row = 5 + i
        vals = [
            f"CN{str(note.get('id',i+1)).zfill(4)}",
            note.get("resident_id",""),
            note.get("date",""),
            note.get("shift",""),
            note.get("staff_name",""),
            note.get("staff_role",""),
            note.get("note_type",""),
            note.get("personal_care",""),
            note.get("mood",""),
            note.get("appetite",""),
            note.get("fluid_intake_ml",""),
            note.get("weight_kg",""),
            note.get("skin_checked",""),
            note.get("skin_concern",""),
            note.get("repositioned",""),
            note.get("activity",""),
            note.get("activity_description",""),
            note.get("sleep_quality",""),
            note.get("pain_observed",""),
            note.get("pain_location",""),
            note.get("falls_this_shift", 0),
            note.get("care_narrative",""),
            note.get("concerns",""),
            note.get("actions_taken",""),
            note.get("handover_notes",""),
            "Yes" if note.get("ai_generated") else "No",
            note.get("approved_by",""),
            note.get("approved_at",""),
        ]
        _write_data_row(ws, row, vals, i)

        # Colour status column (approved_by col = 27)
        status_cell = ws.cell(row=row, column=26)
        ai_val = "Yes" if note.get("ai_generated") else "No"
        if ai_val == "Yes":
            status_cell.fill = _fill(LPURPLE)
            status_cell.font = _font(bold=True, color=PURPLE, size=9)

    _set_col_widths(ws, [
        8,8,12,25,20,18,18,18,14,12,
        12,10,12,20,12,14,30,14,16,14,
        12,55,40,40,40,10,20,22
    ])

    # Freeze panes below header + col headers
    ws.freeze_panes = "A5"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# 03_Incidents
INCIDENT_COLS = [
    "incident_id", "resident_id", "date", "time", "shift",
    "incident_type", "severity", "location", "witnessed", "witness_name",
    "staff_first_on_scene", "incident_description [RAG CHUNK]",
    "immediate_actions [RAG CHUNK]", "injuries_sustained", "medical_attention",
    "outcome [RAG CHUNK]", "gp_notified", "gp_notified_time",
    "family_notified", "family_notified_time", "family_notified_by",
    "family_response [RAG CHUNK]", "local_authority_notified",
    "cqc_notification_required", "cqc_notified", "riddor_reportable",
    "risk_assessment_updated", "care_plan_updated", "investigation_required",
    "investigation_summary [RAG CHUNK]", "lessons_learned [RAG CHUNK]",
    "preventative_actions [RAG CHUNK]", "manager_sign_off", "manager_sign_off_date"
]

def generate_incident_xlsx(incidents: list[dict], residents: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "03_Incidents"
    ws.sheet_view.showGridLines = False

    _write_header_block(ws, "Incident Reports — 03_Incidents",
                        f"Exported: {date.today().strftime('%d %b %Y')}")
    _merge_header(ws, len(INCIDENT_COLS))
    _write_col_headers(ws, 4, INCIDENT_COLS)

    for i, inc in enumerate(incidents):
        row = 5 + i
        vals = [
            inc.get("incident_id",""),
            inc.get("resident_id",""),
            inc.get("date",""),
            inc.get("time",""),
            inc.get("shift",""),
            inc.get("incident_type",""),
            inc.get("severity",""),
            inc.get("location",""),
            inc.get("witnessed",""),
            inc.get("witness_name",""),
            inc.get("staff_first_on_scene",""),
            inc.get("description",""),
            inc.get("immediate_actions",""),
            inc.get("injuries",""),
            inc.get("medical_attention",""),
            inc.get("outcome",""),
            inc.get("gp_notified",""),
            "",  # gp_notified_time
            inc.get("family_notified",""),
            "",  # family_notified_time
            "",  # family_notified_by
            "",  # family_response
            "",  # local_authority_notified
            inc.get("cqc_notification",""),
            "",  # cqc_notified
            "",  # riddor_reportable
            inc.get("risk_assessment_updated",""),
            inc.get("care_plan_updated",""),
            inc.get("investigation_required",""),
            inc.get("investigation_summary",""),
            inc.get("lessons_learned",""),
            inc.get("preventative_actions",""),
            inc.get("manager_sign_off",""),
            inc.get("manager_sign_off_date",""),
        ]
        _write_data_row(ws, row, vals, i)

        # RAG colour on severity (col 7)
        _rag_cell(ws.cell(row=row, column=7), inc.get("severity",""))
        # Status colour on manager_sign_off (col 33)
        status = "closed" if inc.get("manager_sign_off") else "open"
        _rag_cell(ws.cell(row=row, column=33), status)

    _set_col_widths(ws, [
        12,8,12,8,22,18,12,16,10,20,
        20,55,45,25,18,45,12,12,14,14,
        20,45,20,20,12,12,18,18,18,50,
        50,50,22,14
    ])
    ws.freeze_panes = "A5"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# 04_Care_Plans
CARE_PLAN_COLS = [
    "care_plan_id", "resident_id", "version", "effective_from", "review_date",
    "reviewed_by", "status",
    "personal_identity_summary [RAG CHUNK]",
    "mobility_care_plan [RAG CHUNK]",
    "personal_care_plan [RAG CHUNK]",
    "continence_care_plan [RAG CHUNK]",
    "nutrition_hydration_plan [RAG CHUNK]",
    "medication_management_plan [RAG CHUNK]",
    "cognitive_support_plan [RAG CHUNK]",
    "emotional_wellbeing_plan [RAG CHUNK]",
    "social_activity_plan [RAG CHUNK]",
    "end_of_life_preferences [RAG CHUNK]",
    "risk_summary [RAG CHUNK]",
    "goals_of_care [RAG CHUNK]",
    "family_involvement_plan [RAG CHUNK]"
]

def generate_care_plan_xlsx(plans: list[dict], resident: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "04_Care_Plans"
    ws.sheet_view.showGridLines = False

    _write_header_block(ws, "Care Plans — 04_Care_Plans",
        f"Resident: {resident.get('full_name','')} | Room {resident.get('room_number','')}")
    _merge_header(ws, len(CARE_PLAN_COLS))
    _write_col_headers(ws, 4, CARE_PLAN_COLS)

    for i, plan in enumerate(plans):
        row = 5 + i
        ws.row_dimensions[row].height = 120
        vals = [
            f"CP{str(plan.get('id',i+1)).zfill(4)}",
            plan.get("resident_id",""),
            plan.get("version", 1),
            plan.get("effective_from",""),
            plan.get("review_date",""),
            plan.get("reviewed_by",""),
            plan.get("status","Active"),
            plan.get("personal_identity_summary",""),
            plan.get("mobility_care_plan",""),
            plan.get("personal_care_plan",""),
            plan.get("continence_care_plan",""),
            plan.get("nutrition_hydration_plan",""),
            plan.get("medication_management_plan",""),
            plan.get("cognitive_support_plan",""),
            plan.get("emotional_wellbeing_plan",""),
            plan.get("social_activity_plan",""),
            plan.get("end_of_life_preferences",""),
            plan.get("risk_summary",""),
            plan.get("goals_of_care",""),
            plan.get("family_involvement_plan",""),
        ]
        _write_data_row(ws, row, vals, i)
        _rag_cell(ws.cell(row=row, column=7), plan.get("status",""))

    _set_col_widths(ws, [
        10,8,8,14,14,22,12,
        55,55,55,55,55,55,55,55,55,55,55,55,55
    ])
    ws.freeze_panes = "H5"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# 05_Risk_Assessments
RISK_COLS = [
    "assessment_id", "resident_id", "assessment_type", "date_assessed",
    "assessed_by", "review_date", "score", "risk_level",
    "risk_factors_identified [RAG CHUNK]", "current_interventions [RAG CHUNK]",
    "additional_actions [RAG CHUNK]", "outcome_measures [RAG CHUNK]",
    "full_assessment_narrative [RAG CHUNK]"
]

def generate_risk_xlsx(risks: list[dict], resident: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "05_Risk_Assessments"
    ws.sheet_view.showGridLines = False

    _write_header_block(ws, "Risk Assessments — 05_Risk_Assessments",
        f"Resident: {resident.get('full_name','')} | Room {resident.get('room_number','')}")
    _merge_header(ws, len(RISK_COLS))
    _write_col_headers(ws, 4, RISK_COLS)

    for i, risk in enumerate(risks):
        row = 5 + i
        ws.row_dimensions[row].height = 80
        vals = [
            f"RA{str(risk.get('id',i+1)).zfill(4)}",
            risk.get("resident_id",""),
            risk.get("assessment_type",""),
            risk.get("date_assessed",""),
            risk.get("assessed_by",""),
            risk.get("review_date",""),
            risk.get("score",""),
            risk.get("risk_level",""),
            risk.get("risk_factors",""),
            risk.get("interventions",""),
            risk.get("additional_actions",""),
            risk.get("outcome_measures",""),
            risk.get("narrative",""),
        ]
        _write_data_row(ws, row, vals, i)
        _rag_cell(ws.cell(row=row, column=8), risk.get("risk_level",""))

    _set_col_widths(ws, [10,8,22,14,22,14,18,12,55,55,45,45,60])
    ws.freeze_panes = "A5"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# 06_Medications
MED_COLS = [
    "med_id", "resident_id", "medication_name", "generic_name", "dose",
    "route", "frequency", "indication", "prescribing_gp",
    "start_date", "review_date", "is_controlled_drug", "is_prn",
    "prn_instructions [RAG CHUNK]", "administration_notes [RAG CHUNK]",
    "side_effects_to_monitor [RAG CHUNK]", "status", "stopped_reason [RAG CHUNK]"
]

def generate_medication_xlsx(meds: list[dict], mar_records: dict, resident: dict) -> bytes:
    wb = Workbook()

    # Sheet 1 — Medication Register
    ws1 = wb.active
    ws1.title = "06_Medications"
    ws1.sheet_view.showGridLines = False

    _write_header_block(ws1, "Medication Register — 06_Medications",
        f"Resident: {resident.get('full_name','')} | Room {resident.get('room_number','')}")
    _merge_header(ws1, len(MED_COLS))
    _write_col_headers(ws1, 4, MED_COLS)

    for i, med in enumerate(meds):
        row = 5 + i
        ws1.row_dimensions[row].height = 60
        vals = [
            f"MED{str(med.get('id',i+1)).zfill(4)}",
            med.get("resident_id",""),
            med.get("medication_name",""),
            med.get("generic_name",""),
            med.get("dose",""),
            med.get("route",""),
            med.get("frequency",""),
            med.get("indication",""),
            med.get("prescribing_gp",""),
            med.get("start_date",""),
            med.get("review_date",""),
            "Yes" if med.get("is_controlled") else "No",
            "Yes" if med.get("is_prn") else "No",
            med.get("prn_instructions",""),
            med.get("admin_notes",""),
            med.get("side_effects",""),
            med.get("status","Active"),
            med.get("stopped_reason",""),
        ]
        _write_data_row(ws1, row, vals, i)
        _rag_cell(ws1.cell(row=row, column=17), med.get("status",""))
        if med.get("is_controlled"):
            ws1.cell(row=row, column=12).fill = _fill(LRED)
            ws1.cell(row=row, column=12).font = _font(bold=True, color=RED_C, size=9)

    _set_col_widths(ws1, [
        10,8,22,22,12,10,28,22,22,12,12,14,8,45,45,45,12,35
    ])
    ws1.freeze_panes = "A5"

    # Sheet 2 — MAR Records
    ws2 = wb.create_sheet("MAR_Records")
    ws2.sheet_view.showGridLines = False
    mar_cols = ["med_id","medication_name","resident_id","date","time_given",
                "shift","given_by","administered","refusal_reason","notes"]
    _write_header_block(ws2, "Medication Administration Records (MAR)",
        f"Resident: {resident.get('full_name','')}")
    ws2.merge_cells(start_row=1,start_column=1,end_row=1,end_column=len(mar_cols))
    ws2.merge_cells(start_row=2,start_column=1,end_row=2,end_column=len(mar_cols))
    ws2.merge_cells(start_row=3,start_column=1,end_row=3,end_column=len(mar_cols))
    _write_col_headers(ws2, 4, mar_cols)

    for i, rec in enumerate(mar_records):
        row = 5 + i
        ws2.row_dimensions[row].height = 30
        vals = [
            f"MED{str(rec.get('medication_id',0)).zfill(4)}",
            rec.get("medication_name",""),
            rec.get("resident_id",""),
            rec.get("date",""),
            rec.get("time_given",""),
            rec.get("shift",""),
            rec.get("given_by",""),
            rec.get("administered",""),
            rec.get("refusal_reason",""),
            rec.get("notes",""),
        ]
        _write_data_row(ws2, row, vals, i)
        admin_cell = ws2.cell(row=row, column=8)
        adm = str(rec.get("administered","")).lower()
        if adm == "yes":
            _rag_cell(admin_cell, "active")
        elif adm in ("refused","no","not_available"):
            _rag_cell(admin_cell, "red")

    _set_col_widths(ws2, [10,22,8,12,10,14,22,16,30,30])
    ws2.freeze_panes = "A5"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# 07_Handovers
HANDOVER_COLS = [
    "handover_id", "date", "shift_ending", "shift_starting", "compiled_by",
    "resident_id", "resident_name", "room_number",
    "overall_shift_summary [RAG CHUNK]", "care_completed [RAG CHUNK]",
    "concerns_for_next_shift [RAG CHUNK]", "outstanding_tasks [RAG CHUNK]",
    "medication_notes [RAG CHUNK]", "fluid_target_met", "incidents_this_shift",
    "incident_reference", "family_contact_this_shift",
    "family_contact_notes [RAG CHUNK]", "escalation_required",
    "escalation_details [RAG CHUNK]"
]

def generate_handover_xlsx(handovers: list[dict], resident: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "07_Handovers"
    ws.sheet_view.showGridLines = False

    _write_header_block(ws, "Shift Handovers — 07_Handovers",
        f"Resident: {resident.get('full_name','')} | Room {resident.get('room_number','')}")
    _merge_header(ws, len(HANDOVER_COLS))
    _write_col_headers(ws, 4, HANDOVER_COLS)

    for i, h in enumerate(handovers):
        row = 5 + i
        ws.row_dimensions[row].height = 70
        vals = [
            f"HO{str(h.get('id',i+1)).zfill(4)}",
            h.get("date",""),
            h.get("shift_ending",""),
            h.get("shift_starting",""),
            h.get("compiled_by",""),
            h.get("resident_id",""),
            resident.get("full_name",""),
            resident.get("room_number",""),
            h.get("overall_summary",""),
            h.get("care_completed",""),
            h.get("concerns_next_shift",""),
            h.get("outstanding_tasks",""),
            h.get("medication_notes",""),
            h.get("fluid_target_met",""),
            h.get("incidents_this_shift", 0),
            h.get("incident_reference",""),
            h.get("family_contact",""),
            h.get("family_contact_notes",""),
            h.get("escalation_required",""),
            h.get("escalation_details",""),
        ]
        _write_data_row(ws, row, vals, i)

        # Fluid target
        fluid = str(h.get("fluid_target_met","")).lower()
        fluid_cell = ws.cell(row=row, column=14)
        _rag_cell(fluid_cell, "active" if "yes" in fluid else "red")

        # Escalation
        esc_cell = ws.cell(row=row, column=19)
        if str(h.get("escalation_required","")).lower() == "yes":
            _rag_cell(esc_cell, "red")

        # AI badge
        if h.get("ai_generated"):
            ws.cell(row=row, column=9).fill = _fill(LPURPLE)

    _set_col_widths(ws, [
        10,12,25,25,22,8,22,8,
        55,50,45,40,40,18,10,16,16,40,16,40
    ])
    ws.freeze_panes = "I5"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# 08_Family_Comms
FAMILY_COMM_COLS = [
    "comm_id", "resident_id", "date", "communication_type", "direction",
    "staff_member", "family_contact", "subject", "trigger_event",
    "communication_body [RAG CHUNK]", "family_response_summary [RAG CHUNK]",
    "follow_up_required", "follow_up_actions [RAG CHUNK]",
    "ai_drafted", "approved_by"
]

def generate_family_comm_xlsx(comms: list[dict], resident: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "08_Family_Comms"
    ws.sheet_view.showGridLines = False

    _write_header_block(ws, "Family Communications — 08_Family_Comms",
        f"Resident: {resident.get('full_name','')} | NOK: {resident.get('nok_name','')}")
    _merge_header(ws, len(FAMILY_COMM_COLS))
    _write_col_headers(ws, 4, FAMILY_COMM_COLS)

    for i, comm in enumerate(comms):
        row = 5 + i
        ws.row_dimensions[row].height = 80
        vals = [
            f"FC{str(comm.get('id',i+1)).zfill(4)}",
            comm.get("resident_id",""),
            comm.get("date",""),
            comm.get("comm_type",""),
            comm.get("direction",""),
            comm.get("staff_member",""),
            comm.get("family_contact",""),
            comm.get("subject",""),
            comm.get("trigger_event",""),
            comm.get("body",""),
            comm.get("family_response",""),
            comm.get("follow_up",""),
            comm.get("follow_up_actions",""),
            "Yes" if comm.get("ai_drafted") else "No",
            comm.get("approved_by",""),
        ]
        _write_data_row(ws, row, vals, i)
        if comm.get("ai_drafted"):
            ws.cell(row=row, column=10).fill = _fill(LPURPLE)
        if str(comm.get("follow_up","")).lower() == "yes":
            _rag_cell(ws.cell(row=row, column=12), "amber")

    _set_col_widths(ws, [
        10,8,12,18,12,22,22,30,22,60,50,14,45,10,22
    ])
    ws.freeze_panes = "A5"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# 09_Wellbeing
WELLBEING_COLS = [
    "assessment_id", "resident_id", "assessment_date", "assessed_by",
    "period_covered", "physical_health_score", "mental_health_score",
    "social_engagement_score", "personal_care_score", "nutrition_score",
    "pain_management_score", "overall_wellbeing_score",
    "physical_health_notes [RAG CHUNK]", "mental_health_notes [RAG CHUNK]",
    "social_engagement_notes [RAG CHUNK]", "goals_progress_notes [RAG CHUNK]",
    "concerns_this_period [RAG CHUNK]", "positive_outcomes [RAG CHUNK]",
    "actions_for_next_period [RAG CHUNK]", "family_feedback [RAG CHUNK]",
    "resident_voice [RAG CHUNK]", "overall_summary [RAG CHUNK]"
]

def generate_wellbeing_xlsx(assessments: list[dict], resident: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "09_Wellbeing"
    ws.sheet_view.showGridLines = False

    _write_header_block(ws, "Wellbeing Assessments — 09_Wellbeing",
        f"Resident: {resident.get('full_name','')} | Room {resident.get('room_number','')}")
    _merge_header(ws, len(WELLBEING_COLS))
    _write_col_headers(ws, 4, WELLBEING_COLS)

    for i, a in enumerate(assessments):
        row = 5 + i
        ws.row_dimensions[row].height = 90
        overall = a.get("overall_score") or a.get("overall_wellbeing_score","")
        vals = [
            f"WB{str(a.get('id',i+1)).zfill(4)}",
            a.get("resident_id",""),
            a.get("assessment_date",""),
            a.get("assessed_by",""),
            a.get("period_covered",""),
            a.get("physical_health_score",""),
            a.get("mental_health_score",""),
            a.get("social_engagement_score",""),
            a.get("personal_care_score",""),
            a.get("nutrition_score",""),
            a.get("pain_management_score",""),
            overall,
            a.get("physical_notes",""),
            a.get("mental_notes",""),
            a.get("social_notes",""),
            a.get("goals_progress",""),
            a.get("concerns",""),
            a.get("positive_outcomes",""),
            a.get("actions_next_period",""),
            a.get("family_feedback",""),
            a.get("resident_voice",""),
            a.get("summary",""),
        ]
        _write_data_row(ws, row, vals, i)

        # Colour score cells (cols 6-12)
        for col in range(6, 13):
            cell = ws.cell(row=row, column=col)
            try:
                score = float(cell.value or 0)
                if score >= 7:
                    cell.fill = _fill(LGREEN); cell.font = _font(bold=True, color=GREEN, size=9)
                elif score >= 4:
                    cell.fill = _fill(LAMBER); cell.font = _font(bold=True, color=AMBER, size=9)
                else:
                    cell.fill = _fill(LRED);   cell.font = _font(bold=True, color=RED_C, size=9)
            except (ValueError, TypeError):
                pass

        if a.get("ai_generated"):
            ws.cell(row=row, column=22).fill = _fill(LPURPLE)

    _set_col_widths(ws, [
        10,8,14,22,16,8,8,10,10,10,10,10,
        50,50,50,45,45,45,45,45,45,55
    ])
    ws.freeze_panes = "M5"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# 10_Compliance
COMPLIANCE_COLS = [
    "resident_id", "resident_name", "room", "care_type",
    "care_plan_due", "care_plan_status",
    "falls_risk_due", "falls_risk_status",
    "pressure_risk_due", "pressure_risk_status",
    "capacity_assessment_due", "capacity_assessment_status",
    "medication_review_due", "medication_review_status",
    "wellbeing_review_due", "wellbeing_review_status",
    "care_notes_last_7_days", "care_notes_target_7_days",
    "documentation_compliance_pct",
    "incidents_last_30_days", "falls_last_90_days",
    "last_incident_date", "rag_status", "compliance_notes"
]

def generate_compliance_xlsx(residents: list[dict], stats: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "10_Compliance"
    ws.sheet_view.showGridLines = False

    _write_header_block(ws, "Compliance Dashboard — 10_Compliance",
        f"Week ending: {date.today().strftime('%d %b %Y')} | Total residents: {stats.get('total_residents',0)}")
    _merge_header(ws, len(COMPLIANCE_COLS))

    # Summary stats block (rows 4-6)
    stats_labels = [
        ("Total Residents", stats.get("total_residents",0), NAVY),
        ("Open Incidents",  stats.get("open_incidents",0),  RED_C),
        ("Pending Approvals", stats.get("pending_approvals",0), AMBER),
        ("Notes This Week", stats.get("notes_this_week",0), GREEN),
    ]
    for col_idx, (label, val, color) in enumerate(stats_labels, 1):
        lc = ws.cell(row=4, column=col_idx, value=label)
        lc.font = _font(bold=True, color=WHITE, size=9)
        lc.fill = _fill(color)
        lc.alignment = _align("center","center",False)
        vc = ws.cell(row=5, column=col_idx, value=val)
        vc.font = _font(bold=True, color=color, size=14)
        vc.alignment = _align("center","center",False)
        vc.border = _border()
    ws.row_dimensions[4].height = 18
    ws.row_dimensions[5].height = 28

    _write_col_headers(ws, 7, COMPLIANCE_COLS, bg=NAVY)

    for i, r in enumerate(residents):
        row = 8 + i
        ws.row_dimensions[row].height = 35
        notes_7d = r.get("notes_count", 0)
        target_7d = 7
        pct = round((notes_7d / target_7d) * 100, 1) if target_7d else 0
        rag = r.get("rag","green")

        # Derive review statuses from data
        def _status(date_str):
            if not date_str or date_str == "Never":
                return "Overdue"
            try:
                from datetime import datetime as dt
                d = dt.strptime(str(date_str)[:10], "%Y-%m-%d").date()
                delta = (d - date.today()).days
                if delta < 0: return "Overdue"
                if delta < 30: return "Due Soon"
                return "Current"
            except Exception:
                return "Unknown"

        vals = [
            r.get("resident_id",""),
            r.get("full_name",""),
            r.get("room_number",""),
            r.get("care_type",""),
            "",  # care_plan_due
            _status(r.get("last_care_plan_review","")),
            "",  # falls_risk_due
            _status(r.get("last_risk","")),
            "",  # pressure_risk_due
            _status(r.get("last_risk","")),
            "",  # capacity_assessment_due
            "Current",
            "",  # medication_review_due
            "Current",
            "",  # wellbeing_review_due
            _status(r.get("last_wellbeing","")),
            notes_7d,
            target_7d,
            f"{pct}%",
            r.get("open_incidents", 0),
            "",  # falls_last_90_days
            r.get("last_note",""),
            rag.title(),
            "",  # compliance_notes
        ]
        _write_data_row(ws, row, vals, i)

        # Colour status cols
        for status_col in [6, 8, 10, 12, 14, 16]:
            _rag_cell(ws.cell(row=row, column=status_col),
                      ws.cell(row=row, column=status_col).value or "")

        # RAG overall
        _rag_cell(ws.cell(row=row, column=23), rag)

        # Compliance % colour
        pct_cell = ws.cell(row=row, column=19)
        try:
            p = float(pct)
            if p >= 100: pct_cell.fill = _fill(LGREEN); pct_cell.font = _font(bold=True, color=GREEN, size=9)
            elif p >= 70: pct_cell.fill = _fill(LAMBER); pct_cell.font = _font(bold=True, color=AMBER, size=9)
            else:         pct_cell.fill = _fill(LRED);   pct_cell.font = _font(bold=True, color=RED_C, size=9)
        except Exception:
            pass

    _set_col_widths(ws, [
        10,22,6,14,
        14,12, 14,12, 14,12, 18,12, 16,12, 16,12,
        12,12,14,
        12,12,14,12,30
    ])
    ws.freeze_panes = "A8"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def xlsx_available() -> bool:
    return OPENPYXL_OK
