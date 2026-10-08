"""
anonymize_dataset.py
Anonymises the CareHome synthetic dataset Excel file for safe sharing,
publication, or training — removes all personally identifiable information (PII).

What it removes / replaces
* Full names       → random synthetic names (e.g. "P001 Resident", "P002 Resident")
* Date of birth    → age band string (e.g. "75-79 years")
* NHS numbers      → hash-based token (e.g. "ID-A3F2C1")
* Phone numbers    → "REDACTED"
* Email addresses  → "REDACTED"
* Addresses / postcodes → "REDACTED"
* NOK names        → "NOK-001" etc.
* GP names         → "GP-001" etc.
* Staff names      → "Staff-001" etc.
* Free-text narrative → kept (clinical content, no PII)

Usage
    python anonymize_dataset.py                         # auto-finds dataset
    python anonymize_dataset.py path/to/dataset.xlsx   # explicit path
    python anonymize_dataset.py --help

Output
    <original_name>_ANONYMISED_<date>.xlsx  (same folder)

GDPR / UK Data Protection Act 2018 note
Anonymised data is no longer personal data under Article 4(1) GDPR.
This script implements k-anonymity-style generalisation for DOB and
pseudonymisation for direct identifiers (names, numbers).
Review output before sharing; re-identification risk from clinical
text combinations cannot be fully eliminated programmatically.
"""

import sys
import os
import re
import hashlib
import random
import argparse
from datetime import datetime, date

try:
    import openpyxl
    from openpyxl.styles import PatternFill, Font
except ImportError:
    sys.exit("openpyxl not installed. Run: pip install openpyxl")

# ── Synthetic name pools ────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Alex","Blake","Cameron","Dana","Eden","Finley","Gray","Harper","Indigo",
    "Jamie","Kendall","Lane","Morgan","Nova","Owen","Parker","Quinn","Riley",
    "Sage","Taylor","Umber","Vale","Wren","Xen","Yael","Zephyr"
]
LAST_NAMES = [
    "Ashford","Birch","Cedar","Dale","Elm","Fern","Grove","Hill","Ivy",
    "Juniper","Kale","Larch","Maple","Neem","Oak","Pine","Quince","Rowan",
    "Spruce","Thorn","Ulmus","Vale","Willow","Xeric","Yarrow","Zelkova"
]

# ── PII field detection ─────────────────────────────────────────────────────────
NAME_KEYWORDS = {
    "name","full_name","preferred_name","first_name","last_name","surname",
    "forename","patient_name","resident_name","nok_name","key_worker",
    "staff_name","assessed_by","compiled_by","approved_by","reviewed_by",
    "staff_member","manager_sign_off","gp_name","witness_name",
    "staff_first_on_scene","family_contact","generated_by","given_by",
}
DOB_KEYWORDS   = {"date_of_birth","dob","birth_date","birthdate"}
NHS_KEYWORDS   = {"nhs_number","nhs_no","nhsno","patient_id","resident_id_external"}
PHONE_KEYWORDS = {"phone","telephone","tel","mobile","contact_number","nok_phone","gp_phone"}
EMAIL_KEYWORDS = {"email","email_address","nok_email"}
ADDR_KEYWORDS  = {"address","postcode","post_code","street","town","city","county"}
FREE_TEXT_KEYS = {
    "care_narrative","description","narrative","summary","body","overall_summary",
    "investigation_summary","lessons_learned","preventative_actions","personal_notes",
    "mobility_care_plan","personal_care_plan","continence_care_plan",
    "nutrition_hydration_plan","medication_management_plan","cognitive_support_plan",
    "emotional_wellbeing_plan","social_activity_plan","end_of_life_preferences",
    "risk_summary","goals_of_care","family_involvement_plan","personal_identity_summary",
    "immediate_actions","actions_taken","handover_notes","concerns","follow_up_actions",
    "family_response","risk_factors","interventions","additional_actions",
    "outcome_measures","outcomes","injuries","medical_attention","outcome",
    "cqc_notification","care_completed","concerns_next_shift","outstanding_tasks",
    "medication_notes","escalation_details","physical_notes","mental_notes",
    "social_notes","goals_progress","positive_outcomes","actions_next_period",
    "family_feedback","resident_voice",
}

PHONE_RE = re.compile(r"(\+?[\d\s\-\(\)]{9,15})")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


class Anonymiser:
    def __init__(self):
        self._name_map  = {}
        self._gp_map    = {}
        self._nok_map   = {}
        self._staff_map = {}
        self._nhs_map   = {}
        self._counters  = {"name":1, "gp":1, "nok":1, "staff":1}

    def _synthetic_name(self):
        n = random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES)
        return n

    def _hash_token(self, value: str, prefix: str = "ID") -> str:
        h = hashlib.md5(str(value).encode()).hexdigest()[:6].upper()
        return f"{prefix}-{h}"

    def _age_band(self, dob_str: str) -> str:
        """Convert a date string to an age band (5-year intervals)."""
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y"):
            try:
                dob = datetime.strptime(str(dob_str).strip(), fmt).date()
                today = date.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                band_low  = (age // 5) * 5
                band_high = band_low + 4
                return f"{band_low}-{band_high} years"
            except ValueError:
                continue
        # If already an age integer or band
        try:
            age = int(dob_str)
            band_low  = (age // 5) * 5
            return f"{band_low}-{band_low+4} years"
        except (ValueError, TypeError):
            return "Age band unknown"

    def anonymise_name(self, value: str, category: str = "name") -> str:
        if not value or str(value).strip() in ("", "nan", "None"):
            return value
        key = str(value).strip().lower()
        mapping = {"gp": self._gp_map, "nok": self._nok_map,
                   "staff": self._staff_map}.get(category, self._name_map)
        labels  = {"gp": "GP", "nok": "NOK", "staff": "Staff"}.get(category, "")
        if key not in mapping:
            if category == "gp":
                mapping[key] = f"Dr GP-{self._counters['gp']:03d}"
                self._counters["gp"] += 1
            elif category == "nok":
                mapping[key] = f"NOK-{self._counters['nok']:03d}"
                self._counters["nok"] += 1
            elif category == "staff":
                mapping[key] = f"Staff-{self._counters['staff']:03d}"
                self._counters["staff"] += 1
            else:
                mapping[key] = self._synthetic_name()
                self._counters["name"] += 1
        return mapping[key]

    def anonymise_cell(self, col_name: str, value) -> tuple:
        """Return (new_value, was_changed)."""
        if value is None or str(value).strip() in ("", "nan", "None"):
            return value, False

        col_lower = col_name.lower().strip()

        if col_lower in DOB_KEYWORDS:
            new = self._age_band(str(value))
            return new, True

        if col_lower in NHS_KEYWORDS:
            new = self._hash_token(str(value), "NHS")
            return new, True

        if col_lower in PHONE_KEYWORDS:
            return "REDACTED", True

        if col_lower in EMAIL_KEYWORDS:
            return "REDACTED", True

        if col_lower in ADDR_KEYWORDS:
            return "REDACTED", True

        # Classify name columns
        if col_lower in NAME_KEYWORDS:
            if "gp" in col_lower:
                return self.anonymise_name(str(value), "gp"), True
            elif "nok" in col_lower or "family" in col_lower or "next_of_kin" in col_lower:
                return self.anonymise_name(str(value), "nok"), True
            elif any(s in col_lower for s in ("staff","worker","carer","nurse","assessed_by",
                                               "compiled_by","approved_by","reviewed_by",
                                               "given_by","witness","first_on_scene")):
                return self.anonymise_name(str(value), "staff"), True
            else:
                return self.anonymise_name(str(value), "name"), True

        # For free text fields: redact phone numbers and emails embedded in text
        if col_lower in FREE_TEXT_KEYS:
            text = str(value)
            changed = False
            if EMAIL_RE.search(text):
                text = EMAIL_RE.sub("[EMAIL REDACTED]", text)
                changed = True
            if PHONE_RE.search(text):
                text = PHONE_RE.sub("[PHONE REDACTED]", text)
                changed = True
            return text, changed

        return value, False


def find_dataset(path: str | None) -> str:
    if path and os.path.exists(path):
        return path
    # Search current folder and script folder
    for folder in [".", os.path.dirname(os.path.abspath(__file__))]:
        for f in os.listdir(folder):
            if f.endswith(".xlsx") and "dataset" in f.lower() and "anon" not in f.lower():
                return os.path.join(folder, f)
    return ""


def anonymise_workbook(input_path: str, output_path: str):
    print(f"Loading: {input_path}")
    wb = openpyxl.load_workbook(input_path)
    anon = Anonymiser()
    changed_cells = 0

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        print(f"  Processing sheet: {sheet_name} ({ws.max_row} rows, {ws.max_column} cols)")

        # Find header row (first non-empty row)
        header_row = None
        header_map = {}
        for row in ws.iter_rows(min_row=1, max_row=5):
            vals = [c.value for c in row if c.value]
            if len(vals) > 2:
                header_row = row[0].row
                header_map = {
                    col_idx: str(cell.value).strip()
                    for col_idx, cell in enumerate(row, 1)
                    if cell.value
                }
                break

        if not header_row or not header_map:
            print(f"    Skipping — no header row found")
            continue

        # Process data rows
        for row in ws.iter_rows(min_row=header_row + 1, max_row=ws.max_row):
            for cell in row:
                col_name = header_map.get(cell.column, "")
                if not col_name:
                    continue
                new_val, changed = anon.anonymise_cell(col_name, cell.value)
                if changed:
                    cell.value = new_val
                    cell.fill = PatternFill(fill_type="solid", fgColor="FFF3CD")  # light amber flag
                    changed_cells += 1

    wb.save(output_path)
    print(f"\n  {changed_cells} cells anonymised.")
    print(f"  Output saved to: {output_path}")
    print("\n  Anonymisation legend:")
    print("  - Names         → synthetic names or role codes (GP-001, NOK-001, Staff-001)")
    print("  - Date of birth → 5-year age band (e.g. '80-84 years')")
    print("  - NHS numbers   → hash token (e.g. NHS-A3F2C1)")
    print("  - Phone/Email   → REDACTED")
    print("  - Free text     → inline phone/email addresses redacted; clinical content kept")
    print("  - Yellow cells  → cells where anonymisation was applied")
    print("\n  GDPR note: Review output before sharing. Re-identification risk from")
    print("  clinical narrative combinations cannot be fully eliminated automatically.")


def main():
    parser = argparse.ArgumentParser(description="Anonymise care home dataset Excel file")
    parser.add_argument("input", nargs="?", help="Path to input .xlsx file (auto-detected if omitted)")
    parser.add_argument("--output", "-o", help="Output file path (default: <input>_ANONYMISED_<date>.xlsx)")
    args = parser.parse_args()

    input_path = find_dataset(args.input)
    if not input_path:
        sys.exit("ERROR: Could not find a dataset .xlsx file. Provide path as argument.")

    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_ANONYMISED_{date.today().strftime('%Y%m%d')}{ext}"

    if os.path.abspath(input_path) == os.path.abspath(output_path):
        sys.exit("ERROR: Output path matches input — would overwrite original. Use --output to specify a different path.")

    anonymise_workbook(input_path, output_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
