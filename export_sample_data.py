"""Export a small CSV sample of the synthetic cohort for browsing on GitHub.

The full database (carehome.db, ~19 MB) is a build output and is not committed;
`python setup_project.py` regenerates it deterministically from seed 4242.
This script writes a lightweight, human-readable slice to data/sample/ so a
reader can see the shape of the data without running anything.

Every record is synthetic. No real resident, staff member or care home is
represented. The users table (password hashes) is never exported.

    python export_sample_data.py
"""
import csv
import os
import sqlite3

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carehome.db")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample")

# Tables are exported for the first SAMPLE_RESIDENTS residents only, so every
# row in every file joins back to residents.csv.
SAMPLE_RESIDENTS = 5
TABLES = ["residents", "care_plans", "risk_assessments", "incidents",
          "medications", "wellbeing", "care_notes", "handovers", "family_comms"]


def main():
    if not os.path.exists(DB):
        raise SystemExit("carehome.db not found. Run: python setup_project.py --skip-train")
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(DB)
    ids = [f"RES{i:03d}" for i in range(1, SAMPLE_RESIDENTS + 1)]
    marks = ",".join("?" * len(ids))
    for table in TABLES:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        if "resident_id" in cols:
            q = f"SELECT * FROM {table} WHERE resident_id IN ({marks}) ORDER BY rowid"
            rows = con.execute(q, ids).fetchall()
        else:
            rows = con.execute(f"SELECT * FROM {table} ORDER BY rowid LIMIT 200").fetchall()
        path = os.path.join(OUT, f"{table}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        print(f"  {table:<18} {len(rows):>6} rows -> {os.path.relpath(path)}")
    con.close()


if __name__ == "__main__":
    main()
