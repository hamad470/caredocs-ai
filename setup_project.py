"""
setup_project.py — one command to build the whole project from source

Nothing in this project's data layer is source code that happens to sit in a
repository: the database, the model artefact and the two retrieval indexes are
all BUILD OUTPUTS derived from `synthetic_cohort.py` and a fixed seed. This
script runs the derivation end to end so the whole thing is reproducible from a
clean checkout, and so a reader can verify that the numbers in the dissertation
came from a pipeline rather than from a spreadsheet.

    python setup_project.py                 # build everything that is missing
    python setup_project.py --force         # rebuild everything from scratch
    python setup_project.py --skip-train    # data + indexes only (fast)
    python setup_project.py --experiments   # also re-run every reported experiment

Stages
    1. Generate the synthetic cohort          ~2 s     -> carehome.db
    2. Validate it against published rates    <1 s     -> results/synthetic_data_validation.json
    3. Train and persist the fall-risk model  ~3 min   -> ml_fall_risk_model.pkl
    4. Build the v1 retrieval index           ~7 s     -> rag_index/
    5. Build the v2 hybrid index              ~30 s    -> rag_index_v2/
    6. (--experiments) Oracle ceiling         ~1 min   -> results/oracle_benchmark.json
    7. (--experiments) Retrieval experiments  ~3 min   -> results/rag_experiments.json
    8. (--experiments) Cohort-size experiment ~6 min   -> results/learning_curve.json
    9. (--experiments) Grounding calibration  ~1 min   -> results/grounding_instrument.json

Every stage is idempotent and every stage prints what it produced.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
BAR = "=" * 78


def run(cmd: list[str], label: str) -> bool:
    print(f"\n{BAR}\n  {label}\n{BAR}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable] + cmd, cwd=BASE)
    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"  -> {'OK' if ok else 'FAILED'} in {dt:.1f}s", flush=True)
    return ok


def exists(*parts) -> bool:
    return os.path.exists(os.path.join(BASE, *parts))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the CareHome AI project from source.")
    ap.add_argument("--force", action="store_true", help="rebuild everything")
    ap.add_argument("--skip-train", action="store_true", help="skip model training")
    ap.add_argument("--experiments", action="store_true",
                    help="also re-run the reported experiments")
    ap.add_argument("--residents", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    failures = []

    # ── 1 + 2: data ─────────────────────────────────────────────────────────
    if args.force or not exists("carehome.db"):
        cmd = ["synthetic_cohort.py", "--yes", "--validate",
               "--report", os.path.join("results", "synthetic_data_validation.json")]
        if args.residents:
            cmd += ["--residents", str(args.residents)]
        if not run(cmd, "1/5  Generate and validate the synthetic cohort"):
            failures.append("cohort generation")
    else:
        print(f"{BAR}\n  1/5  carehome.db already present — skipping "
              f"(use --force to rebuild)\n{BAR}")

    # ── 3: model ────────────────────────────────────────────────────────────
    if not args.skip_train and (args.force or not exists("ml_fall_risk_model.pkl")):
        if not run(["train_model.py"], "2/5  Train and persist the fall-risk model"):
            failures.append("model training")
    else:
        print(f"{BAR}\n  2/5  model artefact present (or --skip-train) — skipping\n{BAR}")

    # ── 4: v1 index ─────────────────────────────────────────────────────────
    if args.force or not exists("rag_index", "tfidf_index.pkl"):
        code = ("import os,json,rag_engine;"
                "r=rag_engine.build_index(os.path.join(os.getcwd(),'carehome.db'));"
                "print(json.dumps({k:v for k,v in r.items() if k!='sample'},"
                "indent=1,default=str))")
        if not run(["-c", code], "3/5  Build the v1 TF-IDF retrieval index"):
            failures.append("v1 index")
    else:
        print(f"{BAR}\n  3/5  v1 index present — skipping\n{BAR}")

    # ── 5: v2 index ─────────────────────────────────────────────────────────
    if args.force or not exists("rag_index_v2", "store.pkl"):
        if not run(["rag_advanced.py"], "4/5  Build the v2 hybrid (BM25 + dense) index"):
            failures.append("v2 index")
    else:
        print(f"{BAR}\n  4/5  v2 index present — skipping\n{BAR}")

    # ── 6 + 7: experiments ──────────────────────────────────────────────────
    if args.experiments:
        if not run(["oracle_benchmark.py", "--json",
                    os.path.join("results", "oracle_benchmark.json")],
                   "5/5  Bayes-optimal ceiling benchmark"):
            failures.append("oracle benchmark")
        if not run(["rag_experiments.py", "--json",
                    os.path.join("results", "rag_experiments.json")],
                   "5/7  Retrieval experiments (ablation, recency sweep, header, isolation)"):
            failures.append("rag experiments")

        # Both of these read artefacts the steps above produced, so they run
        # last. The cohort-size experiment is the slow one: it refits the whole
        # pipeline at eight cohort sizes with six draws at each.
        if not run([os.path.join("experiments", "learning_curve.py")],
                   "6/7  Cohort-size experiment (Section 7.3.1)"):
            failures.append("learning curve")

        if not run([os.path.join("experiments", "grounding_instrument.py")],
                   "7/7  Grounding instrument calibration (Section 7.6 E5)"):
            failures.append("grounding instrument")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{BAR}\n  BUILD SUMMARY\n{BAR}")
    for label, path in [
        ("Database",        "carehome.db"),
        ("Ground truth",    "carehome_ground_truth.json.gz"),
        ("Model artefact",  "ml_fall_risk_model.pkl"),
        ("v1 index",        os.path.join("rag_index", "tfidf_index.pkl")),
        ("v2 index",        os.path.join("rag_index_v2", "store.pkl")),
    ]:
        full = os.path.join(BASE, path)
        if os.path.exists(full):
            size = os.path.getsize(full) / 1048576
            print(f"  [OK]   {label:<16} {path:<38} {size:8.1f} MB")
        else:
            print(f"  [--]   {label:<16} {path:<38} {'missing':>11}")

    if os.path.isdir(RESULTS):
        for fn in sorted(os.listdir(RESULTS)):
            print(f"  [res]  {'result':<16} results/{fn}")

    if failures:
        print(f"\n  FAILED STAGES: {', '.join(failures)}")
        print(BAR)
        return 1
    print("\n  All stages complete. Start the app with:  python app.py")
    print(BAR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
