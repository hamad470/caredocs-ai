"""
train_model.py — offline trainer for the CareHome fall-risk classifier

Why this file exists
In a professional ML deployment, training and serving are separate lifecycle
stages:

    TRAINING (this script)      run occasionally, by a human, deliberately.
                                Expensive: builds the supervised dataset,
                                runs 5-fold and leave-one-subject-out
                                cross-validation, computes permutation
                                importance, fits the final estimator, and
                                writes ONE versioned artefact to disk.

    SERVING (the Flask app)     runs on every page load. Cheap: loads that
                                artefact once per process and calls
                                predict_proba(). Never fits a model inside a
                                web request.

Keeping them separate gives three properties the previous design lacked:

  1. Reproducibility — the deployed model is a fixed, inspectable file, not
     something re-derived (and potentially re-derived differently) each time
     a page is opened.
  2. Auditability — the artefact records when it was trained, on how much
     data, with which library versions, at which decision threshold, and what
     it scored in validation. In a care setting you must be able to answer
     "which model produced this alert, and how good was it?"
  3. Latency — /ml-evaluation and /risk-ranking respond immediately instead of
     re-running cross-validation for every visitor.

Usage
    python train_model.py                 # train, evaluate, save artefact
    python train_model.py --dry-run       # evaluate only, do NOT overwrite
    python train_model.py --status        # show the current artefact's card
    python train_model.py --json          # full metrics as JSON (for the report)

Exit codes
    0  success
    1  training failed (e.g. not enough data — run seed_data.py first)
"""

import argparse
import json
import sys
from datetime import datetime

import ml_models


BAR = "═" * 74


def _fmt(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def print_model_card(status: dict) -> None:
    """Human-readable summary of the artefact currently on disk."""
    print(BAR)
    print(" MODEL CARD — persisted artefact")
    print(BAR)
    if not status.get("available"):
        print(f" Status        : NO USABLE MODEL")
        print(f" Reason        : {status.get('message')}")
        print(BAR)
        return

    rng = status.get("train_date_range") or [None, None]
    print(f" File          : {status.get('filename')}  ({status.get('size_kb')} KB)")
    print(f" Location      : {status.get('path')}")
    print(f" Artefact ver. : v{status.get('artefact_version')}")
    print(f" Estimator     : {status.get('model_type')}")
    print(f" Trained at    : {status.get('trained_at')}")
    print(f" Training rows : {status.get('n_samples')} "
          f"({status.get('n_positive')} positive) "
          f"from {status.get('n_residents')} residents")
    print(f" Window covered: {rng[0]} → {rng[1]}")
    print(f" Threshold     : {status.get('threshold')} "
          f"(recall-weighted; below 0.5 on purpose)")
    print(f" Environment   : Python {status.get('python_version')}, "
          f"scikit-learn {status.get('sklearn_version')}, "
          f"numpy {status.get('numpy_version')}")
    if status.get("stale"):
        print(f" Freshness     : STALE — {'; '.join(status.get('stale_reasons', []))}")
    else:
        print(f" Freshness     : current (database unchanged since training)")
    print(BAR)


def print_metrics(results: dict) -> None:
    ds = results.get("dataset", {})
    print()
    print(BAR)
    print(" TRAINING RESULTS  (v3 pipeline)")
    print(BAR)
    print(f" Dataset       : {ds.get('n_samples')} windows, "
          f"{ds.get('n_features')} features, "
          f"{ds.get('n_positive')} positive / {ds.get('n_negative')} negative, "
          f"{ds.get('n_residents')} residents")
    print(f" Class balance : {ds.get('positive_rate')} positive; "
          f"events per variable {ds.get('events_per_variable')}")

    hp = results.get("hyperparameters") or {}
    if hp.get("selected"):
        print(f" Hyperparams   : {hp['selected']}  "
              f"(by {hp.get('method', 'n/a')})")
        sp = hp.get("spread_auc") or {}
        if sp:
            print(f"                 search spread in AUC: {sp}")

    print()
    header = (f" {'Model':<38}{'ROC-AUC':>9}{'Avg.P':>8}{'Recall':>8}"
              f"{'Prec.':>8}{'F2':>8}")
    print(header)
    print(" " + "-" * (len(header) - 1))

    families = ["logistic_regression", "random_forest", "gradient_boosting"]
    schemes = [("", "row-shuffled 5-fold  [leaks: exhibit]"),
               ("_groupkfold", "grouped 5-fold"),
               ("_loso", "leave-one-subject-out")]
    for suffix, label in schemes:
        print(f"  -- {label} --")
        for fam in families:
            m = results.get(f"{fam}{suffix}")
            if not m:
                continue
            name = fam.replace("_", " ").title()
            print(f" {name:<38}{_fmt(m.get('roc_auc')):>9}"
                  f"{_fmt(m.get('average_precision')):>8}{_fmt(m.get('recall')):>8}"
                  f"{_fmt(m.get('precision')):>8}{_fmt(m.get('f2_score')):>8}")
    print(f"  -- baselines --")
    for key, name in (("baseline_majority", "Majority class"),
                      ("baseline_static_risk_score", "Existing paper risk score")):
        m = results.get(key)
        if m:
            print(f" {name:<38}{_fmt(m.get('roc_auc')):>9}"
                  f"{_fmt(m.get('average_precision')):>8}{_fmt(m.get('recall')):>8}"
                  f"{_fmt(m.get('precision')):>8}{_fmt(m.get('f2_score')):>8}")

    sel = results.get("model_selection") or {}
    if sel:
        print()
        print(f" Selected for deployment : {sel.get('selected')} "
              f"(by {sel.get('criterion')})")
        print(f"   candidates: {sel.get('candidates')}")

    unc = results.get("auc_uncertainty") or {}
    if unc:
        ci = unc.get("ci95_cluster_bootstrap") or {}
        print(f" Headline LOSO ROC-AUC   : {unc.get('loso_roc_auc')} "
              f"  95% CI [{ci.get('lo')}, {ci.get('hi')}] "
              f"({unc.get('method')})")

    lk = results.get("leakage_analysis") or {}
    if lk:
        print()
        print(" Leakage check (same model, three splits)")
        print(f"   row-shuffled 5-fold {lk['row_shuffled_5fold_auc']}   "
              f"grouped 5-fold {lk['grouped_5fold_auc']}   "
              f"LOSO {lk['loso_auc']}   "
              f"optimism gap {lk['optimism_gap_rowshuffle_minus_loso']:+.4f}")

    cal = results.get("calibration") or {}
    if cal:
        print()
        print(" Calibration (lower Brier is better)")
        print(f"   uncalibrated {cal.get('brier_score_uncalibrated')}"
              f"   ->  calibrated (held out) {cal.get('brier_score')}"
              f"   improvement {cal.get('brier_improvement')}")
        for row in (cal.get("bins_heldout") or [])[:6]:
            print(f"     {row['range']:<12} n={row['n']:>5}  "
                  f"predicted {row['mean_predicted']:<6} observed {row['observed_rate']}")

    ops = results.get("operating_points") or {}
    if ops:
        print()
        print(" Operating points (out-of-fold LOSO probabilities)")
        print(f"   {'point':<38}{'thresh':>8}{'recall':>8}{'prec':>8}"
              f"{'acc':>8}{'bal.acc':>9}{'alerts/wk':>11}")
        for key in ("high_sensitivity", "balanced_f2", "high_precision",
                    "accuracy_matched_to_majority_baseline"):
            r = ops.get(key)
            if not isinstance(r, dict):
                continue
            star = " *" if key == "balanced_f2" else "  "
            print(f"  {star}{key:<36}{r['threshold']:>8}{r['recall']:>8}"
                  f"{r['precision']:>8}{r['accuracy']:>8}"
                  f"{r['balanced_accuracy']:>9}{r['alerts_per_week_across_home']:>11}")
        print(f"   * deployed.  Majority-class accuracy for comparison: "
              f"{ops.get('majority_baseline_accuracy')}")

    fi = results.get("feature_importance") or []
    if fi:
        print()
        print(" Top predictive features (permutation importance, drop in AUC):")
        for row in fi[:6]:
            print(f"   {row['feature']:<26} {row['importance_mean']:+.4f} "
                  f"± {row['importance_std']:.4f}")

    print()
    print(" NOTE: " + (ds.get("note") or "").replace("\n", " "))
    print(BAR)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and persist the fall-risk model.")
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate without overwriting the saved artefact")
    ap.add_argument("--status", action="store_true",
                    help="print the current artefact's model card and exit")
    ap.add_argument("--json", action="store_true",
                    help="print full results as JSON")
    args = ap.parse_args()

    if args.status:
        print_model_card(ml_models.get_model_status())
        return 0

    started = datetime.now()
    print(f"[{started:%Y-%m-%d %H:%M:%S}] Building dataset and training "
          f"({'DRY RUN — nothing will be saved' if args.dry_run else 'artefact will be saved'})...")

    results = ml_models.train_fall_risk_model(save=not args.dry_run)

    if results.get("error"):
        print(f"\n  TRAINING FAILED: {results['error']}", file=sys.stderr)
        return 1

    elapsed = (datetime.now() - started).total_seconds()

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return 0

    print_metrics(results)
    print(f"\n Training completed in {elapsed:.1f}s")

    if args.dry_run:
        print(" Dry run — existing artefact left untouched.")
    else:
        print()
        print_model_card(ml_models.get_model_status())
        print("\n The web app will now serve THIS file. No training happens on page load.")
        print(" Verify with:  python train_model.py --status")

    return 0


if __name__ == "__main__":
    sys.exit(main())
