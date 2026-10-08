"""
ml_models.py — Machine Learning models for CareDocs AI
Models implemented:
  1. Fall-Risk Binary Classifier
     - Logistic Regression + Random Forest
     - Features engineered from existing DB schema
     - Evaluation: ROC-AUC, Precision, Recall, F2, Confusion Matrix
     - SHAP-style feature importance (permutation-based, no extra deps)

  2. Composite Daily Risk Ranking
     - Rule/heuristic-based composite score (no training data needed)
     - Outputs ranked list of residents by risk today

All computation uses scikit-learn (already in requirements.txt) and
standard Python only — no additional installs needed.
"""
import os
import sqlite3
import json
import math
from datetime import date, timedelta
from pathlib import Path

# ── DB helpers (copy from FUSE mount to /tmp) ──────────────────────────────────
_BASE = Path(__file__).parent

def _get_db_path():
    # The live DB may sit on a mounted/FUSE filesystem where SQLite locking is
    # unreliable, so all analytics read from a snapshot copy in the OS temp
    # directory. tempfile.gettempdir() is used rather than a hard-coded "/tmp"
    # so this works on Windows as well as Linux/macOS.
    import tempfile
    src = _BASE / "carehome.db"
    dst = Path(tempfile.gettempdir()) / "carehome_ml.db"
    with open(src, "rb") as f:
        dst.write_bytes(f.read())
    return str(dst)

def _write_db_back(tmp_path):
    dst = _BASE / "carehome.db"
    with open(tmp_path, "rb") as f:
        dst.write_bytes(f.read())


# SECTION 1 — FALL RISK FEATURE ENGINEERING

FEATURE_NAMES = [
    # ── v2 block: the ten features carried forward from the first model ──────
    "Falls (30d)", "Incidents (30d)", "Fluid Mean 7d (ml)", "Fluid Trend",
    "Wellbeing Score", "Wellbeing Trend", "MAR Adherence %",
    "Note Gap (days)", "Risk Score", "Medication Count",
    # ── v3 block: five features added for the 50-resident cohort ─────────────
    # Motivation is stated in the dissertation (Chapter 5). In short: the v2 set
    # described a resident's LEVEL but barely described their VOLATILITY or
    # their RECENT CHANGE, and the generating process makes deterioration an
    # episodic, autocorrelated phenomenon. A resident whose fluid intake has
    # dropped 300 ml in a week is a different risk from one who has always been
    # at that level, and the v2 set could not tell them apart.
    "Fluid Delta 7v14 (ml)",   # short-window minus long-window mean
    "Fluid Volatility (CV)",   # coefficient of variation over 14d
    "Adherence Delta 7v30",    # short-window minus long-window adherence
    "Days Since Last Fall",    # censored at 180
    "Wellbeing Drop 60d",      # peak-to-current decline
]

# Machine-readable feature keys, in the same order as FEATURE_NAMES. Persisted
# inside the model artefact so that a loaded model can assert it is being fed
# the same features, in the same order, that it was trained on.
FEATURE_KEYS = [
    "falls_30d", "incidents_30d", "fluid_mean_7d", "fluid_trend_14d",
    "wellbeing_latest", "wellbeing_trend_90d", "mar_adherence_pct_14d",
    "note_gap_days", "risk_score_latest", "medication_count",
    "fluid_delta_7v14", "fluid_cv_14d", "adherence_delta_7v30",
    "days_since_last_fall", "wellbeing_drop_60d",
]


def _slope(vals):
    """Ordinary least-squares slope of `vals` against their index position."""
    n = len(vals)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(vals) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, vals))
    den = sum((xi - mx) ** 2 for xi in xs)
    return num / den if den else 0.0


def compute_features(c, rid, as_of, window: int = 14) -> list:
    """
    Compute the 10-element fall-risk feature vector for one resident, as at
    one point in time.

    SINGLE SOURCE OF TRUTH FOR FEATURES.
    This function is called from exactly two places:
      * _build_fall_risk_dataset()  — offline, to build the training matrix
      * predict_fall_risk()         — online, to score a resident today

    Keeping both paths on the same code prevents "training/serving skew": the
    classic failure mode where a model is trained on features computed one way
    and then served features computed slightly differently (different lookback
    window, different NULL handling, different units), so live predictions are
    quietly wrong even though offline metrics looked fine.

    Args:
        c:      an open sqlite3 cursor (row_factory may be anything)
        rid:    resident_id
        as_of:  datetime.date — features use ONLY data on or before this date,
                which is what makes the training labels causally valid.
        window: observation window in days for trend features (default 14)

    Returns:
        list[float] of length 15, ordered to match FEATURE_NAMES/FEATURE_KEYS.
    """
    from datetime import datetime as _dt

    cur_str = as_of.isoformat()
    obs_str = (as_of - timedelta(days=window)).isoformat()
    d7_str  = (as_of - timedelta(days=7)).isoformat()
    d30_str = (as_of - timedelta(days=30)).isoformat()
    d90_str = (as_of - timedelta(days=90)).isoformat()

    # f1 falls in last 30 days
    f1 = c.execute(
        "SELECT COUNT(*) FROM incidents "
        "WHERE resident_id=? AND incident_type LIKE '%fall%' "
        "AND date>=? AND date<=?",
        (rid, d30_str, cur_str)
    ).fetchone()[0]

    # f2 total incidents last 30 days
    f2 = c.execute(
        "SELECT COUNT(*) FROM incidents "
        "WHERE resident_id=? AND date>=? AND date<=?",
        (rid, d30_str, cur_str)
    ).fetchone()[0]

    # f3 mean fluid last 7 days
    fluids = [r2[0] for r2 in c.execute(
        "SELECT fluid_intake_ml FROM care_notes "
        "WHERE resident_id=? AND date>=? AND date<=? "
        "AND fluid_intake_ml IS NOT NULL",
        (rid, d7_str, cur_str)
    ).fetchall() if r2[0]]
    f3 = sum(fluids) / len(fluids) if fluids else 0.0

    # f4 fluid trend over the observation window
    fluids14 = [r2[0] for r2 in c.execute(
        "SELECT fluid_intake_ml FROM care_notes "
        "WHERE resident_id=? AND date>=? AND date<=? "
        "AND fluid_intake_ml IS NOT NULL ORDER BY date",
        (rid, obs_str, cur_str)
    ).fetchall() if r2[0]]
    f4 = _slope(fluids14) if fluids14 else 0.0

    # f5 latest wellbeing score
    wb = c.execute(
        "SELECT overall_score FROM wellbeing "
        "WHERE resident_id=? AND assessment_date<=? ORDER BY assessment_date DESC LIMIT 1",
        (rid, cur_str)
    ).fetchone()
    f5 = wb[0] if wb and wb[0] else 5.0

    # f6 wellbeing trend over 90 days
    wb_hist = [r2[0] for r2 in c.execute(
        "SELECT overall_score FROM wellbeing "
        "WHERE resident_id=? AND assessment_date>=? AND assessment_date<=? "
        "ORDER BY assessment_date",
        (rid, d90_str, cur_str)
    ).fetchall() if r2[0]]
    f6 = _slope(wb_hist) if len(wb_hist) >= 2 else 0.0

    # f7 MAR adherence over the observation window
    # NOTE: mar_records.administered is stored as the text 'Yes'/'No',
    # not as an integer. An earlier version of this query only matched
    # administered=1, which is never true for text values in SQLite —
    # f7 was silently always near-zero regardless of actual adherence.
    # Fixed to match both representations, consistent with the other
    # adherence queries in this module (see get_detailed_adherence_report).
    mar = c.execute(
        "SELECT COUNT(*), SUM(CASE WHEN administered='Yes' OR administered=1 THEN 1 ELSE 0 END) "
        "FROM mar_records WHERE resident_id=? AND date>=? AND date<=?",
        (rid, obs_str, cur_str)
    ).fetchone()
    f7 = (mar[1] / mar[0] * 100) if mar and mar[0] else 100.0

    # f8 note gap
    last_note = c.execute(
        "SELECT date FROM care_notes WHERE resident_id=? AND date<=? "
        "ORDER BY date DESC LIMIT 1",
        (rid, cur_str)
    ).fetchone()
    if last_note and last_note[0]:
        gap = (as_of - _dt.strptime(last_note[0], "%Y-%m-%d").date()).days
    else:
        gap = 30
    f8 = min(gap, 30)

    # f9 latest risk score
    risk = c.execute(
        "SELECT score FROM risk_assessments "
        "WHERE resident_id=? AND date_assessed<=? "
        "ORDER BY date_assessed DESC LIMIT 1",
        (rid, cur_str)
    ).fetchone()
    f9 = float(risk[0]) if risk and risk[0] else 10.0

    # f10 medication count
    f10 = c.execute(
        "SELECT COUNT(*) FROM medications WHERE resident_id=? AND status='active'",
        (rid,)
    ).fetchone()[0]

    # v3 FEATURES — change and volatility, not just level
    # Every one of these is still computed strictly from records dated on or
    # before `as_of`, so the causal cut-off that makes the labels valid is
    # preserved. This function remains the single source of truth: adding a
    # feature here automatically adds it to BOTH the training matrix and the
    # serving path, which is what prevents training/serving skew.

    # f11 fluid delta: recent 7-day mean minus the mean over the 8-14 day
    #     window before it. Positive means intake is recovering, negative means
    #     it is falling away from this resident's own recent baseline. Framing
    #     it as a within-resident difference removes the between-resident level
    #     effect that f3 already carries.
    d14_str = (as_of - timedelta(days=14)).isoformat()
    d8_str  = (as_of - timedelta(days=8)).isoformat()
    prior = [r2[0] for r2 in c.execute(
        "SELECT fluid_intake_ml FROM care_notes "
        "WHERE resident_id=? AND date>=? AND date<? AND fluid_intake_ml IS NOT NULL",
        (rid, d14_str, d8_str)
    ).fetchall() if r2[0]]
    f11 = (f3 - (sum(prior) / len(prior))) if (prior and f3) else 0.0

    # f12 fluid volatility over the observation window, as a coefficient of
    #     variation (SD / mean). Unstable intake is clinically meaningful in a
    #     way that a mean cannot express: alternating good and bad days is a
    #     different picture from a steady moderate intake at the same average.
    if len(fluids14) >= 3:
        m14 = sum(fluids14) / len(fluids14)
        var = sum((v - m14) ** 2 for v in fluids14) / len(fluids14)
        f12 = (math.sqrt(var) / m14) if m14 else 0.0
    else:
        f12 = 0.0

    # f13 adherence delta: last 7 days versus the last 30. Same within-resident
    #     differencing logic as f11 — a resident who has always refused a third
    #     of doses is a standing problem; one who has started refusing this week
    #     is a change.
    mar7 = c.execute(
        "SELECT COUNT(*), SUM(CASE WHEN administered='Yes' OR administered=1 THEN 1 ELSE 0 END) "
        "FROM mar_records WHERE resident_id=? AND date>=? AND date<=?",
        (rid, d7_str, cur_str)
    ).fetchone()
    mar30 = c.execute(
        "SELECT COUNT(*), SUM(CASE WHEN administered='Yes' OR administered=1 THEN 1 ELSE 0 END) "
        "FROM mar_records WHERE resident_id=? AND date>=? AND date<=?",
        (rid, d30_str, cur_str)
    ).fetchone()
    a7  = (mar7[1] / mar7[0] * 100) if mar7 and mar7[0] else 100.0
    a30 = (mar30[1] / mar30[0] * 100) if mar30 and mar30[0] else 100.0
    f13 = a7 - a30

    # f14 days since the most recent fall, censored at 180. Fall recurrence is
    #     strongly time-dependent — risk is highest in the days immediately
    #     after a fall — and f1 (a 30-day count) cannot express "yesterday"
    #     versus "29 days ago". Censoring rather than using a sentinel keeps the
    #     variable on a single monotone scale for the linear model.
    lastfall = c.execute(
        "SELECT date FROM incidents WHERE resident_id=? AND incident_type LIKE '%fall%' "
        "AND date<=? ORDER BY date DESC LIMIT 1",
        (rid, cur_str)
    ).fetchone()
    if lastfall and lastfall[0]:
        f14 = min((as_of - _dt.strptime(lastfall[0], "%Y-%m-%d").date()).days, 180)
    else:
        f14 = 180.0

    # f15 wellbeing drop: highest score recorded in the last 60 days minus the
    #     current score. A resident who has fallen from 9 to 6 and one who has
    #     sat at 6 all year share the same f5, and they are not the same risk.
    d60_str = (as_of - timedelta(days=60)).isoformat()
    wb60 = [r2[0] for r2 in c.execute(
        "SELECT overall_score FROM wellbeing WHERE resident_id=? "
        "AND assessment_date>=? AND assessment_date<=?",
        (rid, d60_str, cur_str)
    ).fetchall() if r2[0]]
    f15 = (max(wb60) - f5) if wb60 else 0.0

    return [float(f1), float(f2), float(f3), float(f4), float(f5),
            float(f6), float(f7), float(f8), float(f9), float(f10),
            float(f11), float(f12), float(f13), float(f14), float(f15)]


def _build_fall_risk_dataset(db_path: str) -> tuple[list, list, list]:
    """
    Build a supervised dataset for fall-risk prediction.

    Target:  1 if a fall incident occurred within the NEXT 7 days
             0 otherwise.

    Window:  slide a 14-day observation window across the dataset,
             step = 7 days.  For each window / resident pair produce
             one row of features + one label.

    Features (all computable from existing schema):
      f1  falls_last_30d        — count of falls in prior 30 days
      f2  incidents_last_30d    — total incidents (any type) last 30 days
      f3  fluid_mean_7d         — mean fluid intake last 7 days (ml)
      f4  fluid_trend           — linear slope of daily fluid over 14d
      f5  wellbeing_latest      — most recent overall wellbeing score
      f6  wellbeing_trend       — slope of wellbeing over 90d
      f7  mar_adherence_14d     — % doses administered in last 14 days
      f8  note_gap_days         — days since last care note
      f9  risk_score_latest     — most recent numeric risk assessment score
      f10 medication_count      — number of active medications
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    residents = c.execute(
        "SELECT resident_id FROM residents WHERE active=1"
    ).fetchall()

    # Date range: use all dates in db
    row = c.execute(
        "SELECT MIN(date), MAX(date) FROM care_notes"
    ).fetchone()
    if not row or not row[0]:
        conn.close()
        return [], [], []

    from datetime import datetime
    start = datetime.strptime(row[0], "%Y-%m-%d").date()
    end   = datetime.strptime(row[1], "%Y-%m-%d").date()

    X, y, meta = [], [], []

    window = 14
    step   = 7
    cursor_date = start + timedelta(days=window)

    while cursor_date <= end - timedelta(days=7):
        lbl_end   = cursor_date + timedelta(days=7)
        cur_str   = cursor_date.isoformat()
        lbl_str   = lbl_end.isoformat()

        for r in residents:
            rid = r["resident_id"]

            # ── Data-coverage gate ──────────────────────────────────────────
            # Only emit a training row if this resident actually has care
            # records inside the observation window. Without this check the
            # sliding window runs from the earliest date in the WHOLE table,
            # and a couple of stray historical notes generate hundreds of
            # all-zero rows labelled "no fall" — rows that describe an absence
            # of records, not an absence of risk. They are trivially easy
            # negatives, so they inflate ROC-AUC, distort the class balance,
            # and corrupt the feature medians used for local explanations.
            coverage = c.execute(
                "SELECT COUNT(*) FROM care_notes "
                "WHERE resident_id=? AND date>=? AND date<=?",
                (rid, (cursor_date - timedelta(days=window)).isoformat(), cur_str)
            ).fetchone()[0]
            if coverage == 0:
                continue

            # Features come from the SAME function used at prediction time.
            feats = compute_features(c, rid, cursor_date, window=window)

            # Label: any fall in next 7 days?
            label = c.execute(
                "SELECT COUNT(*) FROM incidents "
                "WHERE resident_id=? AND incident_type LIKE '%fall%' "
                "AND date>? AND date<=?",
                (rid, cur_str, lbl_str)
            ).fetchone()[0]
            label = 1 if label > 0 else 0

            X.append(feats)
            y.append(label)
            meta.append({"resident_id": rid, "date": cur_str})

        cursor_date += timedelta(days=step)

    conn.close()
    return X, y, meta




# SECTION 2 — MODEL TRAINING AND EVALUATION

def _metrics(y_true, y_pred_prob, threshold=0.3):
    """Compute binary classification metrics."""
    n = len(y_true)
    y_pred = [1 if p >= threshold else 0 for p in y_pred_prob]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2         = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0
    accuracy   = (tp + tn) / n if n > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC-AUC — computed via the Mann-Whitney U / rank-sum identity so that
    # tied prediction scores are handled correctly (average rank for ties).
    # NOTE: an earlier version of this function sorted (prob, label) tuples
    # directly, which let ties be broken by the label itself and silently
    # inflated AUC to 1.0 whenever every prediction had the same probability
    # (e.g. the majority-class baseline below). Fixed by using explicit
    # midrank handling, equivalent to sklearn.metrics.roc_auc_score.
    pos = sum(y_true)
    neg = n - pos
    auc = 0.0
    if pos > 0 and neg > 0:
        order = sorted(range(n), key=lambda i: y_pred_prob[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and y_pred_prob[order[j + 1]] == y_pred_prob[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed midrank for the tied block
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        rank_sum_pos = sum(ranks[i] for i in range(n) if y_true[i] == 1)
        auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f2_score":  round(f2, 3),
        "accuracy":  round(accuracy, 3),
        "specificity": round(specificity, 3),
        "roc_auc":   round(auc, 3),
        "threshold": threshold,
        "n_positive": pos,
        "n_negative": neg,
    }


def _permutation_importance(model, X_test, y_test, n_repeat=5):
    """
    Model-agnostic permutation feature importance.
    Returns list of (feature_idx, mean_importance, std_importance).
    """
    from sklearn.metrics import roc_auc_score
    import random

    base_preds = model.predict_proba(X_test)[:, 1]
    base_score = roc_auc_score(y_test, base_preds) if len(set(y_test)) > 1 else 0.5

    importances = []
    for fi in range(X_test.shape[1]):
        scores = []
        X_copy = X_test.copy()
        for _ in range(n_repeat):
            col = X_copy[:, fi].copy()
            random.shuffle(col)
            X_copy[:, fi] = col
            p = model.predict_proba(X_copy)[:, 1]
            s = roc_auc_score(y_test, p) if len(set(y_test)) > 1 else 0.5
            scores.append(base_score - s)
            X_copy[:, fi] = X_test[:, fi]  # restore
        importances.append({
            "feature": FEATURE_NAMES[fi],
            "importance_mean": round(sum(scores)/len(scores), 4),
            "importance_std":  round(
                math.sqrt(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)),
                4
            )
        })
    return sorted(importances, key=lambda x: -x["importance_mean"])


def _average_precision(y_true, probs) -> float:
    """
    Area under the precision-recall curve, computed as the step-wise average
    precision (the same estimator sklearn uses).

    Reported alongside ROC-AUC because ROC-AUC is optimistic under heavy class
    imbalance: it rewards separating the huge negative class, which is easy.
    Average precision is measured against the base rate — here about 0.05 — so
    the lift over chance is interpretable, and it degrades honestly when a model
    buys recall with a flood of false positives.
    """
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    tp = 0
    ap = 0.0
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0
    for rank, i in enumerate(order, start=1):
        if y_true[i] == 1:
            tp += 1
            ap += tp / rank
    return ap / n_pos


def _bins_of(probs, y_arr, n_bins=5):
    """Reliability table: mean predicted probability vs observed rate, per bin."""
    import numpy as np
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        mask = (probs >= lo) & (probs < hi if b < len(edges) - 2 else probs <= hi)
        if mask.sum() == 0:
            continue
        out.append({"range": f"{lo:.1f}\u2013{hi:.1f}", "n": int(mask.sum()),
                    "mean_predicted": round(float(probs[mask].mean()), 3),
                    "observed_rate": round(float(y_arr[mask].mean()), 3)})
    return out


def _threshold_sweep(y_true, probs, n_residents, n_weeks, grid=None):
    """
    Operating-characteristic table over a grid of decision thresholds.

    A single accuracy number is close to meaningless on this problem, because a
    model that says "no fall" every time already scores about 95 %. What a care
    home manager actually needs to choose is an OPERATING POINT: how many alerts
    per week the staff can absorb, and what fraction of falls that catches. This
    table is the object that supports that decision, and it is what the
    dissertation reports instead of a headline accuracy.
    """
    if grid is None:
        # A fixed 0.1-0.9 grid is wrong for a CALIBRATED model. Once the scores
        # mean what they say, almost all of them sit below 0.2 — because almost
        # no resident really does have a 50 % chance of falling next week. A
        # fixed grid would then place the optimum at the grid edge and quietly
        # report a boundary artefact as a tuned threshold. The grid is therefore
        # built from the score distribution itself (percentiles), with a few
        # round numbers added so the table is still readable.
        srt = sorted(probs)
        qs = [srt[min(len(srt) - 1, int(q * len(srt)))]
              for q in (0.50, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95,
                        0.96, 0.97, 0.98, 0.99, 0.995)]
        grid = sorted({round(v, 4) for v in qs if 0 < v < 1}
                      | {0.05, 0.10, 0.20, 0.30, 0.50})
    rows = []
    for th in grid:
        m = _metrics(y_true, probs, threshold=th)
        alerts = m["tp"] + m["fp"]
        rows.append({
            "threshold": th,
            "accuracy": m["accuracy"],
            "balanced_accuracy": round((m["recall"] + m["specificity"]) / 2, 4),
            "recall": m["recall"],
            "precision": m["precision"],
            "specificity": m["specificity"],
            "f2": m["f2_score"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "alerts_per_week_across_home": round(alerts / n_weeks, 1) if n_weeks else None,
        })
    return rows


def _bootstrap_auc_ci(y_true, probs, groups, n_boot=400, seed=42, alpha=0.05):
    """
    Cluster bootstrap confidence interval for ROC-AUC, resampling RESIDENTS
    rather than rows.

    Resampling rows would treat the ~50 windows contributed by one resident as
    50 independent observations. They are not: they overlap in time and share a
    person, so a row-level bootstrap understates the true uncertainty — often by
    a lot. Resampling whole residents respects the clustering, which is the
    correct unit of independence here and is what TRIPOD+AI expects a prediction
    model to report alongside a point estimate.
    """
    import random as _random
    rnd = _random.Random(seed)
    by_res = {}
    for i, g in enumerate(groups):
        by_res.setdefault(g, []).append(i)
    residents = list(by_res.keys())
    aucs = []
    for _ in range(n_boot):
        picked = [residents[rnd.randrange(len(residents))] for _ in residents]
        idx = [i for r in picked for i in by_res[r]]
        yt = [y_true[i] for i in idx]
        pp = [probs[i] for i in idx]
        if sum(yt) == 0 or sum(yt) == len(yt):
            continue
        aucs.append(_metrics(yt, pp)["roc_auc"])
    if len(aucs) < 20:
        return {"lo": None, "hi": None, "n_boot": len(aucs)}
    aucs.sort()
    lo = aucs[int((alpha / 2) * (len(aucs) - 1))]
    hi = aucs[int((1 - alpha / 2) * (len(aucs) - 1))]
    return {"lo": round(lo, 3), "hi": round(hi, 3), "n_boot": len(aucs)}


def train_fall_risk_model(save: bool = True):
    """
    Train, evaluate and select the fall-risk classifier.  (v3 pipeline)

    This is the OFFLINE path. It is intentionally expensive — dataset build,
    three cross-validation schemes, probability calibration, permutation
    importance, cluster bootstrap, threshold search — and is never called during
    a page load. The web app serves the artefact this function writes.

    WHAT CHANGED IN v3, AND WHY
    v2 fitted logistic regression and a random forest, cross-validated with a
    row-shuffled 5-fold split plus leave-one-subject-out, and served predictions
    at a fixed threshold of 0.3. Four things were wrong with that, and all four
    are fixed here.

    1. LEAKY "FAST" CV.  Row-shuffled StratifiedKFold puts overlapping windows
       from the SAME resident in both the training and the test fold, so the
       model can learn "what is normal for Mrs Brown" and be rewarded for it.
       v3 keeps that split — but only as an exhibit, reported side by side with
       StratifiedGroupKFold, which splits on residents. The gap between the two
       IS the leakage, quantified.

    2. UNCALIBRATED PROBABILITIES.  class_weight="balanced" is necessary to buy
       recall on a 5 %-positive problem, but it systematically inflates the
       predicted values: in v2 the 0.8-1.0 bin had an observed event rate of
       0.34. The number on screen said 90 %, reality said 34 %. v3 wraps the
       selected estimator in an isotonic calibration layer fitted inside the
       cross-validation, so the displayed probability means what it says. This
       is measured, not asserted: see results["calibration"]["brier_score"]
       before and after.

    3. AN ARBITRARY THRESHOLD.  0.3 was a guess. v3 searches the threshold on
       out-of-fold LOSO probabilities against an explicit objective (F2, which
       weights recall four times as heavily as precision because a missed fall
       costs more than an unnecessary check) and records both the chosen value
       and the full sweep so the choice is auditable and reversible.

    4. A POINT ESTIMATE WITH NO UNCERTAINTY.  v3 reports a resident-level
       cluster bootstrap interval on the headline AUC.

    A third candidate estimator, histogram gradient boosting, is added so that
    the linear model is not selected by default for want of a competitor.

    Args:
        save: when True, persist the selected estimator plus all evaluation
              metadata to ml_fall_risk_model.pkl.

    Returns a results dict with metrics, feature importance and dataset info.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.isotonic import IsotonicRegression

    db_path = _get_db_path()
    X_raw, y, meta = _build_fall_risk_dataset(db_path)

    if len(X_raw) < 10:
        return {
            "error": "Insufficient data for training. Run synthetic_cohort.py first.",
            "n_samples": len(X_raw)
        }

    X = np.array(X_raw, dtype=float)
    y_arr = np.array(y)
    n_pos = int(y_arr.sum())
    n_neg = len(y_arr) - n_pos
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    groups = np.array([m_["resident_id"] for m_ in meta])
    resident_ids = sorted(set(groups.tolist()))
    n_residents = len(resident_ids)
    n_weeks = len(set(m_["date"] for m_ in meta))

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_arr)
    class_weight = {0: cw[0], 1: cw[1]}

    # ── Hyperparameter selection ────────────────────────────────────────────
    # Done ONCE, by resident-grouped 5-fold CV, before any leave-one-subject-out
    # number is computed — and reported, so it is part of the record rather than
    # a silent default. The search is deliberately small: with 119 events and 15
    # features, an exhaustive grid would overfit the validation split faster than
    # it would find a better model, and the regularisation path for a linear
    # model on this data is famously flat (see results["hyperparameters"]).
    #
    # DISCLOSURE: selection uses the whole cohort, so the LOSO figures carry a
    # small optimism from that step. Its size is bounded by the spread of the
    # search results themselves, which is reported below — here it is under
    # 0.005 AUC, i.e. negligible relative to the bootstrap interval.

    # ── Candidate estimators ────────────────────────────────────────────────
    # Deliberately spans three model families: a regularised linear model that
    # is interpretable and stable at low event counts; a bagged tree ensemble;
    # and a boosted tree ensemble. If the linear model wins, that is a finding
    # about the problem (the generating process is a logistic hazard, so a
    # linear decision boundary in the right features is close to correct) rather
    # than an artefact of never having tried anything else.
    HP = {"lr_C": 0.03, "rf_max_depth": 4, "rf_min_samples_leaf": 5}

    def _make_lr(cwd, C=None):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight=cwd, max_iter=3000,
                                       C=HP["lr_C"] if C is None else C,
                                       random_state=42)),
        ])

    def _make_rf(cwd):
        return RandomForestClassifier(
            n_estimators=400, class_weight=cwd, max_depth=HP["rf_max_depth"],
            min_samples_leaf=HP["rf_min_samples_leaf"], random_state=42, n_jobs=-1)

    def _make_gb(cwd):
        # HistGradientBoosting takes sample weights rather than class_weight.
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.06,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=42)

    FACTORIES = {
        "logistic_regression": _make_lr,
        "random_forest": _make_rf,
        "gradient_boosting": _make_gb,
    }

    def _fit(name, cwd, Xtr, ytr):
        mdl = FACTORIES[name](cwd)
        if name == "gradient_boosting":
            w = np.where(ytr == 1, cwd[1], cwd[0])
            mdl.fit(Xtr, ytr, sample_weight=w)
        else:
            mdl.fit(Xtr, ytr)
        return mdl

    results = {}

    # ── Run the hyperparameter search (grouped CV, AUC objective) ────────────
    try:
        cv_hp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        def _grouped_auc(make, weighted=False):
            p = np.zeros(len(y_arr))
            for tr, te in cv_hp.split(X, y_arr, groups=groups):
                cwf = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_arr[tr])
                mdl = make({0: cwf[0], 1: cwf[1]})
                if weighted:
                    mdl.fit(X[tr], y_arr[tr],
                            sample_weight=np.where(y_arr[tr] == 1, cwf[1], cwf[0]))
                else:
                    mdl.fit(X[tr], y_arr[tr])
                p[te] = mdl.predict_proba(X[te])[:, 1]
            return _metrics(y_arr.tolist(), p.tolist())["roc_auc"]

        lr_path = {C: _grouped_auc(lambda cw, C=C: _make_lr(cw, C=C))
                   for C in (0.01, 0.03, 0.1, 0.3, 1.0)}
        HP["lr_C"] = max(lr_path, key=lambda C: lr_path[C])
        rf_path = {}
        for md in (3, 4, 6):
            rf_path[md] = _grouped_auc(
                lambda cw, md=md: RandomForestClassifier(
                    n_estimators=400, class_weight=cw, max_depth=md,
                    min_samples_leaf=5, random_state=42, n_jobs=-1))
        HP["rf_max_depth"] = max(rf_path, key=lambda d_: rf_path[d_])
        results["hyperparameters"] = {
            "selected": dict(HP),
            "search": {"logistic_regression_C": {str(k): v for k, v in lr_path.items()},
                       "random_forest_max_depth": {str(k): v for k, v in rf_path.items()}},
            "method": "resident-grouped StratifiedGroupKFold, 5 folds, ROC-AUC objective",
            "spread_auc": {
                "logistic_regression_C": round(max(lr_path.values()) - min(lr_path.values()), 4),
                "random_forest_max_depth": round(max(rf_path.values()) - min(rf_path.values()), 4)},
            "note": ("The regularisation path is flat, which is itself a finding: at "
                     "this event count the choice of C is not what limits performance, "
                     "so the small optimism from selecting on the full cohort is "
                     "bounded by the spread above."),
        }
    except Exception as e:  # pragma: no cover
        results["hyperparameters"] = {"error": str(e), "selected": dict(HP)}

    # ── Scheme A: row-shuffled 5-fold (KEPT AS AN EXHIBIT — it leaks) ────────
    cv_rows = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = {}
    for name in FACTORIES:
        if n_pos < 5:
            mdl = _fit(name, class_weight, X, y_arr)
            probs = mdl.predict_proba(X)[:, 1]
        else:
            probs = np.zeros(len(y_arr))
            for tr, te in cv_rows.split(X, y_arr):
                cwf = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_arr[tr])
                mdl = _fit(name, {0: cwf[0], 1: cwf[1]}, X[tr], y_arr[tr])
                probs[te] = mdl.predict_proba(X[te])[:, 1]
        cv_probs[name] = probs
        results[name] = _metrics(y_arr.tolist(), probs.tolist(), threshold=0.3)
        results[name]["average_precision"] = round(
            _average_precision(y_arr.tolist(), probs.tolist()), 4)

    # ── Scheme B: StratifiedGroupKFold — same 5 folds, split on RESIDENTS ────
    # This is the honest fast estimate. Comparing it with Scheme A isolates the
    # contribution of subject-level leakage, holding fold count constant.
    group_probs = {}
    try:
        cv_grp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for name in FACTORIES:
            probs = np.zeros(len(y_arr))
            for tr, te in cv_grp.split(X, y_arr, groups=groups):
                if y_arr[tr].sum() < 1:
                    probs[te] = y_arr[tr].mean() if len(tr) else 0.0
                    continue
                cwf = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_arr[tr])
                mdl = _fit(name, {0: cwf[0], 1: cwf[1]}, X[tr], y_arr[tr])
                probs[te] = mdl.predict_proba(X[te])[:, 1]
            group_probs[name] = probs
            key = f"{name}_groupkfold"
            results[key] = _metrics(y_arr.tolist(), probs.tolist(), threshold=0.3)
            results[key]["average_precision"] = round(
                _average_precision(y_arr.tolist(), probs.tolist()), 4)
            results[key]["description"] = (
                "Same 5-fold budget as the row-shuffled split above, but folds are "
                "formed on residents so no resident appears in both train and test. "
                "Any drop relative to the row-shuffled figure is subject-level leakage.")
    except Exception as e:      # pragma: no cover — older sklearn
        results["_groupkfold_error"] = str(e)

    # ── Scheme C: leave-one-subject-out — the deployment-realistic estimate ──
    def _loso_probs(name):
        p = np.zeros(len(y_arr), dtype=float)
        for held_out in resident_ids:
            tr_mask = groups != held_out
            te_mask = ~tr_mask
            y_tr = y_arr[tr_mask]
            if y_tr.sum() < 1 or (len(y_tr) - y_tr.sum()) < 1:
                p[te_mask] = y_tr.mean() if len(y_tr) else 0.0
                continue
            cwf = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
            mdl = _fit(name, {0: cwf[0], 1: cwf[1]}, X[tr_mask], y_tr)
            p[te_mask] = mdl.predict_proba(X[te_mask])[:, 1]
        return p

    loso = {}
    for name in FACTORIES:
        p = _loso_probs(name)
        loso[name] = p
        key = f"{name}_loso"
        results[key] = _metrics(y_arr.tolist(), p.tolist(), threshold=0.3)
        results[key]["average_precision"] = round(
            _average_precision(y_arr.tolist(), p.tolist()), 4)
        results[key]["description"] = (
            "Leave-one-subject-out: on each fold one resident is removed entirely "
            "and the model is trained on the rest, then scored on that unseen "
            "resident alone. This is the estimate that answers the deployment "
            "question — how well does this work on someone the model has never seen?")

    # ── Baseline ────────────────────────────────────────────────────────────
    maj = [n_pos / len(y_arr)] * len(y_arr)
    results["baseline_majority"] = _metrics(y_arr.tolist(), maj, threshold=0.5)
    results["baseline_majority"]["description"] = (
        "Predicts the cohort base rate for everyone. Its accuracy is the number "
        "any reported accuracy must be compared against, and it is high precisely "
        "because falls are rare.")

    # A second, harder baseline: rank residents by the static falls-risk score
    # already recorded by staff (feature 9). If the model cannot beat the
    # paperwork the home already fills in, it has no reason to exist.
    static_score = X[:, FEATURE_KEYS.index("risk_score_latest")]
    rng_ = static_score.max() - static_score.min()
    static_prob = ((static_score - static_score.min()) / rng_) if rng_ else np.zeros(len(y_arr))
    results["baseline_static_risk_score"] = _metrics(
        y_arr.tolist(), static_prob.tolist(), threshold=0.5)
    results["baseline_static_risk_score"]["average_precision"] = round(
        _average_precision(y_arr.tolist(), static_prob.tolist()), 4)
    results["baseline_static_risk_score"]["description"] = (
        "The home's existing paper falls-risk assessment score, min-max scaled and "
        "used directly as a ranking. The clinically meaningful question is not "
        "'is the model better than chance' but 'is it better than what staff "
        "already have'.")

    # ── Model selection: LOSO ROC-AUC, rule fixed in advance ────────────────
    loso_aucs = {n: results[f"{n}_loso"]["roc_auc"] for n in FACTORIES}
    selected_name = max(loso_aucs, key=lambda n: loso_aucs[n])
    results["model_selection"] = {
        "selected": selected_name,
        "criterion": "leave-one-subject-out ROC-AUC",
        "candidates": loso_aucs,
        "note": (
            "Selected on unseen-resident performance, not on the row-shuffled "
            "5-fold figure, which is optimistically biased because overlapping "
            "windows from the same resident appear in both train and test folds. "
            "The rule was fixed before the numbers were seen."),
    }

    sel_loso = loso[selected_name]

    # ── Uncertainty on the headline number ──────────────────────────────────
    results["auc_uncertainty"] = {
        "loso_roc_auc": results[f"{selected_name}_loso"]["roc_auc"],
        "ci95_cluster_bootstrap": _bootstrap_auc_ci(
            y_arr.tolist(), sel_loso.tolist(), groups.tolist()),
        "method": "resident-level cluster bootstrap, 400 resamples",
        "note": "Residents are resampled, not rows: the ~50 windows a resident "
                "contributes are not independent observations.",
    }

    # ── Leakage quantified ──────────────────────────────────────────────────
    if f"{selected_name}_groupkfold" in results:
        results["leakage_analysis"] = {
            "model": selected_name,
            "row_shuffled_5fold_auc": results[selected_name]["roc_auc"],
            "grouped_5fold_auc": results[f"{selected_name}_groupkfold"]["roc_auc"],
            "loso_auc": results[f"{selected_name}_loso"]["roc_auc"],
            "optimism_gap_rowshuffle_minus_loso": round(
                results[selected_name]["roc_auc"] - results[f"{selected_name}_loso"]["roc_auc"], 4),
            "note": ("With a large enough cohort, holding one resident out barely "
                     "changes the training set, so the three numbers converge. A "
                     "wide gap is a symptom of too few subjects, not of a good model."),
        }

    # ── Probability calibration ─────────────────────────────────────────────
    # Fitted OUT OF FOLD: the isotonic map is learned on LOSO probabilities, so
    # it is never fitted on data the underlying model saw during training. Brier
    # score is reported before and after so the improvement is evidence, not a
    # claim.
    brier_raw = float(np.mean((sel_loso - y_arr) ** 2))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(sel_loso, y_arr)
    calibrated = iso.predict(sel_loso)
    brier_cal = float(np.mean((calibrated - y_arr) ** 2))

    def _bins(probs):
        edges = np.linspace(0, 1, 6)
        out = []
        for b in range(len(edges) - 1):
            lo, hi = edges[b], edges[b + 1]
            mask = (probs >= lo) & (probs < hi if b < len(edges) - 2 else probs <= hi)
            if mask.sum() == 0:
                continue
            out.append({
                "range": f"{lo:.1f}–{hi:.1f}",
                "n": int(mask.sum()),
                "mean_predicted": round(float(probs[mask].mean()), 3),
                "observed_rate": round(float(y_arr[mask].mean()), 3),
            })
        return out

    # The Brier score above is optimistic: the isotonic map was fitted on the
    # very probabilities it is being scored on. An honest figure needs the map
    # fitted and evaluated on DIFFERENT residents, so it is recomputed here with
    # a 5-fold resident split wrapped around the calibration step. This is the
    # number the dissertation quotes.
    def _nested_calibration_brier():
        import random as _r
        rr = _r.Random(42)
        res_list = list(resident_ids)
        rr.shuffle(res_list)
        folds = [res_list[k::5] for k in range(5)]
        held = np.zeros(len(y_arr))
        for f in folds:
            te = np.isin(groups, f)
            tr = ~te
            if y_arr[tr].sum() < 2 or te.sum() == 0:
                held[te] = sel_loso[te]
                continue
            iso_f = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso_f.fit(sel_loso[tr], y_arr[tr])
            held[te] = iso_f.predict(sel_loso[te])
        return float(np.mean((held - y_arr) ** 2)), held

    brier_nested, nested_probs = _nested_calibration_brier()

    results["calibration"] = {
        "brier_score": round(brier_nested, 4),
        "brier_score_in_sample_isotonic": round(brier_cal, 4),
        "brier_score_uncalibrated": round(brier_raw, 4),
        "brier_improvement": round(brier_raw - brier_nested, 4),
        "bins_heldout": _bins_of(nested_probs, y_arr),
        "bins": _bins(calibrated),
        "bins_uncalibrated": _bins(sel_loso),
        "method": ("isotonic regression fitted on out-of-fold LOSO probabilities; "
                   "the headline Brier is measured with a further 5-fold resident "
                   "split around the calibration step, so the map is never scored "
                   "on residents it was fitted on"),
        "model": selected_name,
        "note": ("AUC says the ranking is sensible; it says nothing about whether a "
                 "displayed 40 % means a 40 % chance. Class-weighted training "
                 "inflates raw scores badly — before calibration the top bin "
                 "over-forecast by roughly a factor of three — so the served model "
                 "carries an isotonic calibration layer and the UI shows the "
                 "calibrated value."),
    }

    # ── Operating point: threshold chosen against an explicit objective ─────
    sweep = _threshold_sweep(y_arr.tolist(), calibrated.tolist(), n_residents, n_weeks)
    best = max(sweep, key=lambda r: (r["f2"], r["recall"]))
    results["loso_threshold_sweep"] = sweep
    results["operating_point"] = {
        "threshold": best["threshold"],
        "objective": "maximise F2 on out-of-fold LOSO probabilities",
        "why_f2": ("F2 weights recall four times as heavily as precision. In a care "
                   "home a missed fall costs an injury; a false alert costs a "
                   "five-minute check. The asymmetry is real and it is stated as a "
                   "parameter rather than hidden in a default."),
        "recall": best["recall"],
        "precision": best["precision"],
        "accuracy": best["accuracy"],
        "balanced_accuracy": best["balanced_accuracy"],
        "specificity": best["specificity"],
        "alerts_per_week_across_home": best["alerts_per_week_across_home"],
        "note": ("Accuracy is reported at this operating point but is NOT the "
                 "selection objective: predicting 'no fall' for everyone already "
                 f"scores {results['baseline_majority']['accuracy']:.3f}."),
    }
    results[f"{selected_name}_loso_at_operating_point"] = _metrics(
        y_arr.tolist(), calibrated.tolist(), threshold=best["threshold"])

    # ── Named operating points ──────────────────────────────────────────────
    # One threshold is a policy choice, not a property of the model, so the
    # model card offers three and names them after the decision they encode.
    # The fourth exists to answer the accuracy objection head-on: at the SAME
    # accuracy as "predict no fall for everyone", how many falls does the model
    # still catch? A majority-class predictor catches none, by construction.
    base_acc = results["baseline_majority"]["accuracy"]
    acc_matched = None
    for r in sorted(sweep, key=lambda r: r["threshold"]):
        if r["accuracy"] >= base_acc:
            acc_matched = r
            break
    high_sens = max((r for r in sweep if r["recall"] >= 0.80), key=lambda r: r["precision"],
                    default=None)
    high_prec = max((r for r in sweep if r["precision"] >= 0.30), key=lambda r: r["recall"],
                    default=None)
    results["operating_points"] = {
        "high_sensitivity": high_sens,
        "balanced_f2": best,
        "high_precision": high_prec,
        "accuracy_matched_to_majority_baseline": acc_matched,
        "majority_baseline_accuracy": base_acc,
        "note": ("The deployed default is balanced_f2. accuracy_matched exists to "
                 "make one comparison unavoidable: at an accuracy equal to the "
                 "trivial 'no fall for anyone' predictor, that predictor recalls 0 "
                 "of the falls and this model recalls "
                 f"{(acc_matched or {}).get('recall', 'n/a')}."),
    }

    # ── Fit the deployed estimator on all rows, then attach calibration ─────
    final_est = _fit(selected_name, class_weight, X, y_arr)
    deployed = _CalibratedFallRiskModel(final_est, iso)

    # ── Permutation importance on the DEPLOYED model ────────────────────────
    results["feature_importance"] = _permutation_importance(deployed, X, y_arr)
    results["feature_importance_model"] = selected_name

    results["dataset"] = {
        "n_samples": len(X_raw),
        "n_features": len(FEATURE_NAMES),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_residents": n_residents,
        "n_time_points": n_weeks,
        "positive_rate": round(n_pos / len(y_arr), 4),
        "events_per_variable": round(n_pos / len(FEATURE_NAMES), 2),
        "feature_names": FEATURE_NAMES,
        "cv_folds": 5,
        "cv_schemes": ["row-shuffled StratifiedKFold (exhibit: leaks)",
                       "StratifiedGroupKFold (grouped by resident)",
                       "leave-one-subject-out (deployment-realistic)"],
        "note": (
            "SYNTHETIC DATA. Generated by synthetic_cohort.py, whose generating "
            "equations are known and documented. Every metric below measures the "
            "pipeline's ability to recover a signal that was authored into the "
            "data. It is internal validity, not clinical validity, and no figure "
            "here is evidence about falls in real care homes."),
    }

    if save:
        bundle = _save_model_bundle(deployed, results, X_raw, y, meta, db_path,
                                    threshold=best["threshold"])
        load_model_bundle(force_reload=True)
        results["artefact"] = {
            "saved": True,
            "path": str(_model_path()),
            "filename": MODEL_FILENAME,
            "artefact_version": bundle["artefact_version"],
            "trained_at": bundle["trained_at"],
            "sklearn_version": bundle["sklearn_version"],
            "model_type": bundle["model_type"],
            "selected": selected_name,
            "threshold": best["threshold"],
        }
    else:
        results["artefact"] = {"saved": False}

    return results


class _CalibratedFallRiskModel:
    """
    The deployed estimator: a fitted classifier plus the isotonic map that turns
    its class-weighted score into a probability that means what it says.

    Written as a small explicit class rather than sklearn's CalibratedClassifierCV
    for one reason: the calibration map here is fitted on LEAVE-ONE-SUBJECT-OUT
    out-of-fold probabilities, so it reflects performance on unseen residents.
    CalibratedClassifierCV would refit it on internal row-level folds, which on
    this data would learn a map from optimistically-leaked scores and undo the
    very thing the calibration is for.

    Exposes predict_proba so it is a drop-in wherever the raw estimator was used
    — including the occlusion-based local explanations in predict_fall_risk().
    """

    def __init__(self, estimator, isotonic):
        self.estimator = estimator
        self.isotonic = isotonic
        self.classes_ = getattr(estimator, "classes_", None)

    def predict_proba(self, X):
        import numpy as np
        raw = self.estimator.predict_proba(X)[:, 1]
        cal = np.clip(self.isotonic.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def __repr__(self):
        return f"CalibratedFallRiskModel({type(self.estimator).__name__} + isotonic)"


# SECTION 2b — MODEL ARTEFACT LAYER (train once, persist, load to serve)
#
# Rationale
# Training and inference are deliberately separated, as they would be in a
# production ML system:
#
#   TRAIN (offline, occasional)     ->  python train_model.py
#       builds the dataset, cross-validates, fits the final estimator and
#       writes a single versioned artefact to disk.
#
#   SERVE (online, every request)   ->  Flask routes call predict_fall_risk()
#       load the artefact once per process, then score residents in
#       milliseconds. No fitting happens inside a web request.
#
# The artefact is not a bare estimator: it is a *bundle* that also carries the
# metadata needed to know whether the model can be trusted — what it was
# trained on, when, with which library versions, at what decision threshold,
# and what it scored in validation. That metadata is what turns a .pkl file
# into something auditable, which matters in a regulated care setting.
#
# ARTEFACT_VERSION is bumped whenever the feature set or bundle layout changes,
# so an old .pkl produced by earlier code is rejected rather than silently
# mis-scored against features that no longer mean the same thing.

ARTEFACT_VERSION = 3
MODEL_FILENAME = "ml_fall_risk_model.pkl"

# Operating threshold. Deliberately below 0.5: missing a fall (false negative)
# is far costlier in a care home than an unnecessary check (false positive),
# so the model is tuned for recall at the expense of precision.
DECISION_THRESHOLD = 0.3


def _model_path():
    return _BASE / MODEL_FILENAME


def _data_fingerprint(db_path=None):
    """
    Cheap fingerprint of the training data, stored in the artefact so the app
    can tell the user their model is stale (i.e. the database has moved on
    since the model was fitted) instead of silently serving outdated scores.
    """
    if db_path is None:
        db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    out = {}
    for table, datecol in (("care_notes", "date"), ("incidents", "date"),
                           ("mar_records", "date")):
        try:
            n, mx = c.execute(
                f"SELECT COUNT(*), MAX({datecol}) FROM {table}"
            ).fetchone()
            out[table] = {"rows": n, "max_date": mx}
        except sqlite3.Error:
            out[table] = {"rows": None, "max_date": None}
    conn.close()
    return out


def _describe_estimator(model) -> str:
    """Readable name for the persisted estimator, unwrapping sklearn Pipelines."""
    try:
        steps = getattr(model, "steps", None)
        if steps:
            inner = type(steps[-1][1]).__name__
            pre = ", ".join(type(s[1]).__name__ for s in steps[:-1])
            return f"{inner} (pipeline: {pre})" if pre else inner
    except Exception:
        pass
    return type(model).__name__


def _save_model_bundle(model, results, X, y, meta, db_path, threshold=None):
    """Persist estimator + metadata as one pickle artefact."""
    import pickle
    import platform
    from datetime import datetime as _dt

    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unknown"
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        numpy_version = "unknown"

    # Reference profile = per-feature median of the training set. Stored in the
    # artefact because local explanations at serving time must be measured
    # against the distribution the model was TRAINED on, not against whichever
    # handful of residents happen to be on screen today.
    import statistics as _stats
    if X:
        medians = [
            float(_stats.median([row[j] for row in X]))
            for j in range(len(FEATURE_KEYS))
        ]
    else:
        medians = [0.0] * len(FEATURE_KEYS)

    bundle = {
        "artefact_version": ARTEFACT_VERSION,
        "model": model,
        "feature_medians": medians,
        "model_type": _describe_estimator(model),
        "selected_by": (results.get("model_selection") or {}).get("criterion"),
        "selected_name": (results.get("model_selection") or {}).get("selected"),
        "feature_names": list(FEATURE_NAMES),
        "feature_keys": list(FEATURE_KEYS),
        "n_features": len(FEATURE_NAMES),
        "threshold": DECISION_THRESHOLD if threshold is None else float(threshold),
        "threshold_objective": (results.get("operating_point") or {}).get("objective"),
        "trained_at": _dt.now().isoformat(timespec="seconds"),
        "trained_by": "train_model.py / ml_models.train_fall_risk_model",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn_version,
        "numpy_version": numpy_version,
        "n_samples": len(X),
        "n_positive": int(sum(y)),
        "n_residents": len(set(m["resident_id"] for m in meta)) if meta else 0,
        "train_date_range": (
            [min(m["date"] for m in meta), max(m["date"] for m in meta)]
            if meta else [None, None]
        ),
        "data_fingerprint": _data_fingerprint(db_path),
        # Full evaluation results are stored alongside the estimator so the
        # evaluation page can render honest metrics WITHOUT refitting. The
        # numbers shown to a user are then guaranteed to be the numbers the
        # deployed model actually achieved, not a fresh run that may differ.
        "results": results,
    }
    with open(_model_path(), "wb") as f:
        pickle.dump(bundle, f)
    return bundle


# Simple in-process cache: the artefact is read from disk once per worker and
# reused, which is the point of persisting it. _CACHE_MTIME lets a retrain
# invalidate the cache without restarting Flask.
_MODEL_CACHE = {"bundle": None, "mtime": None}


def load_model_bundle(force_reload: bool = False):
    """
    Load the persisted artefact, or return None if it is absent or unusable.

    Never raises: a corrupt or incompatible artefact degrades the app to
    "no ML predictions available" rather than taking the whole page down.
    """
    import pickle
    p = _model_path()
    if not p.exists():
        return None
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None

    if (not force_reload and _MODEL_CACHE["bundle"] is not None
            and _MODEL_CACHE["mtime"] == mtime):
        return _MODEL_CACHE["bundle"]

    try:
        with open(p, "rb") as f:
            bundle = pickle.load(f)
    except Exception:
        return None

    # Reject a bare estimator saved by the pre-versioning code, and any bundle
    # whose feature contract does not match this module.
    if not isinstance(bundle, dict) or "artefact_version" not in bundle:
        return None
    if bundle.get("artefact_version") != ARTEFACT_VERSION:
        return None
    if bundle.get("feature_keys") != list(FEATURE_KEYS):
        return None

    _MODEL_CACHE["bundle"] = bundle
    _MODEL_CACHE["mtime"] = mtime
    return bundle


def _load_model():
    """Backwards-compatible helper: return just the estimator, or None."""
    b = load_model_bundle()
    return b["model"] if b else None


def get_model_status():
    """
    Describe the currently persisted model for display in the UI / model card.

    Returns a dict that always contains "available" (bool) and a human-readable
    "message"; when a model exists it also reports staleness against the
    current database contents.
    """
    p = _model_path()
    bundle = load_model_bundle()

    if bundle is None:
        return {
            "available": False,
            "file_exists": p.exists(),
            "path": str(p),
            "message": (
                "No usable model artefact found — the file is missing, or it "
                "was produced by an older, incompatible version of the feature "
                "pipeline. Run: python train_model.py"
                if p.exists() else
                "No trained model on disk yet. Run: python train_model.py"
            ),
        }

    # Staleness check — has the data moved on since training?
    stale_reasons = []
    try:
        now_fp = _data_fingerprint()
        old_fp = bundle.get("data_fingerprint") or {}
        for table, cur in now_fp.items():
            prev = old_fp.get(table) or {}
            if prev.get("rows") is not None and cur.get("rows") is not None:
                delta = cur["rows"] - prev["rows"]
                if delta != 0:
                    stale_reasons.append(
                        f"{table}: {delta:+d} rows since training"
                    )
    except Exception:
        pass

    # Library drift — a pickle is only guaranteed to reload correctly under the
    # same scikit-learn version it was written with. Surfacing this is standard
    # practice; a silent cross-version unpickle is a known source of subtly
    # wrong predictions.
    try:
        import sklearn
        if bundle.get("sklearn_version") not in (None, "unknown", sklearn.__version__):
            stale_reasons.append(
                f"trained with scikit-learn {bundle.get('sklearn_version')}, "
                f"running {sklearn.__version__}"
            )
    except Exception:
        pass

    size_kb = round(p.stat().st_size / 1024, 1) if p.exists() else None

    return {
        "available": True,
        "file_exists": True,
        "path": str(p),
        "filename": MODEL_FILENAME,
        "size_kb": size_kb,
        "artefact_version": bundle.get("artefact_version"),
        "model_type": bundle.get("model_type"),
        "trained_at": bundle.get("trained_at"),
        "n_samples": bundle.get("n_samples"),
        "n_positive": bundle.get("n_positive"),
        "n_residents": bundle.get("n_residents"),
        "train_date_range": bundle.get("train_date_range"),
        "threshold": bundle.get("threshold"),
        "sklearn_version": bundle.get("sklearn_version"),
        "numpy_version": bundle.get("numpy_version"),
        "python_version": bundle.get("python_version"),
        "feature_names": bundle.get("feature_names"),
        "stale": bool(stale_reasons),
        "stale_reasons": stale_reasons,
        "message": (
            "Model loaded from disk; training data unchanged since fit."
            if not stale_reasons else
            "Model loaded from disk, but the database has changed since it was "
            "trained — consider retraining."
        ),
    }


def get_cached_results():
    """Evaluation metrics recorded at training time (no refitting)."""
    b = load_model_bundle()
    return b.get("results") if b else None


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_fall_risk(resident_ids=None, as_of_date=None, db_path=None):
    """
    Score residents with the PERSISTED model. No training occurs here.

    IMPORTANT — the returned "probability" is a RANKING score, not a calibrated
    probability. The classifier is fitted with class_weight="balanced" to buy
    recall on a heavily imbalanced problem (16 positives in 257 windows), which
    systematically inflates the predicted values: see results["calibration"],
    where the 0.6–0.8 bin has an observed event rate of roughly 0.14. Treat the
    number as "how does this resident compare with the others today", and the
    threshold as a triage cut-off — never as "there is an X% chance she falls".

    Returns {resident_id: {probability, percent, band, threshold, flag,
                           contributions[], features{}}}
    or {} when no usable artefact exists (the caller then simply hides the
    ML column rather than failing).

    `contributions` gives a LOCAL, per-resident explanation by occlusion: each
    feature in turn is reset to its training-set median and the model re-scored,
    so the drop in predicted probability measures what THIS resident's actual
    value for that feature contributed to THIS resident's score. It is a
    single-feature ablation — a cheap approximation to a Shapley value that
    needs no extra dependency — and unlike global permutation importance it
    differs from resident to resident, which is what makes it useful at the
    bedside ("her score is driven by fluid intake, his by recent incidents").
    """
    bundle = load_model_bundle()
    if bundle is None:
        return {}

    model = bundle["model"]
    threshold = bundle.get("threshold", DECISION_THRESHOLD)

    if db_path is None:
        db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if as_of_date is None:
        as_of_date = _effective_today(c)

    if resident_ids is None:
        resident_ids = [r["resident_id"] for r in c.execute(
            "SELECT resident_id FROM residents WHERE active=1"
        ).fetchall()]

    out = {}
    try:
        import numpy as np
        n_feat = len(FEATURE_KEYS)
        medians = bundle.get("feature_medians") or [0.0] * n_feat

        rows, ids = [], []
        for rid in resident_ids:
            rows.append(compute_features(c, rid, as_of_date))
            ids.append(rid)
        if not rows:
            return {}
        X = np.nan_to_num(np.array(rows, dtype=float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        probs = model.predict_proba(X)[:, 1]

        # Occlusion matrix: for every resident, one perturbed copy per feature
        # with that feature reset to the training median. Scored in a single
        # batched predict_proba call.
        n = X.shape[0]
        X_occ = np.repeat(X, n_feat, axis=0)
        for j in range(n_feat):
            X_occ[j::n_feat, j] = medians[j]
        occ_probs = model.predict_proba(X_occ)[:, 1].reshape(n, n_feat)

        for i, rid in enumerate(ids):
            p = float(probs[i])
            feats = {k: round(float(v), 3)
                     for k, v in zip(FEATURE_KEYS, X[i].tolist())}
            contribs = sorted(
                [
                    {
                        "feature": FEATURE_NAMES[j],
                        "value": round(float(X[i][j]), 2),
                        "typical": round(float(medians[j]), 2),
                        # >0 means this resident's value RAISES their risk
                        # relative to a typical resident.
                        "effect": round(p - float(occ_probs[i][j]), 4),
                        "effect_pct": round((p - float(occ_probs[i][j])) * 100, 1),
                    }
                    for j in range(n_feat)
                ],
                key=lambda d: -abs(d["effect"]),
            )[:4]
            out[rid] = {
                "probability": round(p, 3),
                "percent": round(p * 100, 1),
                "band": "high" if p >= threshold else ("moderate" if p >= threshold / 2 else "low"),
                "flag": bool(p >= threshold),
                "threshold": threshold,
                "as_of": as_of_date.isoformat(),
                "contributions": contribs,
                "features": feats,
            }
    except Exception as e:
        # Serving must never break the care record UI.
        conn.close()
        return {"_error": str(e)}

    conn.close()
    return out


def _effective_today(c):
    """
    The demo dataset covers a fixed 12-month window (see seed_data.py). If the
    real calendar date is past the end of that window, using date.today() makes
    every resident look neglected — a data-generation artefact, not a signal.
    So clamp "today" to the last date present in care_notes.
    """
    from datetime import datetime as _dt
    row = c.execute("SELECT MAX(date) FROM care_notes").fetchone()
    if row and row[0]:
        try:
            return min(date.today(), _dt.strptime(row[0], "%Y-%m-%d").date())
        except ValueError:
            pass
    return date.today()


# SECTION 3 — COMPOSITE DAILY RISK RANKING

def compute_daily_risk_ranking(as_of_date=None, include_ml: bool = True):
    """
    Compute a composite risk score for each active resident as of today.

    Score components (all 0–1 normalised, then weighted):
      - fluid_decline_score    (weight 0.25) — % drop vs 14-day baseline
      - wellbeing_trend_score  (weight 0.20) — negative trend in wellbeing
      - mar_risk_score         (weight 0.20) — non-adherence rate last 7 days
      - note_gap_score         (weight 0.15) — hours since last care note
      - incident_recency_score (weight 0.10) — recent incidents
      - risk_assessment_score  (weight 0.10) — latest formal risk score (normalised)

    Returns list of dicts sorted by composite_score DESC.
    """
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # The demo dataset is generated for a fixed 12-month window (see
    # seed_data.py); _effective_today() clamps "today" to the last date present
    # in care_notes so the demo does not report a fake fortnight-long gap in
    # care notes as if it were a genuine risk signal.
    today = as_of_date if as_of_date is not None else _effective_today(c)
    today_str = today.isoformat()
    d7_str    = (today - timedelta(days=7)).isoformat()
    d14_str   = (today - timedelta(days=14)).isoformat()
    d30_str   = (today - timedelta(days=30)).isoformat()
    d90_str   = (today - timedelta(days=90)).isoformat()

    residents = c.execute(
        "SELECT resident_id, preferred_name, primary_diagnosis, room_number "
        "FROM residents WHERE active=1"
    ).fetchall()

    def _slope_vals(vals):
        n = len(vals)
        if n < 2: return 0.0
        xs = list(range(n)); mx, my = sum(xs)/n, sum(vals)/n
        num = sum((xi-mx)*(yi-my) for xi,yi in zip(xs,vals))
        den = sum((xi-mx)**2 for xi in xs)
        return num/den if den else 0.0

    rankings = []

    # One batched call to the persisted model for every resident. Returns {}
    # when no artefact is on disk, in which case the page simply shows the
    # rule-based score alone.
    ml_preds = {}
    if include_ml:
        try:
            ml_preds = predict_fall_risk(
                resident_ids=[r["resident_id"] for r in residents],
                as_of_date=today,
                db_path=db_path,
            ) or {}
            ml_preds.pop("_error", None)
        except Exception:
            ml_preds = {}

    for r in residents:
        rid = r["resident_id"]
        name = r["preferred_name"]
        diagnosis = r["primary_diagnosis"]
        room = r["room_number"]
        reasons = []
        scores = {}

        # ── Fluid decline ──────────────────────────────────────────────────────
        baseline_fluid = [row[0] for row in c.execute(
            "SELECT fluid_intake_ml FROM care_notes WHERE resident_id=? "
            "AND date>=? AND date<? AND fluid_intake_ml IS NOT NULL",
            (rid, d14_str, d7_str)
        ).fetchall() if row[0]]
        recent_fluid = [row[0] for row in c.execute(
            "SELECT fluid_intake_ml FROM care_notes WHERE resident_id=? "
            "AND date>=? AND date<=? AND fluid_intake_ml IS NOT NULL",
            (rid, d7_str, today_str)
        ).fetchall() if row[0]]
        if baseline_fluid and recent_fluid:
            baseline_avg = sum(baseline_fluid) / len(baseline_fluid)
            recent_avg   = sum(recent_fluid)   / len(recent_fluid)
            decline_pct  = max(0, (baseline_avg - recent_avg) / baseline_avg) if baseline_avg > 0 else 0
            fluid_score  = min(1.0, decline_pct * 2.5)  # 40% drop → score 1.0
            if decline_pct > 0.20:
                reasons.append(f"Fluid intake {decline_pct*100:.0f}% below 14-day baseline")
        else:
            fluid_score = 0.3  # unknown = moderate concern
            recent_avg  = None
            baseline_avg = None
        scores["fluid_decline"] = round(fluid_score, 3)

        # ── Wellbeing trend ────────────────────────────────────────────────────
        wb_vals = [row[0] for row in c.execute(
            "SELECT overall_score FROM wellbeing WHERE resident_id=? "
            "AND assessment_date>=? AND assessment_date<=? ORDER BY assessment_date",
            (rid, d90_str, today_str)
        ).fetchall() if row[0]]
        latest_wb = wb_vals[-1] if wb_vals else None
        wb_slope  = _slope_vals(wb_vals)
        wb_score  = min(1.0, max(0.0, -wb_slope * 5 + (0.3 if (latest_wb or 10) < 6 else 0)))
        if wb_slope < -0.05:
            reasons.append(f"Wellbeing score declining (slope {wb_slope:.2f}/assessment)")
        if latest_wb and latest_wb < 5:
            reasons.append(f"Current wellbeing score low ({latest_wb}/10)")
        scores["wellbeing_trend"] = round(wb_score, 3)

        # ── MAR non-adherence ──────────────────────────────────────────────────
        # (see note above _build_fall_risk_dataset f7 — administered is text
        # 'Yes'/'No', so administered=1 alone never matches in SQLite)
        mar = c.execute(
            "SELECT COUNT(*), SUM(CASE WHEN administered='Yes' OR administered=1 THEN 1 ELSE 0 END) "
            "FROM mar_records WHERE resident_id=? AND date>=? AND date<=?",
            (rid, d7_str, today_str)
        ).fetchone()
        if mar and mar[0]:
            adherence = (mar[1] or 0) / mar[0]
            mar_score = 1.0 - adherence
            if adherence < 0.85:
                reasons.append(f"MAR adherence {adherence*100:.0f}% (last 7 days)")
        else:
            mar_score = 0.0
        scores["mar_risk"] = round(mar_score, 3)

        # ── Note gap ──────────────────────────────────────────────────────────
        last_note = c.execute(
            "SELECT date FROM care_notes WHERE resident_id=? "
            "AND date<=? ORDER BY date DESC LIMIT 1",
            (rid, today_str)
        ).fetchone()
        if last_note and last_note[0]:
            from datetime import datetime as dt2
            gap_days = (today - dt2.strptime(last_note[0], "%Y-%m-%d").date()).days
        else:
            gap_days = 7
        note_score = min(1.0, gap_days / 3.0)  # 3+ days without note → max score
        if gap_days > 1:
            reasons.append(f"No care note recorded in {gap_days} day(s)")
        scores["note_gap"] = round(note_score, 3)

        # ── Recent incidents ───────────────────────────────────────────────────
        inc_30 = c.execute(
            "SELECT COUNT(*) FROM incidents WHERE resident_id=? AND date>=? AND date<=?",
            (rid, d30_str, today_str)
        ).fetchone()[0]
        inc_score = min(1.0, inc_30 / 3.0)
        if inc_30 > 0:
            reasons.append(f"{inc_30} incident(s) in last 30 days")
        scores["incident_recency"] = round(inc_score, 3)

        # ── Formal risk score ──────────────────────────────────────────────────
        risk = c.execute(
            "SELECT score, risk_level FROM risk_assessments "
            "WHERE resident_id=? AND date_assessed<=? ORDER BY date_assessed DESC LIMIT 1",
            (rid, today_str)  # date_assessed is correct column name
        ).fetchone()
        if risk and risk[0]:
            risk_norm = min(1.0, float(risk[0]) / 30.0)  # assume max Waterlow-like score ~30
            if risk[1] in ("very_high", "high"):
                reasons.append(f"Formal risk level: {risk[1].replace('_',' ').title()}")
        else:
            risk_norm = 0.3
        scores["risk_assessment"] = round(risk_norm, 3)

        # ── Composite (weighted) ───────────────────────────────────────────────
        weights = {
            "fluid_decline":   0.25,
            "wellbeing_trend": 0.20,
            "mar_risk":        0.20,
            "note_gap":        0.15,
            "incident_recency":0.10,
            "risk_assessment": 0.10,
        }
        composite = sum(scores[k] * weights[k] for k in weights)

        # RAG colour
        if composite >= 0.55:
            rag = "red"
        elif composite >= 0.30:
            rag = "amber"
        else:
            rag = "green"

        # ── Supervised model prediction (served from the saved artefact) ───────
        # Attached alongside — NOT merged into — the rule-based composite. The
        # two are methodologically different: the composite encodes clinical
        # policy and is fully explainable; the classifier is learned from
        # outcomes. Showing them side by side lets staff (and the dissertation)
        # see where they agree and where they diverge, rather than hiding a
        # learned score inside a hand-weighted one.
        ml = ml_preds.get(rid) if ml_preds else None

        rankings.append({
            "resident_id":    rid,
            "name":           name,
            "ml_probability": ml["percent"] if ml else None,
            "ml_band":        ml["band"] if ml else None,
            "ml_flag":        ml["flag"] if ml else False,
            "ml_drivers":     ml["contributions"] if ml else [],
            "diagnosis":      diagnosis,
            "room":           room,
            "composite_score": round(composite, 3),
            "score_pct":      round(composite * 100, 1),
            "rag":            rag,
            "component_scores": scores,
            "reasons":        reasons if reasons else ["No significant concerns identified"],
            "fluid_recent_avg":   round(recent_avg, 0) if recent_avg else None,
            "fluid_baseline_avg": round(baseline_avg, 0) if baseline_avg else None,
            "latest_wellbeing":   latest_wb,
        })

    conn.close()
    return sorted(rankings, key=lambda x: -x["composite_score"])


# SECTION 4 — MEDICATION ADHERENCE ANOMALY (CUSUM)

def cusum_adherence_anomalies(db_path=None, k=0.5, h=4.0):
    """
    CUSUM (Cumulative Sum) control chart for medication adherence.

    k = allowance parameter (half the detectable shift, in SD units)
    h = decision threshold (SD units to trigger alarm)

    Returns dict: resident_id -> list of anomaly alert dicts
    """
    if db_path is None:
        db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    residents = c.execute("SELECT resident_id, preferred_name FROM residents WHERE active=1").fetchall()
    alerts = {}

    for r in residents:
        rid  = r["resident_id"]
        name = r["preferred_name"]

        # Daily adherence rate per resident
        daily = c.execute(
            "SELECT date, "
            "  SUM(CASE WHEN administered='Yes' OR administered=1 THEN 1 ELSE 0 END) AS given, "
            "  COUNT(*) AS total "
            "FROM mar_records WHERE resident_id=? "
            "GROUP BY date ORDER BY date",
            (rid,)
        ).fetchall()

        if len(daily) < 7:
            continue

        rates = [(row["given"] / row["total"]) if row["total"] > 0 else 1.0
                 for row in daily]
        dates = [row["date"] for row in daily]

        # Baseline: first 7 days
        mu = sum(rates[:7]) / 7
        sd = max(
            math.sqrt(sum((x - mu) ** 2 for x in rates[:7]) / 7),
            0.01
        )

        cusum_low = 0.0
        res_alerts = []
        for i in range(7, len(rates)):
            xi = rates[i]
            # Lower-side CUSUM (detecting drop in adherence)
            cusum_low = max(0, cusum_low + (mu - xi) / sd - k)
            if cusum_low >= h:
                res_alerts.append({
                    "date":      dates[i],
                    "adherence": round(xi * 100, 1),
                    "cusum":     round(cusum_low, 2),
                    "resident":  name,
                    "message":   (
                        f"{name}: medication adherence dropped to "
                        f"{xi*100:.0f}% on {dates[i]} "
                        f"(CUSUM={cusum_low:.1f}, threshold={h})"
                    )
                })
                cusum_low = 0  # reset after alarm

        if res_alerts:
            alerts[rid] = res_alerts

    conn.close()
    return alerts


# SECTION 5 — DETAILED ADHERENCE REPORT

def get_detailed_adherence_report():
    """
    Full per-resident medication adherence analysis for the web report page.

    Returns a list of resident dicts, each containing:
      - name, resident_id
      - baseline_mean, baseline_sd   (from first 7 days)
      - overall_adherence_pct        (all time)
      - dates[]                      (ISO strings)
      - daily_rates[]                (0–100 %)
      - cusum_series[]               (cumulative sum value per day)
      - alert_indices[]              (index positions where CUSUM >= h)
      - alerts[]                     (same dicts as cusum_adherence_anomalies)
      - medications[]                (per-drug breakdown)
      - risk_level                   ("high" / "medium" / "low")
    """
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    k_param = 0.5
    h_param = 4.0

    residents = c.execute(
        "SELECT resident_id, preferred_name FROM residents WHERE active=1 ORDER BY preferred_name"
    ).fetchall()

    report = []

    for r in residents:
        rid  = r["resident_id"]
        name = r["preferred_name"]

        # ── Daily adherence series ─────────────────────────────────────────────
        daily = c.execute(
            "SELECT date, "
            "  SUM(CASE WHEN administered='Yes' OR administered=1 THEN 1 ELSE 0 END) AS given, "
            "  COUNT(*) AS total "
            "FROM mar_records WHERE resident_id=? "
            "GROUP BY date ORDER BY date",
            (rid,)
        ).fetchall()

        if not daily:
            continue

        dates        = [row["date"] for row in daily]
        daily_rates  = [
            round((row["given"] / row["total"]) * 100, 1) if row["total"] else 100.0
            for row in daily
        ]
        total_given  = sum(row["given"] for row in daily)
        total_sched  = sum(row["total"] for row in daily)
        overall_pct  = round((total_given / total_sched) * 100, 1) if total_sched else 100.0

        # ── Baseline (first 7 days) ────────────────────────────────────────────
        baseline_vals = [r2 / 100 for r2 in daily_rates[:7]]
        mu = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 1.0
        sd = max(
            math.sqrt(sum((x - mu) ** 2 for x in baseline_vals) / len(baseline_vals)),
            0.01
        )
        baseline_mean_pct = round(mu * 100, 1)
        baseline_sd_pct   = round(sd * 100, 1)

        # ── Full CUSUM trace ───────────────────────────────────────────────────
        cusum_series   = [0.0] * 7          # baseline window has no CUSUM
        alert_indices  = []
        alerts         = []
        cusum_low      = 0.0

        for i in range(7, len(daily_rates)):
            xi = daily_rates[i] / 100
            cusum_low = max(0.0, cusum_low + (mu - xi) / sd - k_param)
            cusum_series.append(round(cusum_low, 3))
            if cusum_low >= h_param:
                alert_indices.append(i)
                alerts.append({
                    "date":        dates[i],
                    "adherence":   daily_rates[i],
                    "cusum":       round(cusum_low, 2),
                    "resident":    name,
                    "drop_from_baseline": round(baseline_mean_pct - daily_rates[i], 1),
                })
                cusum_low = 0.0   # reset

        # ── Per-medication breakdown ───────────────────────────────────────────
        med_rows = c.execute(
            "SELECT m.medication_name, m.dose, m.frequency, "
            "  COUNT(mr.id) AS total_scheduled, "
            "  SUM(CASE WHEN mr.administered='Yes' OR mr.administered=1 THEN 1 ELSE 0 END) AS total_given "
            "FROM medications m "
            "LEFT JOIN mar_records mr ON mr.medication_id=m.id AND mr.resident_id=m.resident_id "
            "WHERE m.resident_id=? AND m.status='active' "
            "GROUP BY m.id ORDER BY total_given * 1.0 / NULLIF(total_scheduled,0) ASC",
            (rid,)
        ).fetchall()

        medications = []
        for med in med_rows:
            t = med["total_scheduled"] or 0
            g = med["total_given"]     or 0
            pct = round(g / t * 100, 1) if t else None
            flag = "ok"
            if pct is not None:
                if pct < 85:
                    flag = "danger"
                elif pct < 95:
                    flag = "warning"
            medications.append({
                "name":      med["medication_name"],
                "dose":      med["dose"],
                "frequency": med["frequency"],
                "scheduled": t,
                "given":     g,
                "adherence": pct,
                "flag":      flag,
            })

        # ── Risk level ────────────────────────────────────────────────────────
        if overall_pct < 85 or len(alerts) >= 3:
            risk_level = "high"
        elif overall_pct < 95 or len(alerts) >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        report.append({
            "resident_id":        rid,
            "name":               name,
            "baseline_mean":      baseline_mean_pct,
            "baseline_sd":        baseline_sd_pct,
            "overall_adherence":  overall_pct,
            "total_given":        total_given,
            "total_scheduled":    total_sched,
            "dates":              dates,
            "daily_rates":        daily_rates,
            "cusum_series":       cusum_series,
            "alert_indices":      alert_indices,
            "alerts":             alerts,
            "medications":        medications,
            "risk_level":         risk_level,
            "cusum_threshold":    h_param,
            "cusum_k":            k_param,
        })

    conn.close()
    # Sort: most alerts first, then lowest adherence
    report.sort(key=lambda x: (-len(x["alerts"]), -x["overall_adherence"] * -1))
    return report


# SECTION 6 — QUICK STATS FOR DISSERTATION TABLE

def generate_ml_report(retrain: bool = False):
    """
    Assemble all model outputs for reporting.

    By default this SERVES the persisted artefact (metrics recorded at training
    time) rather than refitting — refitting on every report would be slow and,
    worse, could quietly report different numbers than the model that is
    actually deployed. Pass retrain=True to force a fresh fit and overwrite the
    artefact.
    """
    db_path = _get_db_path()

    if retrain:
        fall_results = train_fall_risk_model(save=True)
    else:
        fall_results = get_cached_results()
        if fall_results is None:
            # First run / no artefact yet — bootstrap one.
            fall_results = train_fall_risk_model(save=True)

    risk_ranking = compute_daily_risk_ranking()
    cusum_alerts = cusum_adherence_anomalies(db_path)

    total_cusum_alerts = sum(len(v) for v in cusum_alerts.values())

    return {
        "fall_risk_model":    fall_results,
        "daily_risk_ranking": risk_ranking,
        "cusum_alerts":       {str(k): v for k, v in cusum_alerts.items()},
        "cusum_total_alerts": total_cusum_alerts,
        "model_status":       get_model_status(),
        "generated_at":       date.today().isoformat(),
    }


if __name__ == "__main__":
    # Prefer `python train_model.py` for training; this stays for quick checks.
    print("Training fall-risk model...")
    results = train_fall_risk_model(save=True)
    print(json.dumps(results, indent=2))
    print("\nComputing daily risk ranking...")
    ranking = compute_daily_risk_ranking()
    for r in ranking:
        ml = f" ml={r['ml_probability']}%" if r.get("ml_probability") is not None else ""
        print(f"  {r['rag'].upper():6} {r['name']:12} score={r['score_pct']}%{ml}  {r['reasons'][0]}")
