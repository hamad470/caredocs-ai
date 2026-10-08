"""
How much does the fall-risk model owe to the size of the cohort?

The question this answers is not "is the model any good" — that is what
train_model.py reports — but "how much of what we are seeing is an artefact of
having too few residents". Two numbers are tracked at every cohort size:

  * LOSO ROC-AUC          — what the model scores on a resident it has never
                            seen, which is the only estimate that corresponds
                            to how the tool would actually be used.
  * the optimism gap      — row-shuffled 5-fold AUC minus resident-grouped
                            5-fold AUC. Both use the same number of folds and
                            therefore the same training-set size, so the only
                            thing that differs is whether overlapping windows
                            from one resident can straddle the split. Comparing
                            row-shuffled folds against leave-one-subject-out
                            instead would confound leakage with training-set
                            size, since LOSO trains on 98 % of subjects and
                            5-fold on 80 % of rows; that comparison is recorded
                            separately as `loso_delta` and is not the gap.

Nothing here is calibrated and no threshold is searched. Both are properties of
the deployed model rather than of the estimate under study, and adding them
would multiply the runtime without changing any quantity reported.

Residents are drawn without replacement at each cohort size, repeated across
several draws, so the spread reported is variation between cohorts and not
noise from a single lucky sample. The full 50-resident point admits only one
draw, so it has no spread by construction and is marked as such.

Run:  python experiments/learning_curve.py
Out:  results/learning_curve.json
"""
import json, os, random, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, average_precision_score

COHORT_SIZES = [5, 10, 15, 20, 25, 30, 40, 50]
MIN_EVENTS   = 5       # a draw with fewer positives cannot support an AUC
SEED_REPEATS = 25      # fold seeds used for the stability check
DRAWS        = 6
SEED         = 20260823
CACHE        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dataset_cache.npz")


def load_dataset(db_path="carehome.db"):
    """Feature extraction is the expensive step, so it is cached to disk."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["X"], z["y"], z["groups"]
    import ml_models
    X_raw, y, meta = ml_models._build_fall_risk_dataset(db_path)
    X = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray([m["resident_id"] for m in meta])
    np.savez_compressed(CACHE, X=X, y=y, groups=groups)
    return X, y, groups


def make(name, class_weight):
    if name == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.01, max_iter=2000,
                                       class_weight=class_weight, random_state=42)),
        ])
    return RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=5,
                                  class_weight=class_weight, random_state=42, n_jobs=-1)


def fit_predict(name, X_tr, y_tr, X_te):
    """Fit on a training slice and return P(fall) for the test slice."""
    if y_tr.sum() < 1 or (len(y_tr) - y_tr.sum()) < 1:
        return np.full(len(X_te), float(y_tr.mean()) if len(y_tr) else 0.0)
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    model = make(name, {0: w[0], 1: w[1]})
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def cv_probs(name, X, y, groups, scheme, seed):
    """Out-of-fold probabilities under one of the three splitting schemes."""
    p = np.zeros(len(y), dtype=float)
    if scheme == "loso":
        for held in np.unique(groups):
            te = groups == held
            p[te] = fit_predict(name, X[~te], y[~te], X[te])
        return p

    if scheme == "grouped":
        n = min(5, len(np.unique(groups)))
        if n < 2:
            return None
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        folds = splitter.split(X, y, groups)
    else:                                    # row-shuffled: the optimistic one
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = splitter.split(X, y)

    for tr, te in folds:
        p[te] = fit_predict(name, X[tr], y[tr], X[te])
    return p


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")


def seed_stability(X, y, groups, repeats=SEED_REPEATS):
    """
    How much of the optimism gap is just the fold seed?

    On the full cohort the gap is small, and a small number is only meaningful
    if the measurement is finer than the number. Both 5-fold schemes are
    therefore repeated over many fold seeds and the spread reported, so a reader
    can see whether an observed gap of a few thousandths is a result or is below
    the resolution of the instrument.
    """
    rows, grouped, gaps = [], [], []
    for r in range(repeats):
        seed = SEED + 1000 + r
        p_row = cv_probs("logistic_regression", X, y, groups, "rowshuffle", seed)
        p_grp = cv_probs("logistic_regression", X, y, groups, "grouped", seed)
        if p_grp is None:
            continue
        a, b = safe_auc(y, p_row), safe_auc(y, p_grp)
        rows.append(a)
        grouped.append(b)
        gaps.append(a - b)
    if not gaps:
        return None

    def stat(v):
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        return {"mean": round(m, 4), "sd": round(sd, 4),
                "min": round(min(v), 4), "max": round(max(v), 4)}

    return {
        "n_fold_seeds": len(gaps),
        "rowshuffle_auc": stat(rows),
        "grouped_auc": stat(grouped),
        "optimism_gap": stat(gaps),
        "note": (
            "Full cohort, logistic regression, the two 5-fold schemes repeated "
            "over independent fold seeds. If the spread here is comparable to a "
            "reported gap, that gap is below the resolution of the estimate and "
            "should be reported as such rather than as a value."
        ),
    }


def main():
    t0 = time.time()
    X, y, groups = load_dataset()
    all_residents = sorted(set(groups.tolist()))
    n_features = X.shape[1]
    rng = random.Random(SEED)
    print(f"{len(y)} windows, {int(y.sum())} positive, "
          f"{len(all_residents)} residents, {n_features} features")

    curve, dropped = [], []
    for size in COHORT_SIZES:
        if size > len(all_residents):
            continue
        draws = 1 if size == len(all_residents) else DRAWS
        points = []
        attempted = draws
        for d in range(draws):
            chosen = (all_residents if draws == 1
                      else rng.sample(all_residents, size))
            mask = np.isin(groups, chosen)
            Xs, ys, gs = X[mask], y[mask], groups[mask]
            if ys.sum() < MIN_EVENTS:
                # Recorded rather than skipped silently: the discard is
                # conditioned on the event count, so dropping a draw quietly
                # would remove the lowest-event, highest-variance cohorts —
                # exactly the ones the experiment is about.
                dropped.append({"n_residents": size, "draw": d,
                                "n_positive": int(ys.sum()),
                                "reason": f"fewer than {MIN_EVENTS} events"})
                continue

            row = {
                "draw": d,
                "n_residents": size,
                "n_windows": int(len(ys)),
                "n_positive": int(ys.sum()),
                "positive_rate": round(float(ys.mean()), 4),
                "events_per_variable": round(float(ys.sum()) / n_features, 2),
            }
            for name in ("logistic_regression", "random_forest"):
                p_loso = cv_probs(name, Xs, ys, gs, "loso", SEED + d)
                p_row  = cv_probs(name, Xs, ys, gs, "rowshuffle", SEED + d)
                p_grp  = cv_probs(name, Xs, ys, gs, "grouped", SEED + d)
                loso_auc = safe_auc(ys, p_loso)
                row_auc  = safe_auc(ys, p_row)
                grp_auc = safe_auc(ys, p_grp) if p_grp is not None else None
                row[name] = {
                    "loso_auc": round(loso_auc, 4),
                    "rowshuffle_auc": round(row_auc, 4),
                    "grouped_auc": (round(grp_auc, 4) if grp_auc is not None else None),
                    "loso_average_precision": round(
                        float(average_precision_score(ys, p_loso)), 4),
                    # Like-for-like: same fold count, same training-set size.
                    "optimism_gap": (round(row_auc - grp_auc, 4)
                                     if grp_auc is not None else None),
                    # Not the gap — kept because it is what the naive
                    # comparison would report, and the difference is the point.
                    "loso_delta": round(row_auc - loso_auc, 4),
                }
            points.append(row)
            g = row["logistic_regression"]["optimism_gap"]
            gap_txt = f"{g:+.3f}" if g is not None else "  n/a"
            print(f"  n={size:>2} draw {d}: "
                  f"LOSO {row['logistic_regression']['loso_auc']:.3f}  "
                  f"gap {gap_txt}  ({row['n_positive']} events)")

        if not points:
            continue

        def agg(model, field):
            vals = [pt[model][field] for pt in points
                    if pt[model][field] == pt[model][field]]     # drop NaN
            if not vals:
                return None
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            return {"mean": round(mean, 4), "sd": round(sd, 4),
                    "min": round(min(vals), 4), "max": round(max(vals), 4),
                    "n_draws": len(vals)}

        curve.append({
            "n_residents": size,
            "single_draw": draws == 1,
            "draws_attempted": attempted,
            "draws_used": len(points),
            "draws_dropped": attempted - len(points),
            "mean_windows": round(sum(p["n_windows"] for p in points) / len(points), 1),
            "mean_positive": round(sum(p["n_positive"] for p in points) / len(points), 1),
            "mean_events_per_variable": round(
                sum(p["events_per_variable"] for p in points) / len(points), 2),
            "logistic_regression": {
                "loso_auc": agg("logistic_regression", "loso_auc"),
                "rowshuffle_auc": agg("logistic_regression", "rowshuffle_auc"),
                "grouped_auc": agg("logistic_regression", "grouped_auc"),
                "loso_average_precision": agg("logistic_regression", "loso_average_precision"),
                "optimism_gap": agg("logistic_regression", "optimism_gap"),
                "loso_delta": agg("logistic_regression", "loso_delta"),
            },
            "random_forest": {
                "loso_auc": agg("random_forest", "loso_auc"),
                "rowshuffle_auc": agg("random_forest", "rowshuffle_auc"),
                "grouped_auc": agg("random_forest", "grouped_auc"),
                "loso_average_precision": agg("random_forest", "loso_average_precision"),
                "optimism_gap": agg("random_forest", "optimism_gap"),
                "loso_delta": agg("random_forest", "loso_delta"),
            },
            "draws": points,
        })

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "draws_per_size": DRAWS,
        "n_features": n_features,
        "total_windows": int(len(y)),
        "total_positive": int(y.sum()),
        "total_residents": len(all_residents),
        "cohort_sizes": COHORT_SIZES,
        "runtime_seconds": round(time.time() - t0, 1),
        "curve": curve,
        "dropped_draws": dropped,
        "min_events_per_draw": MIN_EVENTS,
        "fold_seed_stability": seed_stability(X, y, groups),
        "method": (
            "At each cohort size, residents are drawn without replacement from the "
            "50-resident cohort and the whole evaluation is repeated on that subset. "
            "Three splitting schemes are run on every draw: leave-one-subject-out, "
            "resident-grouped 5-fold, and row-shuffled 5-fold. Spread is between draws, "
            "so it measures how much the answer depends on which residents you happen "
            "to have, not how much it depends on the random seed."
        ),
        "caveat": (
            "Every point comes from the same generated cohort, so this curve measures "
            "how the estimation procedure behaves as subjects are added — it is not "
            "evidence about how a real care home's data would behave. What transfers "
            "is the shape of the relationship, not the values on the axis."
        ),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/learning_curve.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote results/learning_curve.json in {out['runtime_seconds']}s")


if __name__ == "__main__":
    main()
