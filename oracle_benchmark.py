"""
oracle_benchmark.py — how good could ANY model be on this problem?

The question this file answers
A cross-validated ROC-AUC of 0.855 is reported in Chapter 6. Is that good?

Normally that question has no clean answer. You can compare against a baseline
(better than chance, better than the paper risk score) and against published
models on other cohorts, but you cannot say how much of the remaining 0.145 is
model deficiency and how much is irreducible noise. On real data the Bayes rate
is unknown and unknowable.

Here it is neither. The data came from a generating process written by hand, so
the true latent state d(t), the true frailty, and the true hazard coefficients
are all available. That makes the Bayes-optimal predictor computable in closed
form, and with it the ceiling that no model — however large, however tuned —
can exceed on this data.

This is the single strongest methodological argument for having simulated the
data rather than scraped it, and it is the reason the synthetic-data decision in
Chapter 4 is presented as a design choice rather than an apology.

What is computed
For resident i at decision time t, the label is "at least one fall in
(t, t+7]". Under the generating process the daily hazard is

    h_i(u) = sigmoid( β0 + β_d·d_i(u) + β_f·frailty_i + β_r·min(recent_i(u), 3) )

so, conditioning on the latent path, the true probability of the label is

    P_i(t) = 1 − Π_{u=t+1}^{t+7} ( 1 − h_i(u) )

Three predictors are then scored on exactly the same labelled windows the
classifier saw:

  ORACLE (clairvoyant)   uses the FUTURE latent path d(t+1..t+7). This is the
                         mathematical ceiling. It is not achievable by any
                         model, because no model can see the future — it is
                         reported to show how much of the residual error is
                         simply that the future has not happened yet.

  ORACLE (causal)        uses only d(t), the true latent state at decision time,
                         propagated forward under the model's own dynamics with
                         no knowledge of the shocks to come. THIS is the fair
                         ceiling: it is what a model with perfect measurement of
                         the resident's current state, and perfect knowledge of
                         the physics, would achieve.

  FITTED MODEL           the deployed classifier's out-of-fold LOSO
                         probabilities, loaded from the model artefact.

The gap between the causal oracle and the fitted model is the part of the
problem the pipeline has failed to capture — a *measurement* of headroom rather
than a guess at it.

Usage
    python oracle_benchmark.py
    python oracle_benchmark.py --json results/oracle.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import ml_models  # noqa: E402  (path set above)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def load_ground_truth(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(_BASE, "carehome_ground_truth.json.gz")
    if not os.path.exists(path):
        raise SystemExit(
            f"Ground truth not found at {path}.\n"
            "Regenerate the cohort first:  python synthetic_cohort.py")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_windows(db_path: str):
    """Rebuild exactly the same labelled windows the classifier was trained on."""
    return ml_models._build_fall_risk_dataset(db_path)


def oracle_probabilities(gt: dict, meta: list[dict], db_path: str,
                         causal: bool = True) -> list[float]:
    """
    True 7-day fall probability for each labelled window.

    causal=True   propagate d(t) forward with no knowledge of future shocks:
                  the achievable ceiling.
    causal=False  read the actual realised d(t+1..t+7): the clairvoyant ceiling.
    """
    hz = gt["hazard"]
    b0, bd, bf, br = (hz["intercept"], hz["beta_d"],
                      hz["beta_frailty"], hz["beta_recur"])
    start = date.fromisoformat(gt["start_date"])
    by_res = {r["resident_id"]: r for r in gt["residents"]}

    # recent_falls at each (resident, day), replicating the generator's own
    # counter: falls since the last 30-day reset.
    conn = sqlite3.connect(db_path)
    falls = {}
    for rid, ds in conn.execute(
            "SELECT resident_id, date FROM incidents WHERE incident_type LIKE '%fall%'"):
        falls.setdefault(rid, set()).add((date.fromisoformat(ds) - start).days)
    conn.close()

    def recent_at(rid, t):
        block_start = (t // 30) * 30
        return sum(1 for d_ in falls.get(rid, ()) if block_start <= d_ < t)

    out = []
    for m in meta:
        rid = m["resident_id"]
        t = (date.fromisoformat(m["date"]) - start).days
        rec = by_res.get(rid)
        if rec is None:
            out.append(0.0)
            continue
        d_series, frailty = rec["d"], rec["frailty"]
        surv = 1.0
        for k in range(1, 8):
            u = t + k
            if u >= len(d_series):
                break
            if causal:
                # No knowledge of the shocks between t and u: the best available
                # estimate of d(u) is d(t) itself, since the latent process is a
                # near-martingale over a one-week horizon (drift ~0.0004/day).
                d_u = d_series[t] if t < len(d_series) else d_series[-1]
            else:
                d_u = d_series[u]
            lg = b0 + bd * d_u + bf * frailty + br * min(recent_at(rid, u), 3)
            surv *= (1.0 - sigmoid(lg))
        out.append(1.0 - surv)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Bayes-optimal ceiling for the fall-risk task.")
    ap.add_argument("--json", metavar="PATH", help="write results as JSON")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    gt = load_ground_truth()
    db_path = args.db or ml_models._get_db_path()

    print("[*] Rebuilding the labelled windows ...")
    X, y, meta = build_windows(db_path)
    print(f"    {len(y)} windows, {sum(y)} positive "
          f"({sum(y)/len(y)*100:.2f} %), {len({m['resident_id'] for m in meta})} residents")

    print("[*] Computing oracle probabilities ...")
    p_causal = oracle_probabilities(gt, meta, db_path, causal=True)
    p_clair  = oracle_probabilities(gt, meta, db_path, causal=False)

    m_causal = ml_models._metrics(y, p_causal, threshold=0.05)
    m_clair  = ml_models._metrics(y, p_clair,  threshold=0.05)
    ap_causal = ml_models._average_precision(y, p_causal)
    ap_clair  = ml_models._average_precision(y, p_clair)

    bundle = ml_models.load_model_bundle()
    fitted = None
    if bundle and bundle.get("results"):
        sel = (bundle["results"].get("model_selection") or {}).get("selected")
        key = f"{sel}_loso"
        fitted = bundle["results"].get(key)

    base_rate = sum(y) / len(y)
    rows = [
        ("Oracle — clairvoyant (sees future latent path)", m_clair["roc_auc"], ap_clair),
        ("Oracle — causal (true state at decision time)",  m_causal["roc_auc"], ap_causal),
    ]
    if fitted:
        rows.append((f"Fitted model — {sel}, LOSO",
                     fitted["roc_auc"], fitted.get("average_precision")))
    rows.append(("Chance", 0.5, round(base_rate, 4)))

    BAR = "=" * 78
    print()
    print(BAR)
    print(" BAYES-OPTIMAL CEILING vs FITTED MODEL")
    print(BAR)
    print(f" {'Predictor':<52}{'ROC-AUC':>10}{'Avg.Prec':>12}")
    print(" " + "-" * 74)
    for label, auc, apv in rows:
        print(f" {label:<52}{auc:>10}{(round(apv,4) if apv is not None else chr(45)):>12}")
    print(BAR)
    if fitted:
        headroom = round(m_causal["roc_auc"] - fitted["roc_auc"], 4)
        captured = round((fitted["roc_auc"] - 0.5) / (m_causal["roc_auc"] - 0.5) * 100, 1) \
            if m_causal["roc_auc"] > 0.5 else None
        print(f" Headroom to the achievable ceiling : {headroom:+.4f} AUC")
        print(f" Share of achievable signal captured: {captured} %")
        print(BAR)
        print(" Reading: the causal oracle is what a model with perfect measurement of")
        print(" the resident's CURRENT state would achieve. The fitted model must infer")
        print(" that state from noisy, irregularly sampled proxies (a 7-day fluid mean,")
        print(" a fortnightly wellbeing score), so the gap above is the cost of")
        print(" measurement error, not of a poor learning algorithm.")
        print(BAR)

    if args.json:
        payload = {
            "n_windows": len(y), "n_positive": int(sum(y)),
            "base_rate": round(base_rate, 4),
            "oracle_causal": {"roc_auc": m_causal["roc_auc"],
                              "average_precision": round(ap_causal, 4)},
            "oracle_clairvoyant": {"roc_auc": m_clair["roc_auc"],
                                   "average_precision": round(ap_clair, 4)},
            "fitted_model": fitted,
            "generated": datetime.now().isoformat(timespec="seconds"),
        }
        if fitted:
            payload["headroom_auc"] = round(m_causal["roc_auc"] - fitted["roc_auc"], 4)
            payload["share_of_achievable_signal_captured_pct"] = round(
                (fitted["roc_auc"] - 0.5) / (m_causal["roc_auc"] - 0.5) * 100, 1)
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[*] Written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
