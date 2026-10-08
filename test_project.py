"""
test_project.py — regression tests for the v3 pipeline

Run:
    python test_project.py            # all suites
    python test_project.py --quick    # skip anything that refits or rebuilds
    python -m unittest test_project   # same tests via the standard runner

What is being protected
These are not tests of the web framework. They target the three places where
this project could produce a *plausible but wrong number* and nobody would
notice — which, in a data science project, is the failure that matters:

  A. THE GENERATING PROCESS.  If the coupling between latent state and hazard
     silently broke, every model result would still compute, still look
     reasonable, and be meaningless. Tested by asserting the coupling is
     measurable in the generated data, and that the marginal fall rate stays
     inside the published range it was calibrated against.

  B. THE CAUSAL CUT-OFF.  Features must use only data at or before the decision
     date. A single query with the wrong comparison operator would leak the
     future into the features and inflate every metric. Tested directly, by
     inserting a future record and asserting the features do not move.

  C. TRAINING/SERVING SKEW.  The offline matrix and the online prediction path
     must come from the same code. Tested by computing both and comparing.

  D. RESIDENT ISOLATION.  A retrieval filter that silently stops filtering is a
     data breach. Tested adversarially.

Every test states, in its docstring, what going wrong would look like.
"""
from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

DB = os.path.join(BASE, "carehome.db")
QUICK = os.environ.get("CAREHOME_TEST_QUICK") == "1"


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


# A — the generating process

class TestSyntheticCohort(unittest.TestCase):
    """The dataset is a model output, so it is tested like one."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DB):
            raise unittest.SkipTest("carehome.db not built — run setup_project.py")
        cls.c = _conn()

    def test_cohort_shape(self):
        """Fifty active residents over a full year. A short run would quietly
        change every event count downstream."""
        n = self.c.execute("SELECT COUNT(*) FROM residents WHERE active=1").fetchone()[0]
        self.assertEqual(n, 50, "expected a 50-resident cohort")
        d0, d1 = self.c.execute("SELECT MIN(date), MAX(date) FROM care_notes").fetchone()
        span = (date.fromisoformat(d1) - date.fromisoformat(d0)).days
        self.assertGreaterEqual(span, 360, "cohort should span roughly a year")

    def test_fall_rate_matches_published_range(self):
        """Calibration to FinCH (Logan et al., HTA 26(9), 2022): 2.2-3.8 falls
        per resident-year. Outside that band the cohort is no longer a plausible
        care home and no result computed on it can be defended."""
        n_res = self.c.execute("SELECT COUNT(*) FROM residents WHERE active=1").fetchone()[0]
        d0, d1 = self.c.execute("SELECT MIN(date), MAX(date) FROM care_notes").fetchone()
        years = n_res * ((date.fromisoformat(d1) - date.fromisoformat(d0)).days + 1) / 365.25
        falls = self.c.execute(
            "SELECT COUNT(*) FROM incidents WHERE incident_type LIKE '%fall%'").fetchone()[0]
        rate = falls / years
        self.assertGreaterEqual(rate, 2.2, f"fall rate {rate:.2f} below published range")
        self.assertLessEqual(rate, 3.8, f"fall rate {rate:.2f} above published range")

    def test_hazard_is_actually_coupled_to_observables(self):
        """THE test for this project. Residents who fall more must show lower
        fluid intake and lower wellbeing — that coupling is the entire reason a
        7-day-ahead prediction is learnable. If it broke, every model metric
        would still compute and all of them would be noise."""
        rows = self.c.execute("""
            SELECT r.resident_id,
              (SELECT COUNT(*) FROM incidents i WHERE i.resident_id=r.resident_id
                AND i.incident_type LIKE '%fall%') AS falls,
              (SELECT AVG(fluid_intake_ml) FROM care_notes n
                WHERE n.resident_id=r.resident_id) AS fluid,
              (SELECT AVG(overall_score) FROM wellbeing w
                WHERE w.resident_id=r.resident_id) AS wb
            FROM residents r WHERE r.active=1""").fetchall()
        falls = [r["falls"] for r in rows]
        fluid = [r["fluid"] for r in rows]
        wb = [r["wb"] for r in rows]
        self.assertLess(_pearson(falls, fluid), -0.2,
                        "falls should correlate NEGATIVELY with fluid intake")
        self.assertLess(_pearson(falls, wb), -0.2,
                        "falls should correlate NEGATIVELY with wellbeing")

    def test_missingness_is_mnar_as_designed(self):
        """Documentation thins as residents deteriorate. This is deliberate, and
        a positive correlation here would mean the mechanism inverted."""
        rows = self.c.execute("""
            SELECT (SELECT COUNT(*) FROM care_notes n WHERE n.resident_id=r.resident_id) AS notes,
                   (SELECT COUNT(*) FROM incidents i WHERE i.resident_id=r.resident_id
                     AND i.incident_type LIKE '%fall%') AS falls
            FROM residents r WHERE r.active=1""").fetchall()
        self.assertLess(_pearson([r["falls"] for r in rows], [r["notes"] for r in rows]), 0.0,
                        "note count should FALL as fall count rises (MNAR by design)")

    def test_class_balance_was_not_engineered(self):
        """A balanced dataset here would mean the hazard had been tuned to make
        the model look good. The positive rate must stay clinically plausible."""
        import ml_models
        X, y, meta = ml_models._build_fall_risk_dataset(DB)
        rate = sum(y) / len(y)
        self.assertGreater(rate, 0.02, "positive rate implausibly low")
        self.assertLess(rate, 0.12, "positive rate suspiciously high — was the hazard tuned?")

    def test_care_plans_are_visible_to_the_status_filter(self):
        """The v1 defect: plans were written with status 'approved' while every
        query looked for 'Active', so no care plan was ever indexed or shown."""
        n = self.c.execute(
            "SELECT COUNT(*) FROM care_plans WHERE "
            "LOWER(COALESCE(status,'')) IN ('active','approved','current')").fetchone()[0]
        self.assertEqual(n, 50, "every resident should have one visible care plan")

    def test_generation_is_reproducible(self):
        """Same seed, same cohort. Without this, no reported number can be
        re-derived by a marker or an examiner."""
        if QUICK:
            self.skipTest("--quick")
        import synthetic_cohort as sc
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "a.db"), os.path.join(td, "b.db")
            sa = sc.generate(a, n_residents=5, days=60, seed=99, verbose=False)
            sb = sc.generate(b, n_residents=5, days=60, seed=99, verbose=False)
            sa.pop("_truth"); sb.pop("_truth")
            sa.pop("_meta"); sb.pop("_meta")
            sa.pop("_ground_truth_path", None); sb.pop("_ground_truth_path", None)
            self.assertEqual(sa, sb, "identical seeds produced different cohorts")

    def test_cohort_size_does_not_change_individual_residents(self):
        """Per-resident random streams: resident 3 must be the same person
        whether the cohort has 5 members or 50. Without this, cohort size cannot
        be varied as an experimental factor."""
        if QUICK:
            self.skipTest("--quick")
        import synthetic_cohort as sc
        with tempfile.TemporaryDirectory() as td:
            small = os.path.join(td, "s.db")
            large = os.path.join(td, "l.db")
            sc.generate(small, n_residents=3, days=40, seed=7, verbose=False)
            sc.generate(large, n_residents=8, days=40, seed=7, verbose=False)
            q = ("SELECT full_name, age, falls_risk FROM residents "
                 "WHERE resident_id='RES003'")
            a = sqlite3.connect(small).execute(q).fetchone()
            b = sqlite3.connect(large).execute(q).fetchone()
            self.assertEqual(a, b, "resident 3 differs between cohort sizes")


# B + C — features, causality, training/serving parity

class TestFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DB):
            raise unittest.SkipTest("carehome.db not built")
        import ml_models
        cls.M = ml_models

    def test_feature_contract_is_internally_consistent(self):
        """Names, keys and the returned vector must agree in length and order,
        or every explanation shown in the UI is mislabelled."""
        self.assertEqual(len(self.M.FEATURE_NAMES), len(self.M.FEATURE_KEYS))
        c = _conn().cursor()
        v = self.M.compute_features(c, "RES001", date(2026, 6, 1))
        self.assertEqual(len(v), len(self.M.FEATURE_KEYS))
        self.assertTrue(all(isinstance(x, float) for x in v))

    def test_features_are_causally_valid(self):
        """Insert a dramatic record AFTER the decision date. If any feature
        moves, the future is leaking into the past and every metric in the
        dissertation is inflated."""
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "c.db")
            with open(DB, "rb") as src, open(tmp, "wb") as dst:
                dst.write(src.read())
            conn = sqlite3.connect(tmp)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            as_of = date(2026, 5, 1)
            before = self.M.compute_features(c, "RES001", as_of)

            future = (as_of + timedelta(days=3)).isoformat()
            c.execute("INSERT INTO care_notes (resident_id,date,shift,staff_name,"
                      "staff_role,note_type,fluid_intake_ml,care_narrative,status) "
                      "VALUES ('RES001',?,'Morning','Test Staff','Care Worker',"
                      "'Daily',60,'catastrophic future note','approved')", (future,))
            c.execute("INSERT INTO incidents (incident_id,resident_id,date,time,"
                      "shift,incident_type,severity,location,witnessed,"
                      "staff_first_on_scene,description,immediate_actions,status) "
                      "VALUES ('INC-FUTURE','RES001',?,'10:00','Morning','Fall',"
                      "'Major','Bedroom','No','Test Staff','future fall',"
                      "'none','closed')", (future,))
            conn.commit()

            after = self.M.compute_features(c, "RES001", as_of)
            conn.close()
            self.assertEqual(before, after,
                             "a record dated AFTER the decision date changed the features")

    def test_no_training_serving_skew(self):
        """The offline matrix and the online prediction path must call the same
        feature code. Divergence here is the classic silent ML failure: offline
        metrics look fine, live predictions are quietly wrong."""
        X, y, meta = self.M._build_fall_risk_dataset(DB)
        self.assertGreater(len(X), 100)
        conn = _conn()
        c = conn.cursor()
        for row, m in list(zip(X, meta))[:25]:
            direct = self.M.compute_features(
                c, m["resident_id"], date.fromisoformat(m["date"]), window=14)
            self.assertEqual([round(v, 6) for v in row],
                             [round(v, 6) for v in direct],
                             f"skew at {m['resident_id']} {m['date']}")
        conn.close()

    def test_labels_look_forward_only_seven_days(self):
        """A label that peeked further than the stated horizon would make the
        task easier than advertised."""
        X, y, meta = self.M._build_fall_risk_dataset(DB)
        conn = _conn()
        c = conn.cursor()
        checked = 0
        for lbl, m in zip(y, meta):
            if lbl != 1:
                continue
            d0 = date.fromisoformat(m["date"])
            n = c.execute(
                "SELECT COUNT(*) FROM incidents WHERE resident_id=? AND "
                "incident_type LIKE '%fall%' AND date>? AND date<=?",
                (m["resident_id"], d0.isoformat(), (d0 + timedelta(days=7)).isoformat())
            ).fetchone()[0]
            self.assertGreater(n, 0, "positive label with no fall in the 7-day window")
            checked += 1
            if checked >= 30:
                break
        conn.close()
        self.assertGreater(checked, 0, "no positive labels found to check")

    def test_new_v3_features_are_populated(self):
        """The five v3 features must actually vary. A feature that is constant
        across the cohort contributes nothing and would silently pad the count."""
        X, y, meta = self.M._build_fall_risk_dataset(DB)
        for key in ("fluid_delta_7v14", "fluid_cv_14d", "adherence_delta_7v30",
                    "days_since_last_fall", "wellbeing_drop_60d"):
            j = self.M.FEATURE_KEYS.index(key)
            col = [r[j] for r in X]
            self.assertGreater(len(set(col)), 5, f"feature {key} is near-constant")


# The model artefact and its serving path

class TestModelArtefact(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import ml_models
        cls.M = ml_models
        cls.bundle = ml_models.load_model_bundle(force_reload=True)
        if cls.bundle is None:
            raise unittest.SkipTest("no model artefact — run train_model.py")

    def test_artefact_carries_its_provenance(self):
        """A .pkl with no record of what it was trained on cannot be audited,
        which in a care setting means it cannot be deployed."""
        for key in ("artefact_version", "trained_at", "sklearn_version",
                    "feature_keys", "threshold", "n_samples", "results"):
            self.assertIn(key, self.bundle, f"artefact missing {key}")
        self.assertEqual(self.bundle["feature_keys"], list(self.M.FEATURE_KEYS))

    def test_artefact_rejects_a_stale_feature_contract(self):
        """If the feature set changes, an old artefact must be REFUSED rather
        than silently scored against features that no longer mean the same."""
        original = list(self.M.FEATURE_KEYS)
        try:
            self.M.FEATURE_KEYS.append("a_feature_that_did_not_exist")
            self.assertIsNone(self.M.load_model_bundle(force_reload=True),
                              "stale artefact was accepted")
        finally:
            self.M.FEATURE_KEYS[:] = original
            self.M.load_model_bundle(force_reload=True)

    def test_predictions_are_calibrated_probabilities(self):
        """Serving must return values in [0,1] for every resident, and the
        deployed model must be the calibrated wrapper, not the raw estimator."""
        preds = self.M.predict_fall_risk()
        self.assertNotIn("_error", preds, preds.get("_error"))
        self.assertGreaterEqual(len(preds), 40)
        for rid, p in preds.items():
            self.assertGreaterEqual(p["probability"], 0.0)
            self.assertLessEqual(p["probability"], 1.0)
            self.assertIn(p["band"], ("low", "moderate", "high"))
            self.assertTrue(p["contributions"], f"{rid} has no explanation")
        self.assertIn("Calibrated", repr(self.bundle["model"]))

    def test_reported_auc_beats_both_baselines(self):
        """Better than chance is not the bar. The model must beat the paper
        risk-assessment score the home already fills in, or it has no purpose."""
        r = self.bundle["results"]
        sel = r["model_selection"]["selected"]
        auc = r[f"{sel}_loso"]["roc_auc"]
        self.assertGreater(auc, 0.70, "LOSO AUC below a useful threshold")
        self.assertGreater(auc, r["baseline_static_risk_score"]["roc_auc"],
                           "model does not beat the existing paper risk score")

    def test_auc_confidence_interval_excludes_chance(self):
        """A point estimate above 0.5 means nothing if the interval spans it —
        which is exactly what happened in the 5-resident version."""
        ci = self.bundle["results"]["auc_uncertainty"]["ci95_cluster_bootstrap"]
        self.assertIsNotNone(ci["lo"])
        self.assertGreater(ci["lo"], 0.5, f"95% CI includes chance: {ci}")

    def test_calibration_actually_improved(self):
        """The calibration layer must be justified by measurement, not asserted."""
        cal = self.bundle["results"]["calibration"]
        self.assertLess(cal["brier_score"], cal["brier_score_uncalibrated"],
                        "calibration made the Brier score worse")
        self.assertLess(cal["brier_score"], 0.08)

    def test_leakage_gap_is_small(self):
        """Row-shuffled CV, grouped CV and LOSO should agree at this cohort
        size. A wide gap would mean subject-level leakage has returned."""
        la = self.bundle["results"].get("leakage_analysis")
        if not la:
            self.skipTest("grouped CV unavailable")
        self.assertLess(abs(la["optimism_gap_rowshuffle_minus_loso"]), 0.05)

    def test_operating_point_is_inside_the_swept_grid(self):
        """A threshold sitting on the edge of the search grid is a boundary
        artefact, not a tuned value."""
        r = self.bundle["results"]
        sweep = [row["threshold"] for row in r["loso_threshold_sweep"]]
        th = r["operating_point"]["threshold"]
        self.assertIn(th, sweep)
        self.assertGreater(th, min(sweep), "threshold sits on the lower grid edge")


# D — retrieval

class TestRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import rag_advanced
        cls.R = rag_advanced
        if rag_advanced._load() is None:
            raise unittest.SkipTest("no v2 index — run setup_project.py")

    def test_corpus_covers_every_record_type_and_resident(self):
        """A missing source type means a whole class of evidence is invisible to
        the assistant — the v1 care-plan defect, generalised."""
        cache = self.R._load()
        types = {c["source_type"] for c in cache["chunks"]}
        for t in ("care_note", "care_plan", "incident", "risk", "wellbeing",
                  "handover", "family_comm", "medication", "profile"):
            self.assertIn(t, types, f"no {t} chunks in the index")
        self.assertEqual(len({c["resident_id"] for c in cache["chunks"]}), 50)

    def test_resident_filter_is_absolute(self):
        """Adversarial: ask about resident A while filtering to resident B."""
        cache = self.R._load()
        residents = sorted({c["resident_id"] for c in cache["chunks"]})
        a, b = residents[0], residents[1]
        name_a = next(c["resident_name"] for c in cache["chunks"]
                      if c["resident_id"] == a)
        leaked = 0
        total = 0
        for q in (f"{name_a} fall bruising bedroom",
                  f"{name_a} fluid intake appetite decline",
                  f"{name_a} medication refused nausea",
                  f"{name_a} wellbeing score mood withdrawn"):
            res = self.R.retrieve(q, resident_ids=[b], top_k=8)
            total += len(res)
            leaked += sum(1 for r in res if r["resident_id"] != b)
        self.assertGreater(total, 0, "filter returned nothing at all")
        self.assertEqual(leaked, 0, f"{leaked} chunks leaked across residents")

    def test_date_filter_is_respected(self):
        """A date-scoped question must not be answered with out-of-window
        evidence — a figure quoted for the wrong quarter is simply wrong."""
        res = self.R.retrieve("fall incident", top_k=10,
                              date_from="2026-01-01", date_to="2026-03-31")
        self.assertTrue(res)
        for r in res:
            if r.get("date"):
                self.assertGreaterEqual(r["date"], "2026-01-01")
                self.assertLessEqual(r["date"], "2026-03-31")

    def test_retrieval_finds_a_known_item(self):
        """Sanity floor: given terms lifted from a chunk, that chunk should come
        back inside the top 8 most of the time."""
        import rag_experiments as E
        cache = self.R._load()
        qs = E.build_query_set(cache["chunks"], 60, seed=1234)
        hits = sum(1 for q in qs
                   if E._rank_of(self.R.retrieve(q["query"], top_k=8,
                                                 resident_ids=[q["resident_id"]]),
                                 q["gold_chunk_id"]) is not None)
        self.assertGreater(hits / len(qs), 0.35,
                           "known-item recall@8 has collapsed")

    def test_context_header_is_in_the_indexed_text(self):
        """The header must be inside what gets vectorised, not merely displayed
        alongside it — that distinction is the whole technique."""
        cache = self.R._load()
        c = cache["chunks"][0]
        self.assertTrue(c["indexed_text"].startswith(c["header"]))
        self.assertNotEqual(c["indexed_text"], c["text"])

    def test_v1_index_contains_care_plans(self):
        """Regression test for the fixed status-vocabulary defect."""
        import rag_engine
        docs = rag_engine._load_documents_from_db(DB)
        self.assertGreater(sum(1 for d in docs if d.get("source_type") == "care_plan"), 0,
                           "v1 index still excludes every care plan")


class TestEvaluatorFixes(unittest.TestCase):

    def test_precision_at_k_no_longer_returns_a_false_zero(self):
        """The v1 guard clause returned a hard 0.0 whenever fewer than k chunks
        were retrieved, producing precision@5 = 0.0000 alongside precision@1 =
        0.83 — arithmetically impossible, and reported as a result."""
        import rag_evaluator
        one = [{"section": "mobility_care_plan", "source_type": "care_plan"}]
        got = rag_evaluator.precision_at_k(one, "mobility_care_plan", k_values=(1, 3, 5))
        self.assertEqual(got[1], 1.0)
        self.assertEqual(got[3], 1.0, "precision@3 collapsed on a short result list")
        self.assertEqual(got[5], 1.0, "precision@5 collapsed on a short result list")
        self.assertIsNone(rag_evaluator.precision_at_k([], "x", k_values=(1,))[1],
                          "an empty result set should be None, not a score of zero")


class TestAppBoots(unittest.TestCase):

    def test_flask_app_serves_the_key_pages(self):
        """End-to-end smoke test: log in and open every page that reads the
        model or the index."""
        os.environ.setdefault("FLASK_SECRET", "test")
        import app as A
        A.app.config["TESTING"] = True
        with A.app.test_client() as cl:
            r = cl.post("/login", data={"username": "manager1",
                                        "password": "manager123"},
                        follow_redirects=True)
            self.assertEqual(r.status_code, 200)
            self.assertNotIn(b"Invalid", r.data[:4000])
            for path in ("/", "/residents", "/risk-ranking", "/ml-evaluation",
                         "/analytics", "/incidents", "/medications", "/chat"):
                resp = cl.get(path, follow_redirects=True)
                self.assertEqual(resp.status_code, 200, f"{path} -> {resp.status_code}")


def _pearson(xs, ys) -> float:
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip tests that regenerate or refit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args, rest = ap.parse_known_args()
    if args.quick:
        os.environ["CAREHOME_TEST_QUICK"] = "1"
        global QUICK
        QUICK = True
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
