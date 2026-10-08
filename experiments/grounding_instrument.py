"""
E4 — Validating the grounding instrument before trusting it.

The system claims that what it writes about a resident is built from that
resident's records. The claim is checked sentence by sentence: each sentence is
matched to the retrieved record that best supports it and scored on the share of
its content words present in that record (rag_evaluator.attribute_sentences).
Sentences at or above 0.60 are called supported, 0.35-0.60 partial, below that
unsupported.

Those thresholds and that score are a measuring instrument, and an instrument
that has not been calibrated is an opinion with decimal places. This experiment
calibrates it, on material where the right answer is known in advance.

Four arms, all drawn from the real corpus so the register is identical and only
the grounding differs:

  VERBATIM   sentences lifted from the retrieved evidence. The instrument must
             score these high; anything else is a false negative.
  PARAPHRASE the same sentences with a share of their content words swapped for
             clinical synonyms. The instrument is lexical, so it should degrade
             here — the question is how fast, because that decay is the size of
             its blind spot.
  OTHER      sentences from a DIFFERENT resident's records, retrieved for this
             resident. Real clinical prose, right register, wrong file. The
             instrument must score these low; anything else is a false positive
             and the metric is measuring genre rather than grounding.
  INVENTED   clinically plausible sentences containing entities that appear
             nowhere in the corpus — invented drugs, doses and events. This is
             the failure the whole exercise exists to catch.

Run:  python experiments/grounding_instrument.py
Out:  results/grounding_instrument.json
"""
import json, os, random, re, sqlite3, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_engine
from rag_evaluator import attribute_sentences, _split_sentences, _ATTR_SUPPORTED, _ATTR_PARTIAL

SEED   = 20260823
TOP_K  = 6
N_RESIDENTS = 50

# Swaps a clinician would make without changing what the sentence says. Kept
# deliberately small and one-directional: the point is to perturb wording while
# holding meaning fixed, not to build a thesaurus.
# Swaps a clinician would make without changing what the sentence says.
#
# Every pair is content-word to content-word. An earlier version mapped words
# that rag_evaluator treats as stop words ("given", "noted", "needs",
# "resident") onto words it does not, which ADDS a term to the denominator of
# the support score and depresses it even when the meaning is untouched:
# "Routine care given" scored 1.000 and "Routine care administered" scored
# 0.500 on the same evidence, from a swap that changed nothing. That is a
# property of the stop list, not of paraphrase, and it is corrected here so the
# paraphrase arms measure what they claim to.
STOP = {"given", "noted", "needs", "resident", "care", "staff", "person",
        "taken", "ensure", "refer", "plan", "section", "home", "individual"}

SYNONYMS = {
    "assisted": "supported", "assistance": "support", "supported": "assisted",
    "declined": "refused", "refused": "declined", "settled": "comfortable",
    "unsettled": "restless", "mobility": "walking", "walking": "mobility",
    "fluids": "drinks", "fluid": "drink", "intake": "consumption",
    "medication": "medicine", "administered": "dispensed",
    "observed": "recorded", "supervision": "oversight", "transfers": "moves",
    "encouragement": "prompting", "prompting": "encouragement",
    "appetite": "eating", "wash": "hygiene", "morning": "breakfast-time",
    "afternoon": "midday", "evening": "night", "continence": "toileting",
    "frame": "walker", "assessed": "evaluated", "required": "necessary",
    "maintained": "sustained", "reported": "stated", "attended": "joined",
    "activity": "session", "mood": "affect", "alert": "responsive",
    "comfortable": "settled", "hydration": "fluids", "nutrition": "diet",
}
# Belt and braces: refuse any pair that crosses the stop-word boundary.
SYNONYMS = {k: v for k, v in SYNONYMS.items() if k not in STOP and v not in STOP}

# Fabricated content: clinically plausible, in the corpus's register, but
# naming drugs, instruments, events and values that appear nowhere in it.
INVENTED = [
    "Commenced on amoxicillin 500 mg three times daily following a chest infection diagnosed by the visiting GP.",
    "A Waterlow score of 22 was recorded at the pressure area review on the fourteenth.",
    "Referred to the community falls team after a witnessed collapse in the dining room.",
    "Rivaroxaban was withheld for forty-eight hours ahead of the dental extraction.",
    "The occupational therapist recommended a perching stool and a second stair rail.",
    "Blood glucose was 4.1 mmol/L before breakfast and the insulin dose was reduced accordingly.",
    "A safeguarding alert was raised with the local authority following a family concern.",
    "The district nurse redressed a grade two pressure ulcer on the left heel.",
    "Weight has fallen by 3.4 kg over the last quarter and a MUST score of 3 was recorded.",
    "A urine dipstick showed leucocytes and nitrites and a sample was sent for culture.",
    "Speech and language therapy advised level four pureed texture and thickened drinks.",
    "An ECG taken after the episode showed atrial fibrillation at 118 beats per minute.",
    "Codeine phosphate 30 mg was prescribed for breakthrough pain in the left hip.",
    "The podiatrist attended and debrided a callus on the right forefoot.",
    "A DoLS authorisation was granted for twelve months by the supervisory body.",
    "Bloods taken on Tuesday showed a haemoglobin of 96 g/L and iron studies were requested.",
    "The dietitian commenced Fortisip twice daily following a MUST score of 2.",
    "A urinary catheter was inserted after 900 ml was drained on bladder scan.",
    "Physiotherapy assessed transfers and issued a stand-aid for two-carer use.",
    "The GP stopped amlodipine after a lying and standing blood pressure drop of 28 mmHg.",
    "An abdominal x-ray was arranged following four days without a bowel movement.",
    "Trimethoprim 200 mg twice daily was started for a suspected urinary tract infection.",
    "The tissue viability nurse reviewed the sacrum and advised a dynamic mattress.",
    "A swallow assessment was completed after coughing was observed on thin fluids.",
    "Warfarin was held and an INR of 5.2 was reported by the anticoagulation clinic.",
]

# The hardest case for a lexical measure: real entities from the corpus, wrong
# values, wrong dates, wrong direction. Built per resident at run time from the
# retrieved evidence itself, so the vocabulary is genuinely the resident's.
NEAR_MISS_TEMPLATES = [
    "Fluid intake was {wrong_ml} ml, well above the daily target, and no concern was recorded.",
    "{name} declined all personal care throughout the shift and refused to be assisted.",
    "{name} required full hoist transfer with two carers and could not weight-bear at any point.",
    "All medication was refused across the whole of {month} and the GP was not informed.",
    "{name} was unsettled and distressed for the entire night and did not sleep at all.",
    "Wellbeing was scored at 10 out of 10 at the review and no actions were identified.",
]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def paraphrase(sentence, rate, rng):
    """Swap content words for synonyms at the given rate, keeping structure."""
    def swap(m):
        w = m.group(0)
        key = w.lower()
        if key in SYNONYMS and rng.random() < rate:
            new = SYNONYMS[key]
            return new.capitalize() if w[0].isupper() else new
        return w
    return re.sub(r"\b[A-Za-z]+\b", swap, sentence)


def score_one(sentence, chunks):
    """Support score for a single sentence against a retrieved evidence set."""
    r = attribute_sentences(sentence, chunks)
    if not r["sentences"]:
        return None
    return r["sentences"][0]["support"]


def summarise(name, scores, expect_supported):
    if not scores:
        return None
    sup = sum(1 for s in scores if s >= _ATTR_SUPPORTED)
    par = sum(1 for s in scores if _ATTR_PARTIAL <= s < _ATTR_SUPPORTED)
    uns = sum(1 for s in scores if s < _ATTR_PARTIAL)
    n = len(scores)
    out = {
        "arm": name, "n": n,
        "mean_support": round(statistics.mean(scores), 4),
        "median_support": round(statistics.median(scores), 4),
        "sd": round(statistics.stdev(scores), 4) if n > 1 else 0.0,
        "share_supported": round(sup / n, 4),
        "share_partial": round(par / n, 4),
        "share_unsupported": round(uns / n, 4),
        "expected_verdict": "supported" if expect_supported else "unsupported",
    }
    out["error_rate"] = round((1 - sup / n) if expect_supported else (sup / n), 4)
    return out


def rank_auc(pos, neg):
    """Probability a grounded sentence outscores an ungrounded one (ties at 0.5)."""
    if not pos or not neg:
        return None
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)


def cluster_ci(by_resident, n_boot=500, seed=SEED):
    """
    95 % interval on the mean support, resampling RESIDENTS rather than
    sentences. Eight sentences from one resident's evidence set are not eight
    independent observations, and an interval that treats them as such is too
    narrow by roughly the square root of the cluster size.
    """
    keys = list(by_resident)
    if len(keys) < 3:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [by_resident[rng.choice(keys)] for _ in keys]
        flat = [v for grp in sample for v in grp]
        if flat:
            means.append(sum(flat) / len(flat))
    if not means:
        return None
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return {"lo": round(lo, 4), "hi": round(hi, 4),
            "n_residents": len(keys), "n_boot": n_boot}


def paired_summary(paired):
    """Within-sentence paraphrase decay: same sentences at all three rates."""
    if not paired:
        return None
    def stat(key):
        v = [p[key] for p in paired]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        return {"mean_drop": round(m, 4), "sd": round(sd, 4)}
    return {
        "n_sentences": len(paired),
        "drop_30": stat("drop_30"),
        "drop_60": stat("drop_60"),
        "drop_100": stat("drop_100"),
        "verdict_flip_rate_at_100": round(
            sum(1 for p in paired if p["flipped_100"]) / len(paired), 4),
        "note": (
            "The same sentences appear at all three perturbation rates, so the "
            "trend is a within-sentence decay rather than a comparison between "
            "three differently-selected samples. A sentence is included only if "
            "it actually changed at every rate."
        ),
    }


def main(db_path="carehome.db"):
    t0 = time.time()
    if not rag_engine.get_index_status().get("built"):
        print("Building the retrieval index first…")
        rag_engine.build_index(db_path, "tfidf")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    residents = [r["resident_id"] for r in conn.execute(
        "SELECT resident_id FROM residents WHERE active=1 ORDER BY resident_id").fetchall()][:N_RESIDENTS]
    names = {r["resident_id"]: r["preferred_name"] for r in conn.execute(
        "SELECT resident_id, preferred_name FROM residents").fetchall()}
    conn.close()

    rng = random.Random(SEED)
    arms = {"verbatim": [], "paraphrase_30": [], "paraphrase_60": [],
            "paraphrase_100": [], "other_resident": [], "near_miss": [],
            "invented": []}
    paired = []          # within-sentence drops, same sentences at every rate
    clusters = {k: {} for k in arms}   # per-resident, for clustered intervals
    examples = []
    invented_used = set()

    def record(arm, rid, score):
        arms[arm].append(score)
        clusters[arm].setdefault(rid, []).append(score)

    for rid in residents:
        donor = rng.choice([r for r in residents if r != rid])
        name = names.get(rid, rid)
        query = f"{name} daily care, mobility, fluids and recent changes"

        evidence = rag_engine.retrieve(query, resident_id=rid, top_k=TOP_K,
                                       mode="tfidf", exclude_source_types=["care_plan"])
        donor_ev = rag_engine.retrieve(f"{names.get(donor, donor)} daily care, mobility, "
                                       f"fluids and recent changes",
                                       resident_id=donor, top_k=TOP_K,
                                       mode="tfidf", exclude_source_types=["care_plan"])
        if len(evidence) < 2 or not donor_ev:
            continue

        pool = []
        for c in evidence:
            pool.extend(_split_sentences(c.get("text", "")))
        rng.shuffle(pool)

        for sent in pool[:8]:
            base = score_one(sent, evidence)
            if base is None:
                continue

            # Paired design: a sentence enters the paraphrase arms only if it
            # actually changes at EVERY rate, so the three arms contain the same
            # sentences and the trend is a within-sentence comparison rather
            # than a comparison between three self-selected populations.
            variants = {}
            for rate, key in ((0.3, "paraphrase_30"), (0.6, "paraphrase_60"),
                              (1.0, "paraphrase_100")):
                variants[key] = paraphrase(sent, rate, rng)
            if all(v != sent for v in variants.values()):
                scored = {k: score_one(v, evidence) for k, v in variants.items()}
                if all(v is not None for v in scored.values()):
                    for k, v in scored.items():
                        record(k, rid, v)
                    paired.append({
                        "base": base,
                        "drop_30":  round(base - scored["paraphrase_30"], 4),
                        "drop_60":  round(base - scored["paraphrase_60"], 4),
                        "drop_100": round(base - scored["paraphrase_100"], 4),
                        "flipped_100": scored["paraphrase_100"] < _ATTR_SUPPORTED <= base,
                    })
                    if len(examples) < 8:
                        examples.append({"arm": "paraphrase_100", "original": sent,
                                         "perturbed": variants["paraphrase_100"],
                                         "support_original": base,
                                         "support_perturbed": scored["paraphrase_100"]})
            record("verbatim", rid, base)

        # Another resident's real prose, scored against this resident's evidence.
        donor_pool = []
        for c in donor_ev:
            donor_pool.extend(_split_sentences(c.get("text", "")))
        rng.shuffle(donor_pool)
        for sent in donor_pool[:8]:
            sc = score_one(sent, evidence)
            if sc is not None:
                record("other_resident", rid, sc)

        # Near miss: this resident's own vocabulary, wrong facts.
        ev_text = " ".join(c.get("text", "") for c in evidence)
        mls = re.findall(r"\b(\d{3,4})\s*ml\b", ev_text, flags=re.I)
        wrong_ml = str(int(mls[0]) + 900) if mls else "2600"
        for tmpl in rng.sample(NEAR_MISS_TEMPLATES, 3):
            sent = tmpl.format(name=name, wrong_ml=wrong_ml, month=rng.choice(MONTHS))
            sc = score_one(sent, evidence)
            if sc is not None:
                record("near_miss", rid, sc)

        # Fabricated content.
        for sent in rng.sample(INVENTED, 3):
            invented_used.add(sent)
            sc = score_one(sent, evidence)
            if sc is not None:
                record("invented", rid, sc)

    summary = {
        "verbatim":       summarise("Verbatim from the evidence", arms["verbatim"], True),
        "paraphrase_30":  summarise("Paraphrased, 30 % of swappable words", arms["paraphrase_30"], True),
        "paraphrase_60":  summarise("Paraphrased, 60 % of swappable words", arms["paraphrase_60"], True),
        "paraphrase_100": summarise("Paraphrased, every swappable word", arms["paraphrase_100"], True),
        "other_resident": summarise("Another resident's real records", arms["other_resident"], False),
        "near_miss":      summarise("This resident's vocabulary, wrong facts", arms["near_miss"], False),
        "invented":       summarise("Invented clinical content", arms["invented"], False),
    }
    for key, block in summary.items():
        if block:
            block["resident_clustered_ci95"] = cluster_ci(clusters[key])
    if summary["invented"]:
        summary["invented"]["n_distinct_sentences"] = len(invented_used)
        summary["invented"]["note"] = (
            f"{len(arms['invented'])} scorings of {len(invented_used)} distinct "
            "sentences, each against a different resident's evidence. The "
            "effective sample for the claim is the distinct count."
        )
    if summary["near_miss"]:
        summary["near_miss"]["n_distinct_templates"] = len(NEAR_MISS_TEMPLATES)

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED, "top_k": TOP_K,
        "thresholds": {"supported": _ATTR_SUPPORTED, "partial": _ATTR_PARTIAL},
        "n_residents": len(residents),
        "arms": summary,
        "separation": {
            "verbatim_vs_invented": rank_auc(arms["verbatim"], arms["invented"]),
            "verbatim_vs_other_resident": rank_auc(arms["verbatim"], arms["other_resident"]),
            "paraphrase60_vs_other_resident": rank_auc(arms["paraphrase_60"], arms["other_resident"]),
            "paraphrase100_vs_invented": rank_auc(arms["paraphrase_100"], arms["invented"]),
            "verbatim_vs_near_miss": rank_auc(arms["verbatim"], arms["near_miss"]),
            "paraphrase100_vs_near_miss": rank_auc(arms["paraphrase_100"], arms["near_miss"]),
            "interpretation": (
                "Probability that a randomly chosen sentence from the first arm scores above "
                "one from the second. 1.0 is perfect separation, 0.5 is no discrimination."
            ),
        },
        "paired_paraphrase": paired_summary(paired),
        "examples": examples,
        "method": (
            "Seven arms drawn from the same corpus so register is held constant and only "
            "grounding varies. Care plan chunks are excluded from retrieval so that no arm "
            "can be scored against a document of its own genre. The three paraphrase arms "
            "contain the same sentences, admitted only when the perturbation changed the "
            "text at every rate, so the decay across them is a within-sentence comparison. "
            "Shares carry resident-clustered bootstrap intervals, because sentences scored "
            "against one resident's evidence set are not independent observations."
        ),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/grounding_instrument.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    for k, v in summary.items():
        if v:
            print(f"{v['arm']:<42} n={v['n']:>4}  mean {v['mean_support']:.3f}  "
                  f"supported {v['share_supported']:.3f}  error {v['error_rate']:.3f}")
    print("separation verbatim vs invented       :", out["separation"]["verbatim_vs_invented"])
    print("separation verbatim vs near miss      :", out["separation"]["verbatim_vs_near_miss"])
    print("separation verbatim vs other resident :", out["separation"]["verbatim_vs_other_resident"])
    print("separation paraphrase60 vs other      :", out["separation"]["paraphrase60_vs_other_resident"])
    print(f"{out['runtime_seconds']}s")


if __name__ == "__main__":
    main()
