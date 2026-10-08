# CareDocs AI

**Automatic care-home documentation and fall-risk prediction.**

[![CI](https://github.com/hamad470/caredocs-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/hamad470/caredocs-ai/actions/workflows/ci.yml)
[![Live demo](https://github.com/hamad470/caredocs-ai/actions/workflows/pages.yml/badge.svg)](https://hamad470.github.io/caredocs-ai/)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A working MVP of a care-home documentation system: staff record care notes,
incidents, medications, wellbeing and risk assessments, and the system drafts
the structured reports a UK care home needs, ranks residents by short-term fall
risk, and answers questions over the records with grounded retrieval.

It runs on a free AI model (Google Gemini free tier) and **still works with no
API key at all**: every generated field falls back to a deterministic offline
template, so records are always complete and nothing leaves the machine.

Built as the practical component of an MSc Data Science dissertation
(Liverpool John Moores University).

### ▶ Live demo: [hamad470.github.io/caredocs-ai](https://hamad470.github.io/caredocs-ai/)

Ask questions over 25,017 care records in your browser. Every sentence in the
answer links to the record ID and line it came from, and every claim is checked
against that line. No install, no sign-up.

> **All data is synthetic.** No real resident, staff member or care home is
> represented. See [`data/README.md`](data/README.md).

## What it does

| Area | Features |
|---|---|
| Records | Residents, care notes (with approval), incidents, shift handovers, medications and MAR charts, wellbeing scores, risk assessments, care plans, family communications |
| Report generation | AI-drafted narratives in a fixed care-home structure, exported to **PDF** and **Excel**; offline template fallback |
| Fall-risk model | 7-day-ahead fall-risk ranking with calibrated probabilities, causally valid features (no look-ahead), evaluation dashboard |
| Ask the Records | Chat assistant over the records: hybrid retrieval (BM25 + LSA dense + reciprocal-rank fusion + MMR) plus a whitelist of safe analytics tools; answers cite source records |
| Evaluation | Retrieval metrics (Recall@k, MRR, nDCG), ROUGE-L, grounding checks, oracle performance ceiling |
| Access | Role-based login (manager, senior carer, carer), audit log |

## Live RAG demo: "Ask the records"

The demo in [`site/`](site/) is a complete retrieval-augmented generation
pipeline that runs as a static web page, with each stage shown on screen.

```mermaid
flowchart LR
    Q[Question] --> U[1. Understand<br/>resident, period,<br/>record type, intent]
    U --> F[2. Filter by metadata<br/>25,017 records → subset]
    F --> R[3. Rank with BM25<br/>+ MMR diversity]
    F --> C[Count / measure<br/>over all matches]
    R --> E[4. Evidence lines<br/>E1, E2 … with record ID + line no.]
    C --> E
    E --> W{5. Write answer}
    W -->|no AI| QA[Quote lines verbatim]
    W -->|Gemini| G[Model must cite an<br/>evidence ID per sentence]
    G --> V[6. Verify each claim<br/>words + numbers vs cited lines]
    QA --> A[Answer with citations<br/>→ record viewer]
    V --> A
```

**How citations work.** Every record is stored with a stable ID (e.g.
`INC-RES009-0189`, `CN-316`) and split into numbered lines. A citation such as
*INC-RES009-0189, line 2* opens the record viewer at that line and also shows
where it lives in the database (`carehome.db`, table `incidents`, row 47).

**How claims are checked.** For a Gemini answer, each sentence's content words
and numbers are looked up in the lines it cites. Numbers must match exactly.
Sentences are marked *Supported*, *Partly supported*, *Not found in source*,
*Number not in source* or *No valid citation*, with the missing words listed.
Counts and trends are calculated over every matching record rather than the
top results, and are labelled *Calculated*.

| Counting with citations | Catching an unsupported sentence |
|---|---|
| ![Count answer](docs/demo-count.png) | ![Verifier](docs/demo-verification.png) |

*The right-hand screenshot uses a deliberately wrong test answer to show the
verifier flagging an invented claim.*

**Answer modes.** *Quote the records* needs no AI and no key: the answer is
made of evidence lines copied word for word. *Write with Gemini* uses Google's
free tier, either with the visitor's own key (kept in their browser) or through
the optional [Cloudflare Worker](worker/README.md) that holds a shared key.

**Same engine as the app.** The documents come from the Flask app's
`rag_advanced.load_documents()`, and the JavaScript BM25 is tested to return the
same records with the same scores as the Python one (`site/test/`).

**Hosting.** [`.github/workflows/pages.yml`](.github/workflows/pages.yml)
regenerates the cohort, exports it with `build_rag_demo.py`, runs the JS tests
and publishes `site/` to GitHub Pages on every push. To run it locally:

```bash
python setup_project.py --skip-train
python build_rag_demo.py
cd site && python -m http.server 8000      # open http://localhost:8000
```

## Quick start

> **New to Python or running this for the first time?** Follow the step-by-step
> guide in **[INSTALL.md](INSTALL.md)** (Windows, macOS and Linux, with
> troubleshooting). All required libraries are listed in
> [`requirements.txt`](requirements.txt).

Python 3.11 or newer. No GPU needed.

```bash
git clone https://github.com/hamad470/caredocs-ai.git
cd caredocs-ai
python -m pip install -r requirements.txt
python setup_project.py --skip-train   # builds the database and indexes (~1 min)
python app.py
```

Open <http://127.0.0.1:5000> and sign in with a demo account:

| Role | Username | Password |
|---|---|---|
| Manager (full access) | `manager1` | `manager123` |
| Senior carer | `senior1` | `senior123` |
| Carer | `carer1` | `carer123` |

On Windows, `run.bat` does all of the above.

### Turning on AI generation (optional, free)

1. Create a free key at <https://aistudio.google.com/apikey> (no card needed).
2. Either paste it in **Settings → AI Settings** inside the app, run
   `setup_ai.bat`, or set an environment variable:
   ```bash
   export GEMINI_API_KEYS=your_key          # several keys: comma-separated
   ```
3. `python check_ai.py` confirms the key works.

Keys saved through the app go to `ai_config.json`, which is git-ignored.
See [`.env.example`](.env.example) for every setting.

## Reproducing the results

```bash
python setup_project.py --force                # retrain the model too (~4 min)
python setup_project.py --force --experiments  # plus all experiments (~15 min)
```

Everything is generated deterministically from seed 4242, so the numbers in the
dissertation can be reproduced from this code alone.

## Tests

```bash
python test_project.py                      # 29 tests: data, features, model, web app
AI_PROVIDER=template python test_chat_rag.py   # 40 checks: retrieval, tools, chat
python build_rag_demo.py --reference && cd site && node --test test/retrieval.test.mjs
                                            # 6 tests: JS/Python BM25 parity, citations, verifier
```

Both run on every push via GitHub Actions.

## Project structure

```
app.py, templates/        Flask web application
database.py               schema
synthetic_cohort.py       the data-generating process (synthetic cohort)
ml_models.py              features, training, validation, calibration
train_model.py            trains and writes ml_fall_risk_model.pkl
oracle_benchmark.py       performance ceiling from the hidden ground truth
rag_engine.py             baseline TF-IDF retrieval
rag_advanced.py           hybrid retrieval: BM25, SVD, fusion, MMR
rag_evaluator.py          retrieval and generation metrics
rag_experiments.py        retrieval experiments
chat_engine.py            conversational assistant (planner + answerer)
analytics_tools.py        whitelisted database functions the assistant may call
ai_service.py             narrative drafting with offline fallback
llm_client.py             Gemini client (urllib, no vendor SDK, key rotation)
ai_config.py, check_ai.py key storage and diagnostics
report_generator.py       PDF export
xlsx_generator.py         Excel export
email_service.py          family communication drafts
anonymize_dataset.py      export with identifiers removed
export_sample_data.py     writes the CSV sample in data/sample/
build_rag_demo.py         exports records with line numbers for the live demo
site/                     the live demo (static HTML/JS, no build step)
  js/query.js             question understanding: resident, period, type, intent
  js/retrieve.js          metadata filter, BM25, MMR, evidence lines, counts, trends
  js/answer.js            quoted/calculated answers, Gemini call, claim verifier
worker/                   optional Cloudflare Worker holding a shared Gemini key
experiments/              cohort-size and grounding experiments
data/                     data card and a browsable CSV sample
```

## Limitations

This is a research MVP, not a certified clinical system.

- Trained and evaluated on synthetic data only; performance on real care-home
  records is unknown.
- No subgroup / fairness analysis yet, and no drift monitoring.
- Demo passwords use unsalted SHA-256 and the Flask development server is used;
  neither is suitable for production. Use a proper WSGI server, salted hashing
  (e.g. `werkzeug.security`) and real secrets before any real deployment.
- Do not enter real personal or health data.

## Licence

[MIT](LICENSE) © Hamad Ur Rehman
