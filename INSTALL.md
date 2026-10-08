# Installing and running CareDocs AI on your computer

This guide assumes no prior setup. It takes about **10 minutes**, most of it
downloading Python packages. Works on **Windows, macOS and Linux**. No GPU, no
paid services and no API key are needed.

---

## 1. What you need first

| Requirement | Version | How to check | Where to get it |
|---|---|---|---|
| Python | **3.11 or newer** | `python --version` (macOS/Linux: `python3 --version`) | <https://www.python.org/downloads/> |
| pip | comes with Python | `python -m pip --version` | — |
| Git *(optional)* | any | `git --version` | <https://git-scm.com/downloads> |
| Disk space | ~500 MB | — | packages + generated data |

> **Windows:** when installing Python, tick **"Add Python to PATH"** on the
> first screen. If you forgot, re-run the installer and choose *Modify*.

---

## 2. Download the project

**Option A — with Git**

```bash
git clone https://github.com/hamad470/caredocs-ai.git
cd caredocs-ai
```

**Option B — without Git**

1. On the GitHub page click the green **Code** button → **Download ZIP**.
2. Unzip it, then open a terminal in the unzipped folder
   (Windows: open the folder in File Explorer, type `cmd` in the address bar and
   press Enter).

---

## 3. Create a virtual environment (recommended)

This keeps the project's libraries separate from anything else on your computer.

**Windows (Command Prompt)**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses with a "running scripts is disabled" error, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. From here on, `python` means the
project's Python on every operating system.

---

## 4. Install the required libraries

All required libraries are listed in **`requirements.txt`**. Install them in one
command:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

What gets installed and why:

| Library | Used for |
|---|---|
| `flask` | the web application |
| `reportlab` | PDF report export |
| `openpyxl` | Excel (.xlsx) export |
| `scikit-learn` | fall-risk model, TF-IDF and LSA search |
| `numpy` | numerical work in retrieval and modelling |
| `faiss-cpu` | fast vector search *(optional — falls back automatically if missing)* |
| `rouge-score` | ROUGE-L evaluation metric *(optional — pure-Python fallback built in)* |

Tested with Python 3.12 and: Flask 3.1, reportlab 4.4, openpyxl 3.1,
scikit-learn 1.8, numpy 2.4, faiss-cpu 1.15, rouge-score 0.1.2.

---

## 5. Build the data

The database is not stored in the repository; it is generated on your machine
from a fixed random seed, so everyone gets exactly the same synthetic data.

```bash
python setup_project.py --skip-train
```

This takes about a minute and creates `carehome.db` plus two search indexes. The
trained fall-risk model (`ml_fall_risk_model.pkl`) is already included, which is
why training is skipped. To retrain it as well (~4 minutes), drop `--skip-train`.

---

## 6. Start the app

```bash
python app.py
```

Open **<http://127.0.0.1:5000>** in your browser and sign in:

| Role | Username | Password |
|---|---|---|
| Manager (sees everything) | `manager1` | `manager123` |
| Senior carer | `senior1` | `senior123` |
| Carer | `carer1` | `carer123` |

Stop the server with **Ctrl + C** in the terminal.

> **Windows shortcut:** double-click **`run.bat`**. It installs the libraries,
> builds the data on first run and starts the app in one go.

### Things to try

- **Residents** → open a resident → add a care note and click *Generate* to
  draft the narrative.
- **Risk Ranking** → residents ordered by predicted 7-day fall risk.
- **Ask the Records** (chat) → e.g. *"How many falls in the last 6 months?"* or
  *"Is Ethel losing weight?"*
- Any record → export to **PDF** or **Excel**.

---

## 7. Turn on AI writing (optional, free)

Without a key the app still works: every generated section uses an offline
template. For AI-written narratives:

1. Get a free Google Gemini key at <https://aistudio.google.com/apikey>
   (no card needed).
2. In the app go to **Settings → AI Settings**, paste the key and save.
   *Or* set it before starting the app:
   - Windows: `set GEMINI_API_KEYS=your_key`
   - macOS/Linux: `export GEMINI_API_KEYS=your_key`
3. Check it works: `python check_ai.py`

Keys saved in the app are stored in `ai_config.json`, which is git-ignored so it
is never uploaded. Never paste a key into any file you commit.

---

## 8. Run the tests (optional)

```bash
python test_project.py      # 29 tests, ~5 minutes
python test_chat_rag.py     # 40 checks, ~1 minute
```

On macOS/Linux you can force offline mode for the second suite with
`AI_PROVIDER=template python test_chat_rag.py`.

---

## Next time

You only do steps 1–5 once. Afterwards:

```bash
cd caredocs-ai
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
python app.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` (Windows) | Python is not on PATH. Re-run the installer, choose *Modify*, tick *Add Python to environment variables*. Or try `py` instead of `python`. |
| `python: command not found` (macOS/Linux) | Use `python3` instead. |
| `faiss-cpu` fails to install | It is optional. Remove that line from `requirements.txt`, install again, and the app uses its built-in search instead. |
| `No module named flask` | The virtual environment is not active. Run the activate command from step 3. |
| `carehome.db not found` / empty pages | Run `python setup_project.py --skip-train`. |
| `Address already in use` / port 5000 busy (common on macOS, AirPlay uses it) | Use another port: Windows `set PORT=5050`, macOS/Linux `export PORT=5050`, then `python app.py` and open <http://127.0.0.1:5050>. |
| Start over with fresh data | `python setup_project.py --force --skip-train` |
| AI says keys configured but none answered | Run `python check_ai.py` for the reason. Free keys have a daily limit; the app falls back to templates meanwhile. |

---

> **Reminder:** all data in this project is synthetic. Do not enter real
> resident or health information — this is a research MVP, not a certified
> clinical system.
